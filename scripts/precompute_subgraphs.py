#!/usr/bin/env python3
"""Pre-compute and cache subgraphs for GraphGPT edge-level training.

This script pre-computes k-hop ego subgraphs for all edges in the dataset
(positive train/valid/test + optionally pre-sampled negative edges for N epochs).
Cached subgraphs are stored as chunked .pt files under the dataset directory.

Usage:
    python scripts/precompute_subgraphs.py \
        --data_dir ./data/OGB \
        --data_path ogbl_ppa_human \
        --depth_neighbors "[[1,14]]" \
        --neg_ratio 1 \
        --neg_epochs 10 \
        --chunk_size 10000 \
        --num_workers 8

The cached subgraphs are saved at:
    {data_dir}/{data_path}/cached_subgraphs/
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_sparse import SparseTensor
from tqdm import tqdm

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute subgraphs for GraphGPT")
    parser.add_argument("--data_dir", type=str, default="./data/OGB")
    parser.add_argument("--data_path", type=str, default="ogbl_ppa_human")
    parser.add_argument(
        "--depth_neighbors",
        type=str,
        default="[[1,14]]",
        help="JSON list of [depth, num_neighbors] pairs",
    )
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument(
        "--neg_epochs",
        type=int,
        default=10,
        help="Number of epochs of negative edges to pre-compute (0=skip)",
    )
    parser.add_argument("--chunk_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=0, help="0=sequential")
    parser.add_argument("--replace", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_graph_and_splits(data_dir: str, data_path: str):
    """Load graph and edge splits from disk."""
    dataset_root = os.path.join(data_dir, data_path)
    graph = torch.load(
        os.path.join(dataset_root, "graph.pt"), map_location="cpu", weights_only=False
    )
    split_edge = torch.load(
        os.path.join(dataset_root, "split_edge.pt"),
        map_location="cpu",
        weights_only=False,
    )
    # Add node IDs and transform x (same as data_sources.py)
    graph.id = torch.arange(graph.num_nodes)
    # Must match _get_global_local_id_from_onehot in data_sources.py exactly
    if graph.x is not None and graph.x.dim() == 2:
        x_gid = torch.argmax(graph.x, dim=-1, keepdim=True) + 1  # +1 matches data_sources.py
        graph.x_gid = x_gid.squeeze(-1)
        x_cum = torch.cumsum(graph.x, dim=0)
        idx = x_cum * graph.x
        local_id = idx.sum(dim=-1).view((-1, 1))
        graph.x = torch.cat([x_gid, local_id], dim=1)  # [N, 2]
    return graph, split_edge, dataset_root


def build_adj_t(graph: Data) -> SparseTensor:
    """Build SparseTensor adjacency (transposed) from edge_index."""
    row, col = graph.edge_index.cpu()
    adj_t = SparseTensor(
        row=row,
        col=col,
        value=torch.arange(col.size(0)),
        sparse_sizes=(graph.num_nodes, graph.num_nodes),
    ).t()
    return adj_t


def _remove_target_edge(edge_index, src, dst, bidirectional=True):
    """Remove the target edge from the subgraph to prevent leakage."""
    forward_bool = (edge_index[0] == src) & (edge_index[1] == dst)
    if bidirectional:
        backward_bool = (edge_index[0] == dst) & (edge_index[1] == src)
        all_bool = ~(forward_bool + backward_bool)
    else:
        all_bool = ~forward_bool
    return edge_index[:, all_bool], all_bool


def extract_subgraph(
    adj_t: SparseTensor,
    graph: Data,
    src: int,
    dst: int,
    y: int,
    depth: int,
    num_neighbors: int,
    replace: bool,
    idx: int,
) -> Data:
    """Extract a k-hop ego subgraph around (src, dst) — mirrors ShaDowKHopSeqFromEdgesMapDataset.__getitem__."""
    seed_node_ids = torch.tensor([src, dst], dtype=torch.int64)

    rowptr, col, _ = adj_t.csr()
    out = torch.ops.torch_sparse.ego_k_hop_sample_adj(
        rowptr, col, seed_node_ids, depth, num_neighbors, replace
    )
    n_id = out[2]
    n_id_unique = torch.unique(n_id)

    adj_sub, e_id = adj_t.saint_subgraph(n_id_unique)
    row_sub, col_sub, _ = adj_sub.t().coo()
    edge_index = torch.vstack([row_sub, col_sub])
    edge_mask = e_id

    root_n_id_src = (n_id_unique == src).nonzero(as_tuple=True)[0]
    root_n_id_dst = (n_id_unique == dst).nonzero(as_tuple=True)[0]
    root_n_id = torch.tensor([root_n_id_src, root_n_id_dst], dtype=torch.int64)

    data = Data(num_nodes=n_id_unique.numel())
    data.root_n_id = root_n_id
    data.seed_node = seed_node_ids

    # Remove target edge (non-pretrain mode)
    edge_index, non_tgt_mask = _remove_target_edge(
        edge_index, root_n_id_src, root_n_id_dst
    )
    data.y = torch.tensor([y], dtype=torch.int64)
    data.edge_index = edge_index

    # Copy node/edge features
    for k, v in graph:
        if k in ["edge_index", "adj_t", "num_nodes", "batch", "ptr", "y"]:
            continue
        if isinstance(v, Tensor) and v.size(0) == graph.num_nodes:
            data[k] = v[n_id_unique]
        elif isinstance(v, Tensor) and v.size(0) == graph.num_edges:
            if edge_mask.numel() == 0 or non_tgt_mask.numel() == 0:
                data[k] = torch.empty((0,) + v.shape[1:], dtype=v.dtype)
            else:
                data[k] = v[edge_mask][non_tgt_mask]
        else:
            data[k] = v

    data.idx = idx
    return data


def compute_subgraphs_for_edges(
    adj_t: SparseTensor,
    graph: Data,
    edges_with_y: torch.Tensor,
    depth: int,
    num_neighbors: int,
    replace: bool,
    desc: str = "Computing subgraphs",
) -> List[Data]:
    """Compute subgraphs for a batch of edges. Returns list of Data objects."""
    results = []
    for i in tqdm(range(len(edges_with_y)), desc=desc):
        src, dst, y = edges_with_y[i].tolist()
        data = extract_subgraph(
            adj_t, graph, int(src), int(dst), int(y), depth, num_neighbors, replace, i
        )
        results.append(data)
    return results


def save_chunks(
    subgraphs: List[Data], output_dir: str, prefix: str, chunk_size: int
):
    """Save subgraphs as chunked .pt files."""
    os.makedirs(output_dir, exist_ok=True)
    num_chunks = (len(subgraphs) + chunk_size - 1) // chunk_size
    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, len(subgraphs))
        chunk = subgraphs[start:end]
        fn = os.path.join(output_dir, f"{prefix}_chunk_{c:03d}.pt")
        torch.save(chunk, fn)
    print(f"  Saved {len(subgraphs)} subgraphs in {num_chunks} chunks to {output_dir}")


def sample_neg_edges(
    pos_edges: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    neg_ratio: int,
) -> Tensor:
    """Sample negative edges globally."""
    self_looped_edge_index, _ = add_self_loops(edge_index)
    cnt_neg = neg_ratio * pos_edges.shape[0]
    neg_edges = negative_sampling(
        edge_index=self_looped_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=cnt_neg,
    ).T  # [N_e, 2]
    return neg_edges


def compute_config_hash(args) -> str:
    """Compute a hash of the sampling config for cache invalidation."""
    config = {
        "depth_neighbors": args.depth_neighbors,
        "neg_ratio": args.neg_ratio,
        "replace": args.replace,
    }
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    depth_neighbors = json.loads(args.depth_neighbors)
    assert len(depth_neighbors) == 1, (
        "Caching only supported with a single depth_neighbors entry (deterministic sampling). "
        f"Got: {depth_neighbors}"
    )
    depth, num_neighbors = depth_neighbors[0]

    dataset_root = os.path.join(args.data_dir, args.data_path)
    cache_dir = os.path.join(dataset_root, "cached_subgraphs")

    print(f"[{datetime.now()}] Loading graph from {dataset_root} ...")
    graph, split_edge, _ = load_graph_and_splits(args.data_dir, args.data_path)
    print(
        f"  Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, x.shape={graph.x.shape}"
    )

    print(f"[{datetime.now()}] Building adjacency matrix ...")
    adj_t = build_adj_t(graph)

    # Save metadata
    os.makedirs(cache_dir, exist_ok=True)
    metadata = {
        "data_path": args.data_path,
        "depth_neighbors": depth_neighbors,
        "neg_ratio": args.neg_ratio,
        "neg_epochs": args.neg_epochs,
        "replace": args.replace,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.edge_index.shape[1],
        "chunk_size": args.chunk_size,
        "config_hash": compute_config_hash(args),
        "created_at": str(datetime.now()),
    }

    # ========== Positive edges for train/valid/test ==========
    for split_name in ["train", "valid", "test"]:
        if split_name not in split_edge:
            print(f"  Skipping split '{split_name}' (not in split_edge)")
            continue

        split_dir = os.path.join(cache_dir, split_name)
        pos_edges = split_edge[split_name]["edge"]  # [N_p, 2]
        y_pos = torch.ones((pos_edges.shape[0], 1), dtype=torch.int64)
        edges_with_y = torch.cat([pos_edges, y_pos], dim=1)

        print(
            f"\n[{datetime.now()}] Processing {split_name} positive edges: {len(pos_edges)} ..."
        )
        subgraphs = compute_subgraphs_for_edges(
            adj_t,
            graph,
            edges_with_y,
            depth,
            num_neighbors,
            args.replace,
            desc=f"{split_name}/pos",
        )
        save_chunks(subgraphs, split_dir, "pos", args.chunk_size)
        metadata[f"{split_name}_pos_count"] = len(pos_edges)
        del subgraphs  # free memory

    # ========== Negative edges for train (pre-sampled for N epochs) ==========
    if args.neg_epochs > 0:
        train_pos_edges = split_edge["train"]["edge"]
        train_dir = os.path.join(cache_dir, "train")

        for ep in range(args.neg_epochs):
            print(
                f"\n[{datetime.now()}] Sampling negative edges for epoch {ep} ..."
            )
            torch.manual_seed(args.seed + ep * 1000)
            neg_edges = sample_neg_edges(
                train_pos_edges,
                graph.edge_index,
                graph.num_nodes,
                args.neg_ratio,
            )
            y_neg = torch.zeros((neg_edges.shape[0], 1), dtype=torch.int64)
            neg_edges_with_y = torch.cat([neg_edges, y_neg], dim=1)

            print(
                f"  Computing subgraphs for {len(neg_edges)} negative edges ..."
            )
            subgraphs = compute_subgraphs_for_edges(
                adj_t,
                graph,
                neg_edges_with_y,
                depth,
                num_neighbors,
                args.replace,
                desc=f"train/neg_epoch_{ep}",
            )
            save_chunks(subgraphs, train_dir, f"neg_epoch_{ep:02d}", args.chunk_size)
            del subgraphs

        # Also pre-compute negatives for valid and test splits
        for split_name in ["valid", "test"]:
            if split_name not in split_edge:
                continue
            split_data = split_edge[split_name]
            if "edge_neg" in split_data:
                # Use fixed negatives from the dataset
                neg_edges = split_data["edge_neg"]
            else:
                neg_edges = sample_neg_edges(
                    split_data["edge"],
                    graph.edge_index,
                    graph.num_nodes,
                    args.neg_ratio,
                )
            y_neg = torch.zeros((neg_edges.shape[0], 1), dtype=torch.int64)
            neg_edges_with_y = torch.cat([neg_edges, y_neg], dim=1)

            print(
                f"\n[{datetime.now()}] Computing {split_name} negative subgraphs: {len(neg_edges)} ..."
            )
            split_dir = os.path.join(cache_dir, split_name)
            subgraphs = compute_subgraphs_for_edges(
                adj_t,
                graph,
                neg_edges_with_y,
                depth,
                num_neighbors,
                args.replace,
                desc=f"{split_name}/neg",
            )
            save_chunks(subgraphs, split_dir, "neg", args.chunk_size)
            metadata[f"{split_name}_neg_count"] = len(neg_edges)
            del subgraphs

    metadata["neg_epochs_computed"] = args.neg_epochs

    # Save metadata
    meta_path = os.path.join(cache_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[{datetime.now()}] Metadata saved to {meta_path}")

    # Print summary
    total_size = 0
    for root, dirs, files in os.walk(cache_dir):
        for fn in files:
            total_size += os.path.getsize(os.path.join(root, fn))
    print(f"\n{'='*60}")
    print(f"Cache directory: {cache_dir}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
