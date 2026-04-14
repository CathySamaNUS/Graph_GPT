#!/usr/bin/env python3
"""
Extract node embeddings from trained GraphGPT checkpoints for visualization.

Runs inference on the test set and saves task_hidden_states (the learned
node representations before the classification head).

Usage:
    python scripts/extract_embeddings.py \
        --exp_dir exp/models/ogbl_ppa_human/baseline_tiny_h128_l2_b128_e6_lr3e-4 \
        --epoch 5 \
        --output_dir exp/embeddings/baseline \
        --data_dir ./data/OGB \
        --data_path ogbl_ppa_human \
        --num_samples 2000 \
        --seed 42
"""
import argparse
import os
import sys
import json
import random

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Extract embeddings from trained GraphGPT")
    p.add_argument("--exp_dir", required=True, help="Experiment output directory")
    p.add_argument("--epoch", type=int, default=-1, help="Epoch to load (-1 = latest)")
    p.add_argument("--output_dir", required=True, help="Where to save extracted embeddings")
    p.add_argument("--data_dir", default="./data/OGB", help="Data directory")
    p.add_argument("--data_path", default="ogbl_ppa_human", help="Dataset sub-path")
    p.add_argument("--num_samples", type=int, default=2000,
                    help="Number of edges to sample for embedding extraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def load_graph_and_degree(data_dir, data_path):
    """Load the graph and compute node degree from training edges."""
    dataset_root = os.path.join(data_dir, data_path)
    graph = torch.load(os.path.join(dataset_root, "graph.pt"),
                       map_location="cpu", weights_only=False)
    split_edge = torch.load(os.path.join(dataset_root, "split_edge.pt"),
                            map_location="cpu", weights_only=False)

    num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph['num_nodes']

    # Compute degree from training edges
    if isinstance(graph, dict):
        edge_index = graph['edge_index']
    else:
        edge_index = graph.edge_index

    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))
    degree.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.long))

    return graph, split_edge, degree


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading graph from {args.data_dir}/{args.data_path} ...")
    graph, split_edge, degree = load_graph_and_degree(args.data_dir, args.data_path)
    num_nodes = degree.shape[0]
    print(f"Nodes: {num_nodes}, Degree range: [{degree.min()}, {degree.max()}], "
          f"Mean: {degree.float().mean():.1f}")

    # Save degree for visualization
    torch.save(degree, os.path.join(args.output_dir, "node_degree.pt"))

    # Load ESM-2 input embeddings if available
    dataset_root = os.path.join(args.data_dir, args.data_path)
    esm_embed_path = os.path.join(dataset_root, "node_x_embed.pt")
    esm_mask_path = os.path.join(dataset_root, "node_has_esm.pt")
    if os.path.exists(esm_embed_path):
        esm_embeds = torch.load(esm_embed_path, map_location="cpu")
        print(f"Loaded ESM-2 input embeddings: {esm_embeds.shape}")
        torch.save(esm_embeds, os.path.join(args.output_dir, "esm2_input_embeds.pt"))
    if os.path.exists(esm_mask_path):
        esm_mask = torch.load(esm_mask_path, map_location="cpu")
        torch.save(esm_mask, os.path.join(args.output_dir, "esm2_mask.pt"))

    # Load model config and checkpoint
    config_path = os.path.join(args.exp_dir, "config.yaml")
    if not os.path.exists(config_path):
        print(f"No config.yaml found at {config_path}, skipping model inference.")
        print("Saved degree and ESM-2 input embeddings only.")
        _save_metadata(args, num_nodes, degree)
        return

    # Find checkpoint
    if args.epoch == -1:
        # Find latest epoch
        epoch_dirs = [d for d in os.listdir(args.exp_dir)
                      if d.startswith("epoch_") and d != "epoch_-1"]
        if not epoch_dirs:
            print("No epoch checkpoints found.")
            _save_metadata(args, num_nodes, degree)
            return
        epochs = sorted([int(d.split("_")[1]) for d in epoch_dirs])
        args.epoch = epochs[-1]

    ckp_dir = os.path.join(args.exp_dir, f"epoch_{args.epoch}")
    model_path = os.path.join(ckp_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"No model.pt at {model_path}, skipping model inference.")
        _save_metadata(args, num_nodes, degree)
        return

    print(f"Loading model from epoch {args.epoch}: {model_path}")

    # Import GraphGPT modules
    from src.conf.config import Config
    from src.models.graphgpt.modeling_graphgpt import GraphGPTTaskModel
    from src.data.tokenizer import StackedGSTTokenizer
    from src.data.dataset_map import ShaDowKHopSeqFromEdgesMapDataset
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.load(config_path)
    cfg = Config(**cfg_dict)

    # Build model
    model_config_path = os.path.join(args.exp_dir, "config.json")
    from src.models.graphgpt.configuration_graphgpt import GraphGPTConfig
    model_cfg = GraphGPTConfig.from_pretrained(model_config_path)
    model = GraphGPTTaskModel(model_cfg)

    # Load weights
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    # Strip 'module.' prefix if present (from DDP)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Build tokenizer and test dataset
    gtokenizer = StackedGSTTokenizer(cfg.tokenization)

    # Load test edges
    test_edges = split_edge["test"]["edge"]
    n_test = min(args.num_samples, test_edges.shape[0])
    perm = torch.randperm(test_edges.shape[0])[:n_test]
    sampled_test_edges = test_edges[perm]
    print(f"Sampled {n_test} test edges for embedding extraction")

    # Save sampled edge indices for reproducibility
    torch.save(sampled_test_edges, os.path.join(args.output_dir, "sampled_edges.pt"))
    torch.save(perm, os.path.join(args.output_dir, "sampled_edge_perm.pt"))

    # Collect unique node IDs from sampled edges
    sampled_nodes = torch.unique(sampled_test_edges.reshape(-1))
    print(f"Unique nodes in sampled edges: {sampled_nodes.shape[0]}")
    torch.save(sampled_nodes, os.path.join(args.output_dir, "sampled_nodes.pt"))

    # Extract hidden states via inference
    from src.data.data_sources import _read_ogbl_ppa_like_dataset
    from torch.utils.data import DataLoader

    # Build full test dataset
    test_dataset = ShaDowKHopSeqFromEdgesMapDataset(
        graph=graph,
        edges=sampled_test_edges,
        labels=torch.ones(n_test, dtype=torch.long),  # dummy labels
        gtokenizer=gtokenizer,
        sampling_config=cfg.tokenization.data.sampling,
        is_training=False,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn if hasattr(test_dataset, 'collate_fn') else None,
    )

    # Run inference
    ls_hidden = []
    ls_logits = []
    ls_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            inputs_raw_embeds = None
            if "embed" in batch:
                inputs_raw_embeds = batch["embed"].to(device)
            res = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_raw_embeds=inputs_raw_embeds,
                position_ids=position_ids,
            )
            ls_hidden.append(res.task_hidden_states.cpu().float())
            ls_logits.append(res.task_logits.cpu().float())
            if "labels" in batch:
                ls_labels.append(batch["labels"].cpu())

    hidden_states = torch.vstack(ls_hidden)
    logits = torch.vstack(ls_logits)
    print(f"Extracted hidden states: {hidden_states.shape}")

    torch.save(hidden_states, os.path.join(args.output_dir, "hidden_states.pt"))
    torch.save(logits, os.path.join(args.output_dir, "logits.pt"))

    _save_metadata(args, num_nodes, degree)
    print(f"All embeddings saved to {args.output_dir}")


def _save_metadata(args, num_nodes, degree):
    meta = {
        "exp_dir": args.exp_dir,
        "epoch": args.epoch,
        "seed": args.seed,
        "num_nodes": int(num_nodes),
        "num_samples": args.num_samples,
        "degree_min": int(degree.min()),
        "degree_max": int(degree.max()),
        "degree_mean": round(float(degree.float().mean()), 2),
        "degree_median": round(float(degree.float().median()), 2),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
