import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch_geometric.utils import is_undirected


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify the filtered ogbl-ppa human-only dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/OGB/ogbl_ppa_human",
        help="Directory containing graph.pt, split_edge.pt and mapping files.",
    )
    parser.add_argument(
        "--taxon_prefix",
        type=str,
        default="9606.",
        help="Expected protein-id prefix for human proteins.",
    )
    return parser.parse_args()


def require_files(dataset_dir: Path) -> Dict[str, Path]:
    files = {
        "graph": dataset_dir / "graph.pt",
        "split_edge": dataset_dir / "split_edge.pt",
        "human_node_mask": dataset_dir / "human_node_mask.pt",
        "old_to_new": dataset_dir / "old_to_new_node_id.pt",
        "new_to_old": dataset_dir / "new_to_old_node_id.pt",
        "mapping": dataset_dir / "mapping" / "nodeidx2proteinid.csv.gz",
        "summary": dataset_dir / "summary.json",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    return files


def edge_hash_set(edge_tensor: torch.Tensor, num_nodes: int) -> set:
    return set(
        (
            edge_tensor[:, 0].to(torch.int64) * num_nodes + edge_tensor[:, 1].to(torch.int64)
        ).tolist()
    )


def check_edge_tensor(name: str, edge_tensor: torch.Tensor, num_nodes: int) -> Tuple[bool, str]:
    if not torch.is_tensor(edge_tensor):
        return False, f"{name} is not a tensor"
    if edge_tensor.ndim != 2 or edge_tensor.shape[1] != 2:
        return False, f"{name} expected shape [N, 2], got {tuple(edge_tensor.shape)}"
    in_range = bool(((edge_tensor >= 0) & (edge_tensor < num_nodes)).all())
    if not in_range:
        return False, f"{name} contains node ids outside [0, {num_nodes - 1}]"
    return True, f"{name} shape={tuple(edge_tensor.shape)}"


def load_mapping_rows(mapping_file: Path) -> List[Dict[str, str]]:
    with gzip.open(mapping_file, "rt", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def check(name: str, ok: bool, details: str, checks: List[Tuple[str, bool, str]]):
    checks.append((name, ok, details))


def main():
    args = parse_args()
    dataset_dir = Path(args.data_dir)
    files = require_files(dataset_dir)

    graph = load_pt(files["graph"])
    split_edge = load_pt(files["split_edge"])
    human_node_mask = load_pt(files["human_node_mask"]).to(torch.bool)
    old_to_new = load_pt(files["old_to_new"]).to(torch.long)
    new_to_old = load_pt(files["new_to_old"]).to(torch.long)
    mapping_rows = load_mapping_rows(files["mapping"])
    with files["summary"].open("r", encoding="utf-8") as fp:
        summary = json.load(fp)

    checks: List[Tuple[str, bool, str]] = []
    num_nodes = int(graph.num_nodes)
    num_edges = int(graph.edge_index.shape[1])

    check(
        "graph.edge_index_shape",
        graph.edge_index.ndim == 2 and graph.edge_index.shape[0] == 2,
        f"edge_index shape={tuple(graph.edge_index.shape)}",
        checks,
    )
    graph_edge_in_range = bool(((graph.edge_index >= 0) & (graph.edge_index < num_nodes)).all())
    check(
        "graph.edge_index_in_range",
        graph_edge_in_range,
        f"graph edges within [0, {num_nodes - 1}]",
        checks,
    )
    check(
        "graph.is_undirected",
        bool(is_undirected(graph.edge_index)),
        f"is_undirected={bool(is_undirected(graph.edge_index))}",
        checks,
    )

    check(
        "mask.sum_matches_num_nodes",
        int(human_node_mask.sum().item()) == num_nodes,
        f"mask_sum={int(human_node_mask.sum().item())}, num_nodes={num_nodes}",
        checks,
    )
    check(
        "new_to_old_length",
        new_to_old.numel() == num_nodes,
        f"new_to_old_len={new_to_old.numel()}, num_nodes={num_nodes}",
        checks,
    )
    check(
        "old_to_new_length",
        old_to_new.numel() == human_node_mask.numel(),
        f"old_to_new_len={old_to_new.numel()}, raw_nodes={human_node_mask.numel()}",
        checks,
    )

    inverse_ok = torch.equal(old_to_new[new_to_old], torch.arange(num_nodes, dtype=torch.long))
    check("node_mapping_inverse", inverse_ok, "old_to_new[new_to_old] matches [0..N-1]", checks)

    removed_nodes_ok = bool((old_to_new[~human_node_mask] == -1).all())
    check(
        "removed_nodes_marked_minus_one",
        removed_nodes_ok,
        "all removed old node ids map to -1",
        checks,
    )

    if hasattr(graph, "old_node_id"):
        graph_old_node_ok = torch.equal(graph.old_node_id.to(torch.long), new_to_old)
        check(
            "graph.old_node_id_matches_new_to_old",
            graph_old_node_ok,
            "graph.old_node_id equals new_to_old_node_id",
            checks,
        )

    split_specs = [
        ("train", "edge"),
        ("valid", "edge"),
        ("valid", "edge_neg"),
        ("test", "edge"),
        ("test", "edge_neg"),
    ]
    for split_name, key in split_specs:
        ok, details = check_edge_tensor(f"{split_name}.{key}", split_edge[split_name][key], num_nodes)
        check(f"{split_name}.{key}", ok, details, checks)

    graph_hash = edge_hash_set(graph.edge_index.t().contiguous(), num_nodes)
    train_hash = edge_hash_set(split_edge["train"]["edge"], num_nodes)
    train_in_graph = train_hash.issubset(graph_hash)
    check(
        "train_edges_in_graph",
        train_in_graph,
        f"train_edges={len(train_hash)}, graph_edges={len(graph_hash)}",
        checks,
    )

    train_pos = edge_hash_set(split_edge["train"]["edge"], num_nodes)
    valid_pos = edge_hash_set(split_edge["valid"]["edge"], num_nodes)
    test_pos = edge_hash_set(split_edge["test"]["edge"], num_nodes)
    positive_splits_disjoint = (
        train_pos.isdisjoint(valid_pos)
        and train_pos.isdisjoint(test_pos)
        and valid_pos.isdisjoint(test_pos)
    )
    check(
        "positive_splits_disjoint",
        positive_splits_disjoint,
        "train/valid/test positive edges are pairwise disjoint",
        checks,
    )

    mapping_rows_ok = len(mapping_rows) == num_nodes
    check(
        "mapping_row_count",
        mapping_rows_ok,
        f"mapping_rows={len(mapping_rows)}, num_nodes={num_nodes}",
        checks,
    )
    protein_ids = [row.get("protein_id", "") for row in mapping_rows]
    prefix_ok = all(pid.startswith(args.taxon_prefix) for pid in protein_ids if pid)
    check(
        "mapping_protein_prefix",
        prefix_ok,
        f"expected_prefix={args.taxon_prefix}",
        checks,
    )

    summary_checks = {
        "summary.filtered_num_nodes": summary.get("filtered_num_nodes") == num_nodes,
        "summary.filtered_num_edges": summary.get("filtered_num_edges") == num_edges,
        "summary.train_pos_edges": summary.get("train_pos_edges")
        == int(split_edge["train"]["edge"].shape[0]),
        "summary.valid_pos_edges": summary.get("valid_pos_edges")
        == int(split_edge["valid"]["edge"].shape[0]),
        "summary.valid_neg_edges": summary.get("valid_neg_edges")
        == int(split_edge["valid"]["edge_neg"].shape[0]),
        "summary.test_pos_edges": summary.get("test_pos_edges")
        == int(split_edge["test"]["edge"].shape[0]),
        "summary.test_neg_edges": summary.get("test_neg_edges")
        == int(split_edge["test"]["edge_neg"].shape[0]),
    }
    for name, ok in summary_checks.items():
        check(name, ok, "matches stored summary.json", checks)

    failures = [item for item in checks if not item[1]]
    for name, ok, details in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {details}")

    print()
    print(
        json.dumps(
            {
                "data_dir": str(dataset_dir),
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "num_failures": len(failures),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
