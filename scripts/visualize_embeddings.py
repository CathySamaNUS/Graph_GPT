#!/usr/bin/env python3
"""
PCA-based embedding visualization for Sequence-Augmented GraphGPT.

Compares node/edge embeddings across experiments:
  - GraphGPT only (graph structure)
  - ESM-2 only (protein sequence)
  - GraphGPT + ESM-2 (combined)
  - GraphGPT + random embedding (control)

Reuses the torch-SVD PCA pattern from new/4_pca_reduce.py.

Usage examples:

  # 1. Compare node embeddings across models
  python scripts/visualize_embeddings.py compare-nodes \
      --dirs baseline=exp/embeddings/baseline \
             esm2_only=exp/embeddings/esm2_only \
             combined=exp/embeddings/combined \
      --output_dir figures/pca_comparison

  # 2. Low-degree protein analysis
  python scripts/visualize_embeddings.py low-degree \
      --dirs baseline=exp/embeddings/baseline \
             combined=exp/embeddings/combined \
      --degree_threshold 10 \
      --output_dir figures/low_degree

  # 3. Edge separability analysis
  python scripts/visualize_embeddings.py edge-separability \
      --dirs baseline=exp/embeddings/baseline \
             combined=exp/embeddings/combined \
      --edge_method hadamard \
      --output_dir figures/edge_sep

  # 4. Visualize raw ESM-2 input embeddings
  python scripts/visualize_embeddings.py esm2-input \
      --embed_path data/OGB/ogbl_ppa_human/node_x_embed.pt \
      --mask_path data/OGB/ogbl_ppa_human/node_has_esm.pt \
      --degree_path exp/embeddings/baseline/node_degree.pt \
      --output_dir figures/esm2_input
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

# ---------------------------------------------------------------------------
# PCA (reuses the torch-SVD pattern from new/4_pca_reduce.py)
# ---------------------------------------------------------------------------

def pca_reduce(X: torch.Tensor, n_components: int = 2) -> Tuple[torch.Tensor, dict]:
    """
    PCA via torch SVD — same approach as new/4_pca_reduce.py.

    Args:
        X: [N, D] float tensor
        n_components: number of components to keep

    Returns:
        X_proj: [N, n_components] projected data
        info: dict with mean, V, singular values, variance explained
    """
    X = X.float()
    mean = X.mean(dim=0)
    X_centered = X - mean
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    V_k = Vt[:n_components].T  # [D, k]
    X_proj = X_centered @ V_k  # [N, k]

    total_var = (S ** 2).sum().item()
    explained_var = (S[:n_components] ** 2).sum().item()
    var_ratio = explained_var / total_var if total_var > 0 else 0.0

    info = {
        "mean": mean,
        "V": V_k,
        "S": S,
        "variance_explained": var_ratio,
        "n_components": n_components,
    }
    return X_proj, info


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {
    "GraphGPT Only":    "#2196F3",  # blue
    "ESM-2 Only":       "#FF9800",  # orange
    "GraphGPT+ESM-2":   "#4CAF50",  # green
    "GraphGPT+Random":  "#9C27B0",  # purple
    "baseline":         "#2196F3",
    "esm2_only":        "#FF9800",
    "combined":         "#4CAF50",
    "random":           "#9C27B0",
}

LABELS = {
    "baseline":   "GraphGPT Only",
    "esm2_only":  "ESM-2 Only",
    "combined":   "GraphGPT+ESM-2",
    "random":     "GraphGPT+Random",
}


def _get_color(name: str) -> str:
    return COLORS.get(name, COLORS.get(LABELS.get(name, ""), "#666666"))


def _get_label(name: str) -> str:
    return LABELS.get(name, name)


def degree_buckets(degree: torch.Tensor, thresholds: List[int] = None):
    """Assign degree bucket labels. Returns bucket_ids and bucket_names."""
    if thresholds is None:
        thresholds = [5, 20, 100, 500]
    deg = degree.numpy()
    bucket_ids = np.zeros(len(deg), dtype=int)
    names = [f"≤{thresholds[0]}"]
    for i, t in enumerate(thresholds):
        bucket_ids[deg > t] = i + 1
    for i in range(len(thresholds) - 1):
        names.append(f"{thresholds[i]+1}–{thresholds[i+1]}")
    names.append(f">{thresholds[-1]}")
    return bucket_ids, names


DEGREE_CMAP = ["#d32f2f", "#ff9800", "#fdd835", "#66bb6a", "#1565c0"]


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Core visualization functions
# ---------------------------------------------------------------------------

def plot_pca_scatter(
    projections: Dict[str, torch.Tensor],
    title: str,
    output_path: str,
    alpha: float = 0.4,
    s: float = 6,
    pca_infos: Dict[str, dict] = None,
):
    """Overlay PCA scatter plots for multiple experiments."""
    n = len(projections)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for idx, (name, proj) in enumerate(projections.items()):
        ax = axes[0, idx]
        p = proj.numpy()
        color = _get_color(name)
        ax.scatter(p[:, 0], p[:, 1], c=color, alpha=alpha, s=s, edgecolors="none")
        label = _get_label(name)
        var_str = ""
        if pca_infos and name in pca_infos:
            var_str = f" (var={pca_infos[name]['variance_explained']:.1%})"
        ax.set_title(f"{label}{var_str}", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_pca_overlay(
    projections: Dict[str, torch.Tensor],
    title: str,
    output_path: str,
    alpha: float = 0.3,
    s: float = 6,
):
    """Single plot with all experiments overlaid."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proj in projections.items():
        p = proj.numpy()
        ax.scatter(p[:, 0], p[:, 1], c=_get_color(name), alpha=alpha, s=s,
                   edgecolors="none", label=_get_label(name))
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_degree_colored(
    proj: torch.Tensor,
    degree: torch.Tensor,
    title: str,
    output_path: str,
    alpha: float = 0.5,
    s: float = 8,
):
    """PCA scatter colored by node degree bucket."""
    bucket_ids, bucket_names = degree_buckets(degree)
    p = proj.numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    for bid in range(len(bucket_names)):
        mask = bucket_ids == bid
        if mask.sum() == 0:
            continue
        ax.scatter(p[mask, 0], p[mask, 1], c=DEGREE_CMAP[bid % len(DEGREE_CMAP)],
                   alpha=alpha, s=s, edgecolors="none",
                   label=f"deg {bucket_names[bid]} (n={mask.sum()})")
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_low_degree_comparison(
    projections: Dict[str, torch.Tensor],
    degrees: Dict[str, torch.Tensor],
    threshold: int,
    output_path: str,
):
    """Compare low-degree node embeddings across models."""
    n = len(projections)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for idx, (name, proj) in enumerate(projections.items()):
        ax = axes[0, idx]
        p = proj.numpy()
        deg = degrees[name].numpy()
        low = deg <= threshold
        high = ~low

        ax.scatter(p[high, 0], p[high, 1], c="#cccccc", alpha=0.2, s=4,
                   edgecolors="none", label=f"deg>{threshold}")
        ax.scatter(p[low, 0], p[low, 1], c=_get_color(name), alpha=0.6, s=10,
                   edgecolors="none", label=f"deg≤{threshold} (n={low.sum()})")
        ax.legend(fontsize=8, markerscale=2)
        ax.set_title(f"{_get_label(name)}", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Low-Degree Proteins (degree ≤ {threshold})", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, output_path)


def build_edge_embeddings(
    hidden_states: torch.Tensor,
    edges: torch.Tensor,
    method: str = "hadamard",
) -> torch.Tensor:
    """
    Build edge embeddings from node hidden states.

    Args:
        hidden_states: [N_edges, hidden_dim] where each pair of consecutive
                       rows is (src, dst) for one edge, OR a node embedding
                       matrix indexed by edges
        edges: [N, 2] edge index
        method: 'hadamard' (element-wise product), 'concat', or 'absdiff'
    """
    # hidden_states from inference are per-edge (src,dst pair -> one hidden state)
    # But if we have per-node embeddings, build edge from pairs
    if edges is not None and hidden_states.shape[0] > edges.shape[0]:
        h_src = hidden_states[edges[:, 0]]
        h_dst = hidden_states[edges[:, 1]]
    else:
        # Assume hidden_states are already per-edge
        # For edge-level tasks, model outputs one hidden state per edge sample
        return hidden_states

    if method == "hadamard":
        return h_src * h_dst
    elif method == "concat":
        return torch.cat([h_src, h_dst], dim=1)
    elif method == "absdiff":
        return torch.abs(h_src - h_dst)
    else:
        raise ValueError(f"Unknown edge method: {method}")


def plot_edge_separability(
    pos_proj: torch.Tensor,
    neg_proj: torch.Tensor,
    title: str,
    output_path: str,
    alpha: float = 0.3,
    s: float = 6,
):
    """PCA scatter of positive vs negative edge embeddings."""
    fig, ax = plt.subplots(figsize=(7, 6))
    pp = pos_proj.numpy()
    np_ = neg_proj.numpy()
    ax.scatter(np_[:, 0], np_[:, 1], c="#ef5350", alpha=alpha, s=s,
               edgecolors="none", label=f"Negative (n={len(np_)})")
    ax.scatter(pp[:, 0], pp[:, 1], c="#42a5f5", alpha=alpha, s=s,
               edgecolors="none", label=f"Positive (n={len(pp)})")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def load_embeddings(dirs: Dict[str, str], filename: str):
    """Load a .pt tensor from each experiment directory."""
    out = {}
    for name, d in dirs.items():
        path = os.path.join(d, filename)
        if os.path.exists(path):
            out[name] = torch.load(path, map_location="cpu")
        else:
            print(f"  Warning: {path} not found, skipping {name}")
    return out


def parse_dirs(dir_args: List[str]) -> Dict[str, str]:
    """Parse 'name=path' arguments into a dict."""
    out = {}
    for arg in dir_args:
        if "=" in arg:
            name, path = arg.split("=", 1)
        else:
            name = Path(arg).stem
            path = arg
        out[name] = path
    return out


def cmd_compare_nodes(args):
    """Compare node embeddings across models via PCA."""
    dirs = parse_dirs(args.dirs)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load hidden states
    hidden = load_embeddings(dirs, "hidden_states.pt")
    if not hidden:
        print("No hidden_states.pt found. Trying esm2_input_embeds.pt ...")
        hidden = load_embeddings(dirs, "esm2_input_embeds.pt")
    if not hidden:
        print("No embeddings found.")
        return

    # Align sample counts (use min across all)
    min_n = min(h.shape[0] for h in hidden.values())
    for k in hidden:
        hidden[k] = hidden[k][:min_n]
    print(f"Using {min_n} samples per model")

    # PCA each model independently
    projections = {}
    pca_infos = {}
    for name, h in hidden.items():
        proj, info = pca_reduce(h, n_components=2)
        projections[name] = proj
        pca_infos[name] = info
        print(f"  {_get_label(name)}: shape={h.shape}, "
              f"var_explained={info['variance_explained']:.4f}")

    # Side-by-side
    plot_pca_scatter(projections, "Node Embedding PCA — Model Comparison",
                     os.path.join(args.output_dir, "node_pca_sidebyside.png"),
                     pca_infos=pca_infos)

    # Overlay
    # For overlay, project all onto the same PCA space (fit on concatenated data)
    all_h = torch.vstack(list(hidden.values()))
    all_proj, all_info = pca_reduce(all_h, n_components=2)
    start = 0
    shared_proj = {}
    for name, h in hidden.items():
        n = h.shape[0]
        shared_proj[name] = all_proj[start:start + n]
        start += n
    plot_pca_overlay(shared_proj, "Node Embedding PCA — Overlay (shared axes)",
                     os.path.join(args.output_dir, "node_pca_overlay.png"))

    # Degree-colored if degree available
    degrees = load_embeddings(dirs, "node_degree.pt")
    if degrees:
        first_name = list(hidden.keys())[0]
        if first_name in degrees:
            # Use sampled_nodes to index into degree
            sampled_nodes = {}
            for name, d in dirs.items():
                sn_path = os.path.join(d, "sampled_nodes.pt")
                if os.path.exists(sn_path):
                    sampled_nodes[name] = torch.load(sn_path, map_location="cpu")

            for name in hidden:
                if name in degrees and name in projections:
                    deg = degrees[name]
                    if name in sampled_nodes and len(deg) > projections[name].shape[0]:
                        deg = deg[sampled_nodes[name]]
                    if len(deg) >= projections[name].shape[0]:
                        deg = deg[:projections[name].shape[0]]
                        plot_degree_colored(
                            projections[name], deg,
                            f"Node PCA — {_get_label(name)} (colored by degree)",
                            os.path.join(args.output_dir,
                                         f"node_pca_degree_{name}.png"))

    # Save metadata
    meta = {
        "command": "compare-nodes",
        "dirs": dirs,
        "num_samples": min_n,
        "pca_variance": {k: v["variance_explained"] for k, v in pca_infos.items()},
    }
    with open(os.path.join(args.output_dir, "viz_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nAll figures saved to {args.output_dir}")


def cmd_low_degree(args):
    """Compare low-degree protein embeddings across models."""
    dirs = parse_dirs(args.dirs)
    os.makedirs(args.output_dir, exist_ok=True)

    hidden = load_embeddings(dirs, "hidden_states.pt")
    degrees_all = load_embeddings(dirs, "node_degree.pt")
    sampled_nodes_all = load_embeddings(dirs, "sampled_nodes.pt")

    if not hidden:
        print("No hidden_states.pt found.")
        return

    min_n = min(h.shape[0] for h in hidden.values())
    for k in hidden:
        hidden[k] = hidden[k][:min_n]

    # Get per-sample degree
    degrees = {}
    for name in hidden:
        if name in degrees_all:
            deg = degrees_all[name]
            if name in sampled_nodes_all and len(deg) > min_n:
                deg = deg[sampled_nodes_all[name]]
            degrees[name] = deg[:min_n]

    if not degrees:
        print("No degree data found.")
        return

    # PCA each
    projections = {}
    for name, h in hidden.items():
        proj, _ = pca_reduce(h, n_components=2)
        projections[name] = proj

    plot_low_degree_comparison(
        projections, degrees, args.degree_threshold,
        os.path.join(args.output_dir, f"low_degree_le{args.degree_threshold}.png"))

    # Also show degree distribution comparison for low-degree nodes
    fig, axes = plt.subplots(1, len(hidden), figsize=(5 * len(hidden), 4), squeeze=False)
    for idx, (name, proj) in enumerate(projections.items()):
        ax = axes[0, idx]
        deg = degrees[name].numpy()
        low = deg <= args.degree_threshold
        # Show spread (std of PC1, PC2) for low vs high degree
        p = proj.numpy()
        spread_low = np.std(p[low], axis=0) if low.sum() > 1 else [0, 0]
        spread_high = np.std(p[~low], axis=0) if (~low).sum() > 1 else [0, 0]
        ax.bar(["PC1 low", "PC2 low", "PC1 high", "PC2 high"],
               [spread_low[0], spread_low[1], spread_high[0], spread_high[1]],
               color=[_get_color(name)] * 2 + ["#cccccc"] * 2, alpha=0.7)
        ax.set_title(f"{_get_label(name)}", fontsize=11)
        ax.set_ylabel("Std dev in PCA space")
    fig.suptitle(f"Embedding Spread: Low-Degree (≤{args.degree_threshold}) vs High-Degree",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(args.output_dir,
                               f"degree_spread_le{args.degree_threshold}.png"))

    meta = {
        "command": "low-degree",
        "dirs": dirs,
        "degree_threshold": args.degree_threshold,
        "num_samples": min_n,
    }
    with open(os.path.join(args.output_dir, "viz_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nAll figures saved to {args.output_dir}")


def cmd_edge_separability(args):
    """Compare positive vs negative edge embedding separability."""
    dirs = parse_dirs(args.dirs)
    os.makedirs(args.output_dir, exist_ok=True)

    hidden = load_embeddings(dirs, "hidden_states.pt")
    logits = load_embeddings(dirs, "logits.pt")

    if not hidden or not logits:
        print("Need both hidden_states.pt and logits.pt")
        return

    for name in hidden:
        if name not in logits:
            continue
        h = hidden[name]
        l = logits[name]

        # Use logits to determine predicted pos/neg
        # logits shape: [N, 2] for binary classification
        preds = l.argmax(dim=1) if l.dim() > 1 else (l > 0).long()
        pos_mask = preds == 1
        neg_mask = preds == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            print(f"  {name}: all same prediction, skipping")
            continue

        # PCA on all, then split
        proj, info = pca_reduce(h, n_components=2)
        plot_edge_separability(
            proj[pos_mask], proj[neg_mask],
            f"Edge Separability — {_get_label(name)} "
            f"(var={info['variance_explained']:.1%})",
            os.path.join(args.output_dir, f"edge_sep_{name}.png"))
        print(f"  {_get_label(name)}: pos={pos_mask.sum()}, neg={neg_mask.sum()}")

    meta = {
        "command": "edge-separability",
        "dirs": dirs,
        "edge_method": args.edge_method,
    }
    with open(os.path.join(args.output_dir, "viz_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nAll figures saved to {args.output_dir}")


def cmd_esm2_input(args):
    """Visualize raw ESM-2 input embeddings with PCA."""
    os.makedirs(args.output_dir, exist_ok=True)

    embeds = torch.load(args.embed_path, map_location="cpu")
    print(f"ESM-2 embeddings: {embeds.shape}")

    mask = None
    if args.mask_path and os.path.exists(args.mask_path):
        mask = torch.load(args.mask_path, map_location="cpu").bool()
        print(f"Coverage: {mask.sum()}/{len(mask)} ({mask.float().mean():.1%})")

    degree = None
    if args.degree_path and os.path.exists(args.degree_path):
        degree = torch.load(args.degree_path, map_location="cpu")

    # PCA on covered nodes only
    if mask is not None:
        X = embeds[mask]
        deg_subset = degree[mask] if degree is not None else None
    else:
        X = embeds
        deg_subset = degree

    # Subsample if too many
    n = X.shape[0]
    max_n = args.max_nodes
    if n > max_n:
        torch.manual_seed(args.seed)
        perm = torch.randperm(n)[:max_n]
        X = X[perm]
        if deg_subset is not None:
            deg_subset = deg_subset[perm]
        print(f"Subsampled to {max_n} nodes")

    proj, info = pca_reduce(X, n_components=2)
    print(f"PCA variance explained: {info['variance_explained']:.4f}")

    # Basic scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    p = proj.numpy()
    ax.scatter(p[:, 0], p[:, 1], c="#FF9800", alpha=0.3, s=4, edgecolors="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"ESM-2 Input Embeddings (PCA, var={info['variance_explained']:.1%})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_fig(fig, os.path.join(args.output_dir, "esm2_input_pca.png"))

    # Degree-colored
    if deg_subset is not None:
        plot_degree_colored(proj, deg_subset,
                           "ESM-2 Input Embeddings — Colored by Node Degree",
                           os.path.join(args.output_dir, "esm2_input_pca_degree.png"))

    meta = {
        "command": "esm2-input",
        "embed_path": args.embed_path,
        "shape": list(embeds.shape),
        "num_plotted": int(X.shape[0]),
        "seed": args.seed,
        "pca_variance_explained": info["variance_explained"],
    }
    with open(os.path.join(args.output_dir, "viz_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nAll figures saved to {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCA embedding visualization for Sequence-Augmented GraphGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # compare-nodes
    p1 = sub.add_parser("compare-nodes",
                        help="Compare node embeddings across models")
    p1.add_argument("--dirs", nargs="+", required=True,
                    help="name=path pairs for experiment embedding dirs")
    p1.add_argument("--output_dir", default="figures/pca_comparison")

    # low-degree
    p2 = sub.add_parser("low-degree",
                        help="Analyze low-degree protein embeddings")
    p2.add_argument("--dirs", nargs="+", required=True)
    p2.add_argument("--degree_threshold", type=int, default=10)
    p2.add_argument("--output_dir", default="figures/low_degree")

    # edge-separability
    p3 = sub.add_parser("edge-separability",
                        help="Compare positive vs negative edge separability")
    p3.add_argument("--dirs", nargs="+", required=True)
    p3.add_argument("--edge_method", default="hadamard",
                    choices=["hadamard", "concat", "absdiff"])
    p3.add_argument("--output_dir", default="figures/edge_sep")

    # esm2-input
    p4 = sub.add_parser("esm2-input",
                        help="Visualize raw ESM-2 input embeddings")
    p4.add_argument("--embed_path", required=True)
    p4.add_argument("--mask_path", default=None)
    p4.add_argument("--degree_path", default=None)
    p4.add_argument("--max_nodes", type=int, default=5000)
    p4.add_argument("--seed", type=int, default=42)
    p4.add_argument("--output_dir", default="figures/esm2_input")

    args = parser.parse_args()

    if args.command == "compare-nodes":
        cmd_compare_nodes(args)
    elif args.command == "low-degree":
        cmd_low_degree(args)
    elif args.command == "edge-separability":
        cmd_edge_separability(args)
    elif args.command == "esm2-input":
        cmd_esm2_input(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
