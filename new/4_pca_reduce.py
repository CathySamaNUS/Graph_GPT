#!/usr/bin/env python3
"""PCA reduce node_x_embed.pt from 1280-dim to 256-dim."""
import torch
import json
from pathlib import Path

DATA_DIR = "/oceanstor/home/e1553200/graph-gpt-main/data/OGB/ogbl_ppa_human"

# Load embeddings
x_embed = torch.load(f"{DATA_DIR}/node_x_embed.pt", map_location="cpu")
has_esm = torch.load(f"{DATA_DIR}/node_has_esm.pt", map_location="cpu")
print(f"Original shape: {x_embed.shape}")
print(f"Coverage: {has_esm.sum().item()}/{has_esm.shape[0]} ({has_esm.float().mean().item():.4f})")

# Backup original
torch.save(x_embed, f"{DATA_DIR}/node_x_embed_1280.pt")
torch.save(has_esm, f"{DATA_DIR}/node_has_esm_1280.pt")
print("Backed up originals as node_x_embed_1280.pt")

# PCA on nodes that have embeddings
mask = has_esm.bool()
X = x_embed[mask].float()  # [N_covered, 1280]
print(f"PCA input: {X.shape}")

# Center
mean = X.mean(dim=0)
X_centered = X - mean

# SVD for PCA
U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
V_256 = Vt[:256].T  # [1280, 256] projection matrix
print(f"Projection matrix: {V_256.shape}")

# Variance explained
total_var = (S ** 2).sum().item()
var_256 = (S[:256] ** 2).sum().item()
print(f"Variance explained by 256 components: {var_256/total_var:.4f} ({var_256/total_var*100:.1f}%)")

# Project all embeddings
x_pca = torch.zeros((x_embed.shape[0], 256), dtype=torch.float32)
x_pca[mask] = (X_centered @ V_256).float()

# L2 normalize
norms = x_pca.norm(dim=1, keepdim=True).clamp(min=1e-12)
x_pca[mask] = x_pca[mask] / norms[mask]

print(f"PCA output shape: {x_pca.shape}")
print(f"Norm range (covered): {x_pca[mask].norm(dim=1).min():.4f} - {x_pca[mask].norm(dim=1).max():.4f}")

# Save
torch.save(x_pca, f"{DATA_DIR}/node_x_embed.pt")
# has_esm unchanged
print(f"Saved PCA-reduced embeddings to {DATA_DIR}/node_x_embed.pt")

# Save PCA params for reference
torch.save({"mean": mean, "V_256": V_256, "S": S}, f"{DATA_DIR}/pca_params.pt")

stats = {
    "num_nodes": int(has_esm.shape[0]),
    "num_nodes_with_esm": int(has_esm.sum().item()),
    "num_nodes_without_esm": int((~has_esm).sum().item()),
    "coverage": round(has_esm.float().mean().item(), 6),
    "embed_dim": 256,
    "original_embed_dim": 1280,
    "pca_variance_explained": round(var_256/total_var, 6),
}
with open(f"{DATA_DIR}/node_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"Stats: {json.dumps(stats, indent=2)}")
