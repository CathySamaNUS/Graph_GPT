#!/usr/bin/env python3
"""Generate ESM-2 embeddings from pre-fetched sequences (run on compute node with GPU).

Reads sequences.json, runs ESM-2, saves node_x_embed.pt and node_has_esm.pt,
then PCA-reduces to 256 dimensions.

Usage:
    python scripts/embed_sequences.py --data_dir data/OGB/ogbl_ppa_human
"""
import argparse
import json
from pathlib import Path

import esm
import torch
from tqdm import tqdm


def pool_embeddings(token_reps, tokens, alphabet, mode="mean"):
    pad = alphabet.padding_idx
    bos = alphabet.cls_idx
    eos = alphabet.eos_idx
    valid = (tokens != pad) & (tokens != bos) & (tokens != eos)
    if mode == "mean":
        masked = token_reps * valid.unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return masked.sum(dim=1) / denom
    raise ValueError(mode)


@torch.no_grad()
def embed_sequences(ensp2seq, model_name="esm2_t33_650M_UR50D", layer=33, batch_size=2, device="cuda"):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    max_seq_len = 1022

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    items = []
    for ensp_id, seq in ensp2seq.items():
        seq = seq[:max_seq_len]
        seq = "".join(c if c in valid_aa else "X" for c in seq.upper())
        if len(seq) > 0:
            items.append((ensp_id, seq))

    ensp2emb = {}
    for i in tqdm(range(0, len(items), batch_size), desc="ESM-2 embedding"):
        batch = items[i : i + batch_size]
        labels, strs, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        try:
            results = model(tokens, repr_layers=[layer], return_contacts=False)
            reps = results["representations"][layer].cpu()
            pooled = pool_embeddings(reps, tokens.cpu(), alphabet, mode="mean")
            for j, (ensp_id, _) in enumerate(batch):
                ensp2emb[ensp_id] = pooled[j]
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print(f"  OOM at batch {i}, processing one-by-one")
                for ensp_id, seq_str in batch:
                    try:
                        _, _, tok = batch_converter([(ensp_id, seq_str)])
                        tok = tok.to(device)
                        res = model(tok, repr_layers=[layer], return_contacts=False)
                        rep = res["representations"][layer].cpu()
                        p = pool_embeddings(rep, tok.cpu(), alphabet, mode="mean")
                        ensp2emb[ensp_id] = p[0]
                    except RuntimeError:
                        torch.cuda.empty_cache()
                        print(f"  Skipping {ensp_id} (OOM)")
            else:
                raise
    return ensp2emb


def pca_reduce(x_embed, has_esm, target_dim=256):
    mask = has_esm.bool()
    X = x_embed[mask].float()
    mean = X.mean(dim=0)
    X_centered = X - mean
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vt[:target_dim].T
    total_var = (S ** 2).sum().item()
    var_kept = (S[:target_dim] ** 2).sum().item()
    print(f"PCA: {X.shape[1]} -> {target_dim} dims, variance explained: {var_kept/total_var*100:.1f}%")

    x_pca = torch.zeros((x_embed.shape[0], target_dim), dtype=torch.float32)
    x_pca[mask] = (X_centered @ V).float()
    norms = x_pca.norm(dim=1, keepdim=True).clamp(min=1e-12)
    x_pca[mask] = x_pca[mask] / norms[mask]
    return x_pca, {"mean": mean, "V": V, "S": S}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/OGB/ogbl_ppa_human")
    parser.add_argument("--model_name", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--pca_dim", type=int, default=256)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load pre-fetched sequences
    with open(data_dir / "sequences.json") as f:
        seq_data = json.load(f)
    protein_ids = seq_data["protein_ids"]
    ensp_ids = seq_data["ensp_ids"]
    ensp2seq = seq_data["ensp2seq"]
    print(f"Nodes: {len(protein_ids)}, Sequences available: {len(ensp2seq)}")

    # Run ESM-2
    ensp2emb = embed_sequences(ensp2seq, args.model_name, args.layer, args.batch_size, device)
    embed_dim = next(iter(ensp2emb.values())).shape[0]
    print(f"Embedded: {len(ensp2emb)} proteins, dim={embed_dim}")

    # Build node-aligned matrices
    x_embed = torch.zeros((len(protein_ids), embed_dim), dtype=torch.float32)
    has_esm = torch.zeros(len(protein_ids), dtype=torch.bool)
    for idx, ensp_id in enumerate(ensp_ids):
        ensp_clean = ensp_id.strip()
        if "." in ensp_clean and ensp_clean.split(".")[-1].startswith("ENSP"):
            ensp_clean = ensp_clean.split(".")[-1]
        if ensp_clean in ensp2emb:
            x_embed[idx] = ensp2emb[ensp_clean]
            has_esm[idx] = True

    coverage = has_esm.float().mean().item()
    print(f"Coverage: {has_esm.sum().item()}/{len(protein_ids)} ({coverage*100:.1f}%)")

    # Save full-dim embeddings as backup
    torch.save(x_embed, data_dir / "node_x_embed_1280.pt")
    torch.save(has_esm, data_dir / "node_has_esm.pt")

    # PCA reduce
    x_pca, pca_params = pca_reduce(x_embed, has_esm, args.pca_dim)
    torch.save(x_pca, data_dir / "node_x_embed.pt")
    torch.save(pca_params, data_dir / "pca_params.pt")

    stats = {
        "num_nodes": len(protein_ids),
        "num_with_esm": int(has_esm.sum().item()),
        "coverage": round(coverage, 6),
        "original_dim": embed_dim,
        "pca_dim": args.pca_dim,
        "model": args.model_name,
    }
    with open(data_dir / "esm2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats: {json.dumps(stats, indent=2)}")
    print(f"Saved: node_x_embed.pt ({args.pca_dim}-dim), node_has_esm.pt, node_x_embed_1280.pt")


if __name__ == "__main__":
    main()
