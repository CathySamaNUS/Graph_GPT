#!/usr/bin/env python3
"""
Load pre-fetched Ensembl sequences and run ESM-2 embedding.
Merge into existing node_x_embed.pt.
"""
import argparse, json, sys, time
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
    if mode == "max":
        masked = token_reps.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values
        return torch.nan_to_num(pooled, neginf=0.0)
    raise ValueError(mode)


@torch.no_grad()
def esm2_embed(seqs, model_name, layer, pooling, batch_size, device):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    max_seq_len = 1022
    truncated = 0
    items = []
    for key, seq in seqs.items():
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
            truncated += 1
        items.append((key, seq))
    if truncated:
        print(f"  Truncated {truncated} sequences to {max_seq_len} residues")

    # Clean sequences: remove non-standard amino acid characters (e.g., '*' stop codons)
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    cleaned = 0
    clean_items = []
    for key, seq in items:
        clean_seq = "".join(c for c in seq if c in valid_aa)
        if len(clean_seq) != len(seq):
            cleaned += 1
        if clean_seq:  # Skip if entirely non-standard
            clean_items.append((key, clean_seq))
    if cleaned:
        print(f"  Cleaned {cleaned} sequences (removed non-standard characters like *)")
    items = clean_items

    items.sort(key=lambda x: len(x[1]))
    out = {}
    for i in tqdm(range(0, len(items), batch_size), desc="ESM2 embed"):
        batch_items = items[i : i + batch_size]
        labels, strs, tokens = batch_converter(batch_items)
        del labels, strs
        tokens = tokens.to(device)
        try:
            res = model(tokens, repr_layers=[layer], return_contacts=False)
            token_reps = res["representations"][layer]
            pooled = pool_embeddings(token_reps, tokens, alphabet, mode=pooling)
            for b, (key, _) in enumerate(batch_items):
                out[key] = pooled[b].detach().cpu()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n  OOM at batch {i//batch_size}, retrying one-by-one...")
            for key, seq in batch_items:
                try:
                    _, _, tok = batch_converter([(key, seq)])
                    tok = tok.to(device)
                    r = model(tok, repr_layers=[layer], return_contacts=False)
                    p = pool_embeddings(r["representations"][layer], tok, alphabet, mode=pooling)
                    out[key] = p[0].detach().cpu()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  Skipping {key} (len={len(seq)}): OOM even with bs=1")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/OGB/ogbl_ppa_human")
    parser.add_argument("--model_name", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load existing embeddings
    x_embed = torch.load(str(data_dir / "node_x_embed.pt"), map_location="cpu")
    has_esm = torch.load(str(data_dir / "node_has_esm.pt"), map_location="cpu")
    print(f"Existing coverage: {has_esm.sum().item()}/{has_esm.shape[0]} ({has_esm.float().mean().item():.4f})")

    # Load pre-fetched Ensembl sequences
    seq_path = data_dir / "ensembl_sequences.json"
    with open(seq_path) as f:
        ensp2seq = json.load(f)
    print(f"Loaded {len(ensp2seq)} pre-fetched Ensembl sequences")

    # Build missing node index
    import pandas as pd
    df = pd.read_csv(str(data_dir / "mapping" / "nodeidx2proteinid.csv.gz"))
    df = df.sort_values("node_idx")
    protein_ids = df["protein_id"].tolist()

    missing_idx = {}
    for idx, pid in enumerate(protein_ids):
        if not has_esm[idx].item():
            ensp = pid.strip()
            if "." in ensp and ensp.split(".")[-1].startswith("ENSP"):
                ensp = ensp.split(".")[-1]
            missing_idx[ensp] = idx

    # Filter sequences to only missing nodes
    seqs_to_embed = {k: v for k, v in ensp2seq.items() if k in missing_idx}
    print(f"Sequences to embed (missing nodes only): {len(seqs_to_embed)}")

    if not seqs_to_embed:
        print("Nothing to embed!")
        return

    # Run ESM-2
    print(f"\nRunning ESM-2 ({args.model_name}) on {len(seqs_to_embed)} sequences...")
    ensp2emb = esm2_embed(
        seqs_to_embed,
        model_name=args.model_name,
        layer=args.layer,
        pooling=args.pooling,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"ESM-2 embeddings generated: {len(ensp2emb)}")

    # L2 normalize
    for key in ensp2emb:
        ensp2emb[key] = torch.nn.functional.normalize(
            ensp2emb[key].float(), p=2, dim=0, eps=1e-12
        )

    # Merge
    new_hits = 0
    for ensp, emb in ensp2emb.items():
        idx = missing_idx.get(ensp)
        if idx is not None:
            x_embed[idx] = emb.float()
            has_esm[idx] = True
            new_hits += 1

    new_coverage = has_esm.float().mean().item()
    print(f"\n{'='*60}")
    print(f"Added {new_hits} new embeddings")
    print(f"New coverage: {has_esm.sum().item()}/{has_esm.shape[0]} ({new_coverage:.4f})")
    print(f"Still missing: {(~has_esm).sum().item()}")
    print(f"{'='*60}")

    # Save
    torch.save(x_embed, str(data_dir / "node_x_embed.pt"))
    torch.save(has_esm, str(data_dir / "node_has_esm.pt"))

    stats = {
        "num_nodes": int(has_esm.shape[0]),
        "num_nodes_with_esm": int(has_esm.sum().item()),
        "num_nodes_without_esm": int((~has_esm).sum().item()),
        "coverage": round(new_coverage, 6),
        "embed_dim": int(x_embed.shape[1]),
    }
    with open(str(data_dir / "node_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved updated embeddings to {data_dir}")


if __name__ == "__main__":
    main()
