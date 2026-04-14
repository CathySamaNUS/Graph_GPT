"""
Improve ESM-2 embedding coverage by fetching protein sequences directly
from the Ensembl REST API using ENSP IDs, bypassing UniProt entirely.

The original pipeline (1.py) achieves only ~47.7% coverage because:
  ENSP -> UniProt mapping: 95.2% (good)
  UniProt FASTA fetch:     50%   (many TrEMBL entries obsoleted/merged)
  Final coverage:          47.7%

This script:
  1. Loads existing node_x_embed.pt and node_has_esm.pt to find missing nodes
  2. Fetches protein sequences from Ensembl REST API using ENSP IDs directly
  3. Runs ESM-2 on the newly fetched sequences
  4. Merges new embeddings into the existing matrix
  5. Saves updated node_x_embed.pt and node_has_esm.pt

Usage:
  python new/2_improve_coverage.py \
    --data_dir data/OGB/ogbl_ppa_human \
    --model_name esm2_t33_650M_UR50D \
    --layer 33 \
    --batch_size 2
"""

import argparse
import json
import time
from pathlib import Path

import requests
import torch
from tqdm import tqdm

ENSEMBL_REST = "https://rest.ensembl.org"


# ---------------------------------------------------------------------------
# 1. Identify missing nodes
# ---------------------------------------------------------------------------

def load_protein_ids(data_dir: str) -> list[str]:
    """Load ordered protein IDs from the mapping file."""
    import pandas as pd
    mapping_path = Path(data_dir) / "mapping" / "nodeidx2proteinid.csv.gz"
    df = pd.read_csv(mapping_path)
    df = df.sort_values("node_idx")
    return df["protein_id"].tolist()


def find_missing_nodes(
    protein_ids: list[str], has_esm: torch.Tensor
) -> list[tuple[int, str]]:
    """Return [(node_idx, ensp_id), ...] for nodes without embeddings."""
    missing = []
    for idx, pid in enumerate(protein_ids):
        if not has_esm[idx].item():
            ensp = normalize_to_ensp(pid)
            missing.append((idx, ensp))
    return missing


def normalize_to_ensp(pid: str) -> str:
    pid = pid.strip()
    if "." in pid and pid.split(".")[-1].startswith("ENSP"):
        return pid.split(".")[-1]
    return pid


# ---------------------------------------------------------------------------
# 2. Fetch sequences from Ensembl REST API
# ---------------------------------------------------------------------------

def fetch_ensembl_sequences_batch(
    ensp_ids: list[str], batch_size: int = 50
) -> dict[str, str]:
    """
    Fetch protein sequences from Ensembl REST API using POST /sequence/id.
    Returns {ensp_id: amino_acid_sequence}.
    """
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
    })

    seqs = {}
    for i in tqdm(range(0, len(ensp_ids), batch_size), desc="Ensembl fetch"):
        batch = ensp_ids[i : i + batch_size]
        payload = {"ids": batch}

        for attempt in range(3):
            try:
                r = session.post(
                    f"{ENSEMBL_REST}/sequence/id",
                    json=payload,
                    params={"type": "protein"},
                    timeout=60,
                )
                if r.status_code == 429:
                    # Rate limited
                    retry_after = float(r.headers.get("Retry-After", 2))
                    print(f"\n  Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                r.raise_for_status()
                results = r.json()
                for entry in results:
                    eid = entry.get("query") or entry.get("id", "")
                    seq = entry.get("seq", "")
                    if seq:
                        seqs[eid] = seq
                break
            except Exception as e:
                print(f"\n  Batch {i//batch_size} attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                else:
                    print(f"  Skipping batch {i//batch_size} after 3 failures")

        # Be polite to Ensembl API
        time.sleep(0.1)

    return seqs


def fetch_ensembl_sequences_single(
    ensp_ids: list[str],
) -> dict[str, str]:
    """
    Fallback: fetch one-by-one for IDs that failed in batch mode.
    """
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    seqs = {}

    for ensp in tqdm(ensp_ids, desc="Ensembl single fetch"):
        for attempt in range(2):
            try:
                r = session.get(
                    f"{ENSEMBL_REST}/sequence/id/{ensp}",
                    params={"type": "protein"},
                    timeout=30,
                )
                if r.status_code == 429:
                    time.sleep(float(r.headers.get("Retry-After", 2)))
                    continue
                if r.status_code == 404:
                    break  # Genuinely not found
                r.raise_for_status()
                data = r.json()
                seq = data.get("seq", "")
                if seq:
                    seqs[ensp] = seq
                break
            except Exception:
                if attempt == 0:
                    time.sleep(2)
        time.sleep(0.05)

    return seqs


# ---------------------------------------------------------------------------
# 3. ESM-2 embedding (reuse from 1.py)
# ---------------------------------------------------------------------------

def pool_embeddings(token_reps, tokens, alphabet, mode: str = "mean"):
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
def esm2_embed(
    seqs: dict[str, str],
    model_name: str = "esm2_t33_650M_UR50D",
    layer: int = 33,
    pooling: str = "mean",
    batch_size: int = 2,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    import esm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
                    p = pool_embeddings(
                        r["representations"][layer], tok, alphabet, mode=pooling
                    )
                    out[key] = p[0].detach().cpu()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  Skipping {key} (len={len(seq)}): OOM even with bs=1")
    return out


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Improve ESM-2 coverage using Ensembl REST API"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/OGB/ogbl_ppa_human",
        help="Directory containing node_x_embed.pt and mapping/",
    )
    parser.add_argument("--model_name", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--no_l2_normalize", action="store_true",
        help="Skip L2 normalization of embeddings",
    )
    parser.add_argument(
        "--ensembl_batch_size", type=int, default=50,
        help="Batch size for Ensembl REST API requests",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    embed_path = Path(data_dir) / "node_x_embed.pt"
    mask_path = Path(data_dir) / "node_has_esm.pt"

    # --- Load existing embeddings ---
    print("=" * 60)
    print("Improve ESM-2 Coverage via Ensembl REST API")
    print("=" * 60)

    protein_ids = load_protein_ids(data_dir)
    num_nodes = len(protein_ids)
    print(f"Total nodes: {num_nodes}")

    if embed_path.exists() and mask_path.exists():
        x_embed = torch.load(str(embed_path), map_location="cpu")
        has_esm = torch.load(str(mask_path), map_location="cpu")
        old_coverage = has_esm.float().mean().item()
        embed_dim = x_embed.shape[1]
        print(f"Existing coverage: {has_esm.sum().item()}/{num_nodes} ({old_coverage:.4f})")
    else:
        print("No existing embeddings found, starting fresh.")
        embed_dim = None  # Will be inferred from ESM-2 output
        x_embed = None
        has_esm = torch.zeros(num_nodes, dtype=torch.bool)
        old_coverage = 0.0

    # --- Find missing nodes ---
    missing = find_missing_nodes(protein_ids, has_esm)
    print(f"Missing nodes: {len(missing)}")

    if not missing:
        print("All nodes already have embeddings!")
        return

    missing_ensp = [ensp for _, ensp in missing]
    missing_idx = {ensp: idx for idx, ensp in missing}

    # --- Fetch sequences from Ensembl ---
    print(f"\nFetching sequences from Ensembl for {len(missing_ensp)} proteins...")
    ensp2seq = fetch_ensembl_sequences_batch(
        missing_ensp, batch_size=args.ensembl_batch_size
    )
    print(f"Ensembl batch fetch: {len(ensp2seq)}/{len(missing_ensp)} sequences")

    # Retry failed ones individually
    still_missing = [e for e in missing_ensp if e not in ensp2seq]
    if still_missing:
        print(f"\nRetrying {len(still_missing)} failed IDs individually...")
        extra = fetch_ensembl_sequences_single(still_missing)
        ensp2seq.update(extra)
        print(f"After single fetch: {len(ensp2seq)}/{len(missing_ensp)} total")

    if not ensp2seq:
        print("No new sequences fetched. Coverage unchanged.")
        return

    # --- Run ESM-2 ---
    print(f"\nRunning ESM-2 ({args.model_name}) on {len(ensp2seq)} new sequences...")
    ensp2emb = esm2_embed(
        ensp2seq,
        model_name=args.model_name,
        layer=args.layer,
        pooling=args.pooling,
        batch_size=args.batch_size,
    )
    print(f"ESM-2 embeddings generated: {len(ensp2emb)}")

    # L2 normalize
    if not args.no_l2_normalize:
        for key in ensp2emb:
            ensp2emb[key] = torch.nn.functional.normalize(
                ensp2emb[key].float(), p=2, dim=0, eps=1e-12
            )

    # --- Merge into existing matrix ---
    if x_embed is None:
        embed_dim = next(iter(ensp2emb.values())).shape[0]
        x_embed = torch.zeros((num_nodes, embed_dim), dtype=torch.float32)

    new_hits = 0
    for ensp, emb in ensp2emb.items():
        idx = missing_idx.get(ensp)
        if idx is not None:
            x_embed[idx] = emb.float()
            has_esm[idx] = True
            new_hits += 1

    new_coverage = has_esm.float().mean().item()
    print(f"\n{'=' * 60}")
    print(f"Coverage improved: {old_coverage:.4f} -> {new_coverage:.4f}")
    print(f"  Old: {int(old_coverage * num_nodes)}/{num_nodes}")
    print(f"  New: {has_esm.sum().item()}/{num_nodes}")
    print(f"  Added: {new_hits} embeddings")
    print(f"  Still missing: {(~has_esm).sum().item()}")
    print(f"{'=' * 60}")

    # --- Save ---
    torch.save(x_embed, str(embed_path))
    torch.save(has_esm, str(mask_path))

    stats = {
        "num_nodes": num_nodes,
        "num_nodes_with_esm": int(has_esm.sum().item()),
        "num_nodes_without_esm": int((~has_esm).sum().item()),
        "coverage": round(new_coverage, 6),
        "embed_dim": embed_dim,
        "old_coverage": round(old_coverage, 6),
        "new_embeddings_added": new_hits,
        "ensembl_sequences_fetched": len(ensp2seq),
        "esm2_embeddings_generated": len(ensp2emb),
    }
    stats_path = Path(data_dir) / "node_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved updated embeddings to {data_dir}")
    print(f"Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
