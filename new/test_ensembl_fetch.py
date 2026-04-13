#!/usr/bin/env python3
"""Test Ensembl coverage on login node (no GPU needed)."""
import json, sys, time
from pathlib import Path
import torch, pandas as pd, requests
from tqdm import tqdm

REPO = "/oceanstor/home/e1553200/graph-gpt-main"
DATA_DIR = f"{REPO}/data/OGB/ogbl_ppa_human"
ENSEMBL_REST = "https://rest.ensembl.org"

# Load existing coverage
has_esm = torch.load(f"{DATA_DIR}/node_has_esm.pt", map_location="cpu")
df = pd.read_csv(f"{DATA_DIR}/mapping/nodeidx2proteinid.csv.gz")
df = df.sort_values("node_idx")
protein_ids = df["protein_id"].tolist()

print(f"Total nodes: {len(protein_ids)}")
print(f"Existing coverage: {has_esm.sum().item()}/{len(protein_ids)} ({has_esm.float().mean().item():.4f})")

# Find missing ENSP IDs
missing_ensp = []
for idx, pid in enumerate(protein_ids):
    if not has_esm[idx].item():
        ensp = pid.strip()
        if "." in ensp and ensp.split(".")[-1].startswith("ENSP"):
            ensp = ensp.split(".")[-1]
        missing_ensp.append(ensp)

print(f"Missing nodes: {len(missing_ensp)}")

# Test batch fetch on first 100
print(f"\nTesting Ensembl batch fetch on first 100 missing IDs...")
session = requests.Session()
session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

test_batch = missing_ensp[:100]
payload = {"ids": test_batch}
try:
    r = session.post(
        f"{ENSEMBL_REST}/sequence/id",
        json=payload,
        params={"type": "protein"},
        timeout=60,
    )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        results = r.json()
        found = sum(1 for e in results if e.get("seq"))
        print(f"Found sequences: {found}/{len(test_batch)}")
        # Show a sample
        for e in results[:3]:
            eid = e.get("query") or e.get("id", "?")
            seq = e.get("seq", "")
            print(f"  {eid}: len={len(seq)}")
    else:
        print(f"Response: {r.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Now do the full fetch in batches of 50
print(f"\nFull Ensembl fetch for all {len(missing_ensp)} missing IDs...")
all_seqs = {}
batch_size = 50
for i in tqdm(range(0, len(missing_ensp), batch_size), desc="Ensembl fetch"):
    batch = missing_ensp[i : i + batch_size]
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
                retry_after = float(r.headers.get("Retry-After", 2))
                time.sleep(retry_after)
                continue
            if r.status_code == 200:
                for entry in r.json():
                    eid = entry.get("query") or entry.get("id", "")
                    seq = entry.get("seq", "")
                    if seq:
                        all_seqs[eid] = seq
                break
            else:
                print(f"\n  Batch {i//batch_size}: status {r.status_code}")
                break
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                print(f"\n  Batch {i//batch_size} failed: {e}")
    time.sleep(0.15)

print(f"\n{'='*60}")
print(f"Ensembl sequences fetched: {len(all_seqs)} / {len(missing_ensp)}")
potential_coverage = (has_esm.sum().item() + len(all_seqs)) / len(protein_ids)
print(f"Potential new coverage: {has_esm.sum().item() + len(all_seqs)}/{len(protein_ids)} ({potential_coverage:.4f})")
print(f"{'='*60}")

# Save fetched sequences for later ESM-2 embedding
seq_path = f"{DATA_DIR}/ensembl_sequences.json"
with open(seq_path, "w") as f:
    json.dump(all_seqs, f)
print(f"Saved {len(all_seqs)} sequences to {seq_path}")
