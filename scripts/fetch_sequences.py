#!/usr/bin/env python3
"""Fetch all protein sequences from Ensembl REST API (run on login node with internet).

Saves sequences to {data_dir}/sequences.json for offline ESM-2 embedding on compute nodes.

Usage:
    python scripts/fetch_sequences.py --data_dir data/OGB/ogbl_ppa_human
"""
import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

ENSEMBL_REST = "https://rest.ensembl.org"


def load_protein_ids(data_dir: str):
    mapping_path = Path(data_dir) / "mapping" / "nodeidx2proteinid.csv.gz"
    with gzip.open(mapping_path, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = sorted(reader, key=lambda r: int(r[0]))
    return [row[2] for row in rows]


def normalize_to_ensp(pid: str) -> str:
    pid = pid.strip()
    if "." in pid and pid.split(".")[-1].startswith("ENSP"):
        return pid.split(".")[-1]
    return pid


def fetch_sequences_batch(session, ensp_ids, batch_size=50):
    """Fetch sequences from Ensembl REST API in batches."""
    results = {}
    for i in tqdm(range(0, len(ensp_ids), batch_size), desc="Fetching from Ensembl"):
        batch = ensp_ids[i : i + batch_size]
        payload = {"ids": batch}
        for attempt in range(3):
            try:
                r = session.post(
                    f"{ENSEMBL_REST}/sequence/id",
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                    json=payload,
                    timeout=30,
                )
                if r.status_code == 429:
                    retry_after = float(r.headers.get("Retry-After", 5))
                    time.sleep(retry_after)
                    continue
                if r.status_code == 200:
                    for entry in r.json():
                        results[entry["id"]] = entry["seq"]
                    break
                else:
                    print(f"  Batch {i}: HTTP {r.status_code}, retrying...")
                    time.sleep(2)
            except Exception as e:
                print(f"  Batch {i}: {e}, retrying...")
                time.sleep(3)
        time.sleep(0.5)  # rate limit
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/OGB/ogbl_ppa_human")
    args = parser.parse_args()

    protein_ids = load_protein_ids(args.data_dir)
    print(f"Total nodes: {len(protein_ids)}")

    ensp_ids = [normalize_to_ensp(pid) for pid in protein_ids]
    unique_ensp = list(set(ensp_ids))
    print(f"Unique ENSP IDs: {len(unique_ensp)}")

    session = requests.Session()
    ensp2seq = fetch_sequences_batch(session, unique_ensp, batch_size=50)
    print(f"Fetched sequences: {len(ensp2seq)} / {len(unique_ensp)} ({len(ensp2seq)/len(unique_ensp)*100:.1f}%)")

    # Save ordered sequences (matching node order)
    output = {
        "protein_ids": protein_ids,
        "ensp_ids": ensp_ids,
        "ensp2seq": ensp2seq,
        "num_total": len(protein_ids),
        "num_fetched": len(ensp2seq),
    }
    out_path = Path(args.data_dir) / "sequences.json"
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
