import argparse
import csv
import io
import json
import time
from pathlib import Path

import esm
import requests
import torch
from tqdm import tqdm

UNIPROT_REST = "https://rest.uniprot.org"


def normalize_to_ensp(pid: str) -> str:
    """
    Support:
      - 9606.ENSP00000334393 -> ENSP00000334393
      - ENSP00000334393      -> ENSP00000334393
    """
    pid = pid.strip()
    if "." in pid and pid.split(".")[-1].startswith("ENSP"):
        return pid.split(".")[-1]
    return pid


def read_ids_from_txt(path: str) -> list[str]:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(line)
    return ids


def get_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "ogbl-ppa-esm2-pipeline/1.0"})
    return session


def uniprot_submit_idmapping(
    session: requests.Session, from_db: str, to_db: str, ids: list[str]
) -> str:
    r = session.post(
        f"{UNIPROT_REST}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["jobId"]


def uniprot_wait_job(
    session: requests.Session, job_id: str, poll_sec: int = 3, timeout_sec: int = 600
):
    t0 = time.time()
    while True:
        r = session.get(f"{UNIPROT_REST}/idmapping/status/{job_id}", timeout=60)
        r.raise_for_status()
        js = r.json()
        if js.get("jobStatus") in {"RUNNING", "NEW"}:
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(f"UniProt idmapping timeout: {job_id}")
            time.sleep(poll_sec)
            continue
        return js


def uniprot_fetch_results_tsv(
    session: requests.Session, job_id: str
) -> list[tuple[str, str]]:
    """
    Return: [(from_id, to_uniprot_acc), ...]
    """
    r = session.get(
        f"{UNIPROT_REST}/idmapping/stream/{job_id}",
        params={"format": "tsv"},
        timeout=120,
    )
    r.raise_for_status()

    rows = []
    reader = csv.DictReader(io.StringIO(r.text), delimiter="\t")
    for row in reader:
        rows.append((row["From"], row["To"]))
    return rows


def map_ensp_to_uniprot(
    ensp_list: list[str], batch_size: int = 5000
) -> dict[str, list[str]]:
    """
    Input: ENSP list
    Output: {ENSP: [UniProtAcc1, UniProtAcc2, ...]}
    """
    session = get_http_session()
    mapping: dict[str, list[str]] = {}
    for i in tqdm(range(0, len(ensp_list), batch_size), desc="UniProt mapping batches"):
        batch = ensp_list[i : i + batch_size]
        job = uniprot_submit_idmapping(session, "Ensembl_Protein", "UniProtKB", batch)
        _ = uniprot_wait_job(session, job)
        pairs = uniprot_fetch_results_tsv(session, job)
        for src, dst in pairs:
            mapping.setdefault(src, []).append(dst)
        time.sleep(0.2)
    return mapping


def choose_uniprot_accessions(
    ensp_list: list[str], ensp2uni: dict[str, list[str]]
) -> tuple[dict[str, str], dict[str, int]]:
    """
    Keep the selection rule explicit so the experiment is reproducible.
    Current rule: choose the first returned UniProt accession.
    """
    chosen_uni = {}
    multi_map = 0
    unmapped = 0

    for ensp in ensp_list:
        candidates = ensp2uni.get(ensp, [])
        if not candidates:
            unmapped += 1
            continue
        if len(candidates) > 1:
            multi_map += 1
        chosen_uni[ensp] = candidates[0]

    stats = {
        "ensp_total": len(ensp_list),
        "ensp_unmapped_to_uniprot": unmapped,
        "ensp_multi_mapped_to_uniprot": multi_map,
        "ensp_chosen_uniprot": len(chosen_uni),
    }
    return chosen_uni, stats


def fetch_uniprot_fasta(
    accessions: list[str], batch_size: int = 50
) -> dict[str, str]:
    """
    Use UniProtKB search to fetch FASTA in batches.
    Return: {acc: sequence}
    """
    session = get_http_session()
    seqs = {}

    for i in tqdm(range(0, len(accessions), batch_size), desc="Fetch UniProt FASTA"):
        batch = accessions[i : i + batch_size]
        q = " OR ".join([f"accession:{a}" for a in batch])
        fasta = None
        for attempt in range(3):
            try:
                r = session.get(
                    f"{UNIPROT_REST}/uniprotkb/search",
                    params={"query": q, "format": "fasta"},
                    timeout=120,
                )
                r.raise_for_status()
                fasta = r.text.strip()
                break
            except Exception as e:
                print(f"\n  Batch {i//batch_size} attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  Skipping batch {i//batch_size} after 3 failures")
        if not fasta:
            continue

        cur = None
        buf = []
        for line in fasta.splitlines():
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf)
                parts = line.split("|")
                cur = parts[1] if len(parts) >= 3 else line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if cur is not None:
            seqs[cur] = "".join(buf)

        time.sleep(0.2)
    return seqs


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
    batch_size: int = 8,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Input: {key: sequence}
    Output: {key: embedding[D]}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    # Truncate long sequences to ESM-2 max context (1022 tokens = 1024 - BOS - EOS)
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

    # Sort by length (shortest first) to minimize padding waste
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
            # Retry one-by-one
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
                    print(f"  Skipping {key} (len={len(seq)}): OOM even with batch_size=1")
    return out


def l2_normalize_embeddings(emb_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in emb_map.items():
        normalized[key] = torch.nn.functional.normalize(
            value.float(), p=2, dim=0, eps=1e-12
        )
    return normalized


def build_node_embedding_matrix(
    node_ids: list[str],
    ensp2emb: dict[str, torch.Tensor],
    embed_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Build GraphGPT-ready artifacts:
      - x_embed: [num_nodes, D]
      - has_esm: [num_nodes]
    """
    normalized_node_ids = [normalize_to_ensp(x) for x in node_ids]
    if embed_dim is None:
        if not ensp2emb:
            raise ValueError("ensp2emb is empty; cannot infer embed_dim.")
        embed_dim = next(iter(ensp2emb.values())).shape[0]

    x_embed = torch.zeros((len(normalized_node_ids), embed_dim), dtype=torch.float32)
    has_esm = torch.zeros(len(normalized_node_ids), dtype=torch.bool)

    hit = 0
    miss = 0
    for idx, ensp in enumerate(normalized_node_ids):
        emb = ensp2emb.get(ensp)
        if emb is None:
            miss += 1
            continue
        x_embed[idx] = emb.float()
        has_esm[idx] = True
        hit += 1

    stats = {
        "num_nodes": len(normalized_node_ids),
        "num_nodes_with_esm": hit,
        "num_nodes_without_esm": miss,
        "coverage": round(hit / max(len(normalized_node_ids), 1), 6),
        "embed_dim": embed_dim,
    }
    return x_embed, has_esm, stats


def run_pipeline(
    protein_ids: list[str],
    pooling: str = "mean",
    batch_size: int = 8,
    model_name: str = "esm2_t33_650M_UR50D",
    layer: int = 33,
    l2_normalize: bool = True,
):
    ensp = [normalize_to_ensp(x) for x in protein_ids]
    ensp = list(dict.fromkeys(ensp))

    ensp2uni = map_ensp_to_uniprot(ensp)
    chosen_uni, uni_stats = choose_uniprot_accessions(ensp, ensp2uni)

    uni_list = sorted(set(chosen_uni.values()))
    print(f"ENSP mapped to UniProt: {len(chosen_uni)} / {len(ensp)}")
    print(f"Unique UniProt accessions: {len(uni_list)}")

    uni2seq = fetch_uniprot_fasta(uni_list)
    print(f"UniProt sequences fetched: {len(uni2seq)} / {len(uni_list)}")

    uni2emb = esm2_embed(
        uni2seq,
        pooling=pooling,
        batch_size=batch_size,
        model_name=model_name,
        layer=layer,
    )
    if l2_normalize:
        uni2emb = l2_normalize_embeddings(uni2emb)

    ensp2emb = {}
    missing_embedding = 0
    for ensp_id, uni_acc in chosen_uni.items():
        if uni_acc in uni2emb:
            ensp2emb[ensp_id] = uni2emb[uni_acc]
        else:
            missing_embedding += 1

    summary = {
        **uni_stats,
        "uniprot_total_after_choice": len(uni_list),
        "uniprot_sequence_fetched": len(uni2seq),
        "uniprot_embedding_ready": len(uni2emb),
        "ensp_embedding_ready": len(ensp2emb),
        "ensp_missing_embedding_after_uniprot_choice": missing_embedding,
        "pooling": pooling,
        "esm_model_name": model_name,
        "esm_layer": layer,
        "l2_normalize": l2_normalize,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return ensp2emb, chosen_uni, ensp2uni, summary


def save_pipeline_outputs(
    output_dir: str,
    ensp2emb: dict[str, torch.Tensor],
    chosen_uni: dict[str, str],
    ensp2uni: dict[str, list[str]],
    summary: dict,
    node_ids: list[str] | None = None,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(ensp2emb, out_dir / "ensp2emb.pt")
    with open(out_dir / "ensp2uniprot_best.json", "w", encoding="utf-8") as f:
        json.dump(chosen_uni, f, indent=2, ensure_ascii=False)
    with open(out_dir / "ensp2uniprot_all.json", "w", encoding="utf-8") as f:
        json.dump(ensp2uni, f, indent=2, ensure_ascii=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if node_ids is not None:
        x_embed, has_esm, node_stats = build_node_embedding_matrix(node_ids, ensp2emb)
        torch.save(x_embed, out_dir / "node_x_embed.pt")
        torch.save(has_esm, out_dir / "node_has_esm.pt")
        with open(out_dir / "node_stats.json", "w", encoding="utf-8") as f:
            json.dump(node_stats, f, indent=2, ensure_ascii=False)
        print(json.dumps(node_stats, indent=2, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ESM-2 protein embeddings for GraphGPT/ogbl-ppa."
    )
    parser.add_argument(
        "--input_txt",
        type=str,
        default="",
        help="Text file of protein IDs, one per line. Supports STRING-style IDs like 9606.ENSP...",
    )
    parser.add_argument(
        "--node_order_txt",
        type=str,
        default="",
        help="Optional node-order text file. If provided, save GraphGPT-ready node_x_embed.pt and node_has_esm.pt.",
    )
    parser.add_argument("--output_dir", type=str, default="new/esm_outputs")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--no_l2_normalize", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input_txt:
        protein_ids = read_ids_from_txt(args.input_txt)
    else:
        protein_ids = ["9606.ENSP00000255882"]

    node_ids = read_ids_from_txt(args.node_order_txt) if args.node_order_txt else None

    ensp2emb, ensp2uni_best, ensp2uni_all, summary = run_pipeline(
        protein_ids,
        pooling=args.pooling,
        batch_size=args.batch_size,
        model_name=args.model_name,
        layer=args.layer,
        l2_normalize=not args.no_l2_normalize,
    )
    save_pipeline_outputs(
        args.output_dir,
        ensp2emb,
        ensp2uni_best,
        ensp2uni_all,
        summary,
        node_ids=node_ids,
    )
    print(f"Saved outputs to: {args.output_dir}")


'''
x_embed[0] = emb["ENSP00000255882"]
x_embed[1] = emb["ENSP00000334393"]
x_embed[2] = emb["ENSP00000412345"]
x_embed[3] = emb["ENSP00000567890"]

'''