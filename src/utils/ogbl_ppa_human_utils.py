import argparse
import builtins
import copy
import csv
import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
try:
    from ogb.linkproppred import PygLinkPropPredDataset
except ImportError:
    from ogb.linkproppred.dataset_pyg import PygLinkPropPredDataset
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected


@contextmanager
def suppress_ogb_update_prompt(default_answer: str = "n"):
    original_input = builtins.input
    builtins.input = lambda prompt="": default_answer
    try:
        yield
    finally:
        builtins.input = original_input


@contextmanager
def force_torch_load_weights_only_false():
    original_torch_load = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_compat
    try:
        yield
    finally:
        torch.load = original_torch_load


def load_ogbl_ppa_dataset(root: str) -> PygLinkPropPredDataset:
    # OGB's processed PyG files contain Data objects, so PyTorch 2.6 needs
    # weights_only=False when torch.load is called inside the dataset class.
    with suppress_ogb_update_prompt(), force_torch_load_weights_only_false():
        return PygLinkPropPredDataset(name="ogbl-ppa", root=root)


def find_mapping_file(dataset_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "protein" in name and "nodeidx2" in name:
            candidates.append(path)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: (p.suffix != ".gz", len(str(p)), str(p)))
    return candidates[0]


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_nodeidx_to_protein_ids(mapping_file: Path, num_nodes: int) -> List[str]:
    with _open_text(mapping_file) as fp:
        sample = fp.read(2048)
        fp.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = "," if "," in sample else "\t"

        reader = csv.reader(fp, delimiter=delimiter)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError(f"Mapping file is empty: {mapping_file}")

    has_header = not rows[0][0].strip().isdigit()
    if has_header:
        rows = rows[1:]

    protein_ids = [""] * num_nodes
    seen = 0
    for row in rows:
        if len(row) < 2:
            continue
        idx = int(row[0])
        if idx < 0 or idx >= num_nodes:
            raise IndexError(
                f"node index {idx} from {mapping_file} is out of range for num_nodes={num_nodes}"
            )
        protein_ids[idx] = row[1].strip()
        seen += 1

    missing = protein_ids.count("")
    if missing > 0:
        raise ValueError(
            f"Mapping file {mapping_file} does not cover all nodes: missing {missing} / {num_nodes}"
        )
    if seen != num_nodes:
        raise ValueError(
            f"Unexpected mapping row count in {mapping_file}: {seen} != {num_nodes}"
        )
    return protein_ids


def extract_taxon_id(protein_id: str) -> Optional[str]:
    protein_id = protein_id.strip()
    if not protein_id:
        return None
    prefix = protein_id.split(".", 1)[0]
    return prefix if prefix.isdigit() else None


def build_taxon_mask_from_protein_ids(
    protein_ids: List[str], taxon_id: str
) -> torch.BoolTensor:
    mask = [extract_taxon_id(pid) == str(taxon_id) for pid in protein_ids]
    return torch.tensor(mask, dtype=torch.bool)


def build_taxon_mask_from_species_col(graph: Data, species_col: int) -> torch.BoolTensor:
    if not hasattr(graph, "x"):
        raise ValueError("Graph has no `x`, cannot use --species-col fallback.")
    if graph.x.ndim != 2:
        raise ValueError(f"Expected graph.x to be 2D one-hot, got shape {tuple(graph.x.shape)}")
    if not (0 <= species_col < graph.x.shape[1]):
        raise ValueError(f"species_col {species_col} is out of range for x.shape[1]={graph.x.shape[1]}")
    return graph.x[:, species_col].to(torch.bool)


def build_old_to_new_index(keep_nodes: torch.BoolTensor) -> Tensor:
    old_to_new = torch.full((keep_nodes.shape[0],), -1, dtype=torch.long)
    old_to_new[keep_nodes] = torch.arange(int(keep_nodes.sum().item()), dtype=torch.long)
    return old_to_new


def filter_pair_index_tensor(
    pair_tensor: Tensor, keep_nodes: torch.BoolTensor, old_to_new: Tensor
) -> Tensor:
    if pair_tensor.ndim != 2 or pair_tensor.shape[1] != 2:
        raise ValueError(
            f"Expected pair tensor with shape [N, 2], got {tuple(pair_tensor.shape)}"
        )
    mask = keep_nodes[pair_tensor[:, 0]] & keep_nodes[pair_tensor[:, 1]]
    if mask.numel() == 0:
        return pair_tensor.new_empty((0, 2))
    filtered = pair_tensor[mask]
    return old_to_new[filtered]


def filter_graph_by_nodes(graph: Data, keep_nodes: torch.BoolTensor) -> Tuple[Data, Tensor]:
    old_to_new = build_old_to_new_index(keep_nodes)
    edge_index = graph.edge_index
    edge_mask = keep_nodes[edge_index[0]] & keep_nodes[edge_index[1]]
    new_edge_index = old_to_new[edge_index[:, edge_mask]]

    filtered = Data(num_nodes=int(keep_nodes.sum().item()))
    filtered.edge_index = new_edge_index

    for key, value in graph:
        if key in {"edge_index", "num_nodes"}:
            continue
        if isinstance(value, Tensor):
            if value.size(0) == graph.num_nodes:
                filtered[key] = value[keep_nodes]
            elif value.size(0) == graph.edge_index.shape[1]:
                filtered[key] = value[edge_mask]
            else:
                filtered[key] = value.clone()
        else:
            filtered[key] = copy.deepcopy(value)

    filtered.old_node_id = keep_nodes.nonzero(as_tuple=True)[0].to(torch.long)
    return filtered, old_to_new


def filter_split_edge(
    split_edge: Dict[str, Dict[str, Tensor]],
    keep_nodes: torch.BoolTensor,
    old_to_new: Tensor,
) -> Dict[str, Dict[str, Tensor]]:
    filtered: Dict[str, Dict[str, Tensor]] = {}
    for split_name, part in split_edge.items():
        new_part: Dict[str, Tensor] = {}
        for key, value in part.items():
            if torch.is_tensor(value) and value.ndim == 2 and value.shape[1] == 2:
                new_part[key] = filter_pair_index_tensor(value, keep_nodes, old_to_new)
            else:
                raise ValueError(
                    f"Unsupported split tensor for ogbl-ppa filtering: split={split_name}, "
                    f"key={key}, shape={getattr(value, 'shape', None)}"
                )
        filtered[split_name] = new_part
    return filtered


def infer_species_columns(graph: Data, keep_nodes: torch.BoolTensor) -> List[int]:
    if not hasattr(graph, "x") or graph.x.ndim != 2:
        return []
    if keep_nodes.sum().item() == 0:
        return []
    cols = torch.nonzero(graph.x[keep_nodes].sum(dim=0), as_tuple=True)[0].tolist()
    return cols


def verify_train_edges_in_graph(graph: Data, split_edge: Dict[str, Dict[str, Tensor]]):
    train_edges = split_edge["train"]["edge"]
    graph_hash = graph.edge_index[0].to(torch.int64) * graph.num_nodes + graph.edge_index[1].to(
        torch.int64
    )
    graph_hash = set(graph_hash.tolist())
    train_hash = train_edges[:, 0].to(torch.int64) * graph.num_nodes + train_edges[:, 1].to(
        torch.int64
    )
    missing = [idx for idx, val in enumerate(train_hash.tolist()) if val not in graph_hash]
    if missing:
        raise ValueError(
            f"{len(missing)} filtered training edges are not present in the filtered graph."
        )


def summarize_filtered_dataset(
    raw_graph: Data,
    filtered_graph: Data,
    split_edge: Dict[str, Dict[str, Tensor]],
    protein_ids: Optional[List[str]],
    new_to_old: Tensor,
    taxon_id: str,
    mapping_file: Optional[Path],
    species_cols: List[int],
) -> Dict:
    filtered_protein_example = None
    if protein_ids and new_to_old.numel() > 0:
        filtered_protein_example = protein_ids[int(new_to_old[0].item())]
    summary = {
        "taxon_id": str(taxon_id),
        "mapping_file": str(mapping_file) if mapping_file is not None else None,
        "raw_num_nodes": int(raw_graph.num_nodes),
        "raw_num_edges": int(raw_graph.edge_index.shape[1]),
        "filtered_num_nodes": int(filtered_graph.num_nodes),
        "filtered_num_edges": int(filtered_graph.edge_index.shape[1]),
        "filtered_is_undirected": bool(is_undirected(filtered_graph.edge_index)),
        "train_pos_edges": int(split_edge["train"]["edge"].shape[0]),
        "valid_pos_edges": int(split_edge["valid"]["edge"].shape[0]),
        "valid_neg_edges": int(split_edge["valid"]["edge_neg"].shape[0]),
        "test_pos_edges": int(split_edge["test"]["edge"].shape[0]),
        "test_neg_edges": int(split_edge["test"]["edge_neg"].shape[0]),
        "species_cols_after_filter": species_cols,
        "protein_id_prefix_example": filtered_protein_example,
        "isolated_nodes_after_filter": int(
            filtered_graph.num_nodes - torch.unique(filtered_graph.edge_index).numel()
        ),
    }
    return summary


def save_mapping_csv(
    output_file: Path,
    new_to_old: Tensor,
    protein_ids: Optional[List[str]],
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_file, "wt", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["node_idx", "old_node_idx", "protein_id"])
        for new_idx, old_idx in enumerate(new_to_old.tolist()):
            protein_id = protein_ids[old_idx] if protein_ids is not None else ""
            writer.writerow([new_idx, old_idx, protein_id])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a human-only ogbl-ppa subgraph dataset compatible with GraphGPT."
    )
    parser.add_argument("--data_root", type=str, default="data/OGB")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/OGB/ogbl_ppa_human",
        help="Directory to store graph.pt, split_edge.pt, mapping files and summary.",
    )
    parser.add_argument(
        "--taxon_id",
        type=str,
        default="9606",
        help="NCBI taxon id prefix in STRING-style protein IDs. Human is 9606.",
    )
    parser.add_argument(
        "--species_col",
        type=int,
        default=None,
        help="Fallback zero-based one-hot species column in graph.x when protein-id mapping is unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_ogbl_ppa_dataset(args.data_root)
    raw_graph = dataset[0]
    with force_torch_load_weights_only_false():
        split_edge = dataset.get_edge_split()
    dataset_root = Path(dataset.root)

    mapping_file = find_mapping_file(dataset_root)
    protein_ids: Optional[List[str]] = None
    if mapping_file is not None:
        protein_ids = read_nodeidx_to_protein_ids(mapping_file, raw_graph.num_nodes)
        keep_nodes = build_taxon_mask_from_protein_ids(protein_ids, args.taxon_id)
    elif args.species_col is not None:
        keep_nodes = build_taxon_mask_from_species_col(raw_graph, args.species_col)
    else:
        raise FileNotFoundError(
            "Could not find a protein-id mapping file under the ogbl-ppa dataset root. "
            "Please rerun with --species_col as a fallback."
        )

    if keep_nodes.sum().item() == 0:
        raise ValueError(f"No nodes matched taxon_id={args.taxon_id}")

    filtered_graph, old_to_new = filter_graph_by_nodes(raw_graph, keep_nodes)
    filtered_split_edge = filter_split_edge(split_edge, keep_nodes, old_to_new)
    verify_train_edges_in_graph(filtered_graph, filtered_split_edge)

    new_to_old = keep_nodes.nonzero(as_tuple=True)[0].to(torch.long)
    species_cols = infer_species_columns(raw_graph, keep_nodes)
    summary = summarize_filtered_dataset(
        raw_graph=raw_graph,
        filtered_graph=filtered_graph,
        split_edge=filtered_split_edge,
        protein_ids=protein_ids,
        new_to_old=new_to_old,
        taxon_id=args.taxon_id,
        mapping_file=mapping_file,
        species_cols=species_cols,
    )

    torch.save(filtered_graph, output_dir / "graph.pt")
    torch.save(filtered_split_edge, output_dir / "split_edge.pt")
    torch.save(keep_nodes, output_dir / "human_node_mask.pt")
    torch.save(old_to_new, output_dir / "old_to_new_node_id.pt")
    torch.save(new_to_old, output_dir / "new_to_old_node_id.pt")
    save_mapping_csv(output_dir / "mapping" / "nodeidx2proteinid.csv.gz", new_to_old, protein_ids)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved filtered dataset to: {output_dir}")


if __name__ == "__main__":
    main()
