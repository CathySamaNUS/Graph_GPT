import argparse
from pprint import pformat

import torch
from omegaconf import OmegaConf

from src.conf.base_configs import TrainingConfig
from src.conf.tokenization.token_configs import DataConfig
from src.data.data_sources import read_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dry-run GraphGPT dataset reading for ogbl-ppa-human."
    )
    parser.add_argument("--data_dir", type=str, default="data/OGB")
    parser.add_argument("--data_path", type=str, default="ogbl_ppa_human")
    parser.add_argument("--dataset", type=str, default="ogbl-ppa-human")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_neighbors", type=int, default=14)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument(
        "--skip_sample",
        action="store_true",
        help="Only build the dataset objects without materializing a sample.",
    )
    return parser.parse_args()


def tensor_brief(value):
    if not torch.is_tensor(value):
        return repr(value)
    return f"shape={tuple(value.shape)}, dtype={value.dtype}"


def data_brief(data):
    lines = [f"num_nodes={data.num_nodes}"]
    for key, value in data:
        if key == "num_nodes":
            continue
        lines.append(f"{key}: {tensor_brief(value)}")
    return "\n".join(lines)


def main():
    args = parse_args()

    sampling = OmegaConf.create(
        {
            "edge_ego": {
                "depth_neighbors": [[args.depth, args.num_neighbors]],
                "neg_ratio": args.neg_ratio,
                "replace": args.replace,
                "percent": 100,
                "method": {"name": "global"},
            }
        }
    )
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        data_path=args.data_path,
        dataset=args.dataset,
        sampling=sampling,
        return_valid_test=True,
    )
    train_cfg = TrainingConfig(pretrain_mode=False, task_type="edge")

    print("Reading dataset with config:")
    print(
        pformat(
            {
                "dataset": data_cfg.dataset,
                "data_dir": data_cfg.data_dir,
                "data_path": data_cfg.data_path,
                "sampling": OmegaConf.to_container(sampling, resolve=True),
            }
        )
    )

    train_dataset, valid_dataset, test_dataset, raw_dataset = read_dataset(
        name=data_cfg.dataset,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        with_prob=False,
        true_valid=train_cfg.ft_eval.true_valid,
    )

    print("\nLoaded raw graph:")
    print(raw_dataset[0])
    print(
        {
            "train_len": len(train_dataset),
            "valid_len": len(valid_dataset),
            "test_len": len(test_dataset),
        }
    )

    if args.skip_sample:
        return

    sample_idx = args.sample_index
    if hasattr(train_dataset, "sampler") and train_dataset.sampler is not None:
        if not train_dataset.sampler:
            raise RuntimeError("train_dataset.sampler is empty")
        sample_idx = train_dataset.sampler[min(args.sample_index, len(train_dataset.sampler) - 1)]

    idx, sample = train_dataset[sample_idx]
    print("\nOne train sample loaded through GraphGPT:")
    print({"dataset_idx": idx})
    print(data_brief(sample))


if __name__ == "__main__":
    main()
