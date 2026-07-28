"""Create stratified cross-validation split files from a training config."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from dataset_modules.dataset_generic import (
    Generic_WSI_Classification_Dataset,
    save_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_camelyon16.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.1)
    return parser


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = build_parser().parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_config = config["data"]
    split_config = config["splits"]
    experiment_config = config["experiment"]

    for name, value in (("val_frac", args.val_frac), ("test_frac", args.test_frac)):
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    if args.val_frac + args.test_frac >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")

    dataset = Generic_WSI_Classification_Dataset(
        csv_path=_project_path(data_config["csv_path"]),
        shuffle=False,
        seed=experiment_config["seed"],
        print_info=True,
        label_dict=data_config["label_dict"],
        patient_strat=data_config.get("patient_strat", False),
        ignore=data_config.get("ignore", []),
        label_col=data_config.get("label_col", "label"),
    )
    class_sizes = np.asarray(
        [len(indices) for indices in dataset.patient_cls_ids]
        if dataset.patient_strat
        else [len(indices) for indices in dataset.slide_cls_ids]
    )
    validation_counts = np.round(class_sizes * args.val_frac).astype(int)
    test_counts = np.round(class_sizes * args.test_frac).astype(int)
    if np.any(validation_counts + test_counts >= class_sizes):
        raise ValueError("A class is too small for the requested split fractions")

    label_fraction = float(split_config["label_frac"])
    label_fractions = (
        (0.1, 0.25, 0.5, 0.75, 1.0)
        if label_fraction == 0
        else (label_fraction,)
    )
    for fraction in label_fractions:
        if args.output_dir is not None:
            base_output = (
                args.output_dir / f"labels_{int(fraction * 100)}"
                if len(label_fractions) > 1
                else args.output_dir
            )
        else:
            base_output = (
                PROJECT_ROOT
                / "splits"
                / f"{data_config['task']}_{int(fraction * 100)}"
            )
        output_dir = Path(base_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.create_splits(
            k=int(split_config["k"]),
            val_num=validation_counts,
            test_num=test_counts,
            label_frac=fraction,
        )
        for fold in range(int(split_config["k"])):
            dataset.set_splits()
            descriptor = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(
                splits,
                ("train", "val", "test"),
                output_dir / f"splits_{fold}.csv",
            )
            save_splits(
                splits,
                ("train", "val", "test"),
                output_dir / f"splits_{fold}_bool.csv",
                boolean_style=True,
            )
            descriptor.to_csv(output_dir / f"splits_{fold}_descriptor.csv")
        print(f"Saved {split_config['k']} folds to {output_dir}")


if __name__ == "__main__":
    main()
