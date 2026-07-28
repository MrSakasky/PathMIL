"""Evaluate PathMIL checkpoints on configured dataset splits."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd

from dataset_modules.dataset_generic import Generic_MIL_Dataset
from utils.config_utils import DEFAULTS, load_yaml_config, validate_config
from utils.eval_utils import evaluate_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_camelyon16.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PathMIL checkpoints.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="test",
    )
    parser.add_argument("--fold", type=int, action="append")
    return parser


def _project_path(value) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_evaluation_config(config_path: Path) -> Namespace:
    values = dict(DEFAULTS)
    values.update(load_yaml_config(config_path))
    validate_config(values)

    values["data_root_dir"] = str(_project_path(values["data_root_dir"]))
    values["csv_path"] = str(_project_path(values["csv_path"]))
    feature_dir = Path(values["feature_dir"])
    if not feature_dir.is_absolute():
        feature_dir = Path(values["data_root_dir"]) / feature_dir
    values["feature_dir"] = str(feature_dir)

    if values["split_dir"] is None:
        split_name = f"{values['task']}_{int(values['label_frac'] * 100)}"
        values["split_dir"] = str(PROJECT_ROOT / "splits" / split_name)
    else:
        values["split_dir"] = str(_project_path(values["split_dir"]))

    values["n_classes"] = len(set(values["label_dict"].values()))
    return Namespace(**values)


def build_dataset(config: Namespace):
    return Generic_MIL_Dataset(
        csv_path=config.csv_path,
        data_dir=config.feature_dir,
        shuffle=False,
        seed=config.seed,
        print_info=True,
        label_dict=config.label_dict,
        patient_strat=config.patient_strat,
        ignore=config.ignore,
        label_col=config.label_col,
    )


def main(argv=None) -> None:
    cli_args = build_parser().parse_args(argv)
    config = load_evaluation_config(cli_args.config)

    checkpoint_dir = cli_args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = (
            _project_path(config.results_dir)
            / f"{config.exp_code}_s{config.seed}"
        )
    else:
        checkpoint_dir = _project_path(checkpoint_dir)

    output_dir = cli_args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "eval_results" / checkpoint_dir.name
    else:
        output_dir = _project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = cli_args.fold or list(range(config.k))
    dataset = build_dataset(config)
    split_index = {"train": 0, "validation": 1, "test": 2}
    summary_rows = []

    for fold in folds:
        if cli_args.split == "all":
            split_dataset = dataset
        else:
            split_file = Path(config.split_dir) / f"splits_{fold}.csv"
            split_datasets = dataset.return_splits(
                from_id=False,
                csv_path=str(split_file),
            )
            split_dataset = split_datasets[split_index[cli_args.split]]

        checkpoint_path = checkpoint_dir / f"s_{fold}_checkpoint.pt"
        _, _, error, auc, predictions = evaluate_dataset(
            split_dataset,
            config,
            checkpoint_path,
        )
        predictions.to_csv(
            output_dir / f"fold_{fold}.csv",
            index=False,
        )
        summary_rows.append(
            {
                "fold": fold,
                "auc": auc,
                "accuracy": 1.0 - error,
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
