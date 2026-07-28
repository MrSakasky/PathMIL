"""Training entry point for PathMIL experiments."""

from __future__ import annotations

import os
from pathlib import Path

from utils.config_utils import dump_config, parse_training_args


PROJECT_ROOT = Path(__file__).resolve().parent


def seed_torch(seed: int = 7) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _project_path(path_value: str | Path) -> Path:
    """Resolve project paths consistently, independent of the launch directory."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_paths(args):
    """Resolve input/output paths for the selected experiment."""
    args.data_root_dir = str(_project_path(args.data_root_dir))
    args.csv_path = str(_project_path(args.csv_path))

    feature_dir = Path(args.feature_dir).expanduser()
    if not feature_dir.is_absolute():
        feature_dir = Path(args.data_root_dir) / feature_dir
    args.feature_dir = str(feature_dir)

    if args.split_dir is None:
        split_name = f"{args.task}_{int(args.label_frac * 100)}"
        split_dir = PROJECT_ROOT / "splits" / split_name
    else:
        split_dir = Path(args.split_dir).expanduser()
        if not split_dir.is_absolute():
            if split_dir.parts and split_dir.parts[0] == "splits":
                split_dir = PROJECT_ROOT / split_dir
            else:
                split_dir = PROJECT_ROOT / "splits" / split_dir
    args.split_dir = str(split_dir)

    results_root = _project_path(args.results_dir)
    experiment_dir = results_root / f"{args.exp_code}_s{args.seed}"
    args.results_dir = str(experiment_dir)
    return args


def validate_runtime_config(args) -> None:
    """Validate settings that depend on multiple merged configuration values."""
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    if not 0 <= start < end <= args.k:
        raise ValueError(
            "Fold range must satisfy 0 <= k_start < k_end <= k "
            "(use -1 for the default boundary)."
        )

    if len(set(args.label_dict.values())) < 2:
        raise ValueError("MIL training requires at least two classes.")


def build_dataset(args):
    """Construct a dataset entirely from the merged YAML/CLI configuration."""
    from dataset_modules.dataset_generic import Generic_MIL_Dataset

    args.n_classes = len(set(args.label_dict.values()))
    return Generic_MIL_Dataset(
        csv_path=args.csv_path,
        data_dir=args.feature_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict=args.label_dict,
        patient_strat=args.patient_strat,
        ignore=args.ignore,
        label_col=args.label_col,
    )


def build_settings(args) -> dict:
    """Create the compact settings record saved with every experiment."""
    settings = {
        "num_splits": args.k,
        "k_start": args.k_start,
        "k_end": args.k_end,
        "task": args.task,
        "max_epochs": args.max_epochs,
        "results_dir": args.results_dir,
        "lr": args.lr,
        "experiment": args.exp_code,
        "reg": args.reg,
        "label_frac": args.label_frac,
        "classification_loss": args.classification_loss,
        "seed": args.seed,
        "drop_out": args.drop_out,
        "weighted_sample": args.weighted_sample,
        "opt": args.opt,
        "split_dir": args.split_dir,
        "csv_path": args.csv_path,
        "feature_dir": args.feature_dir,
        "n_classes": args.n_classes,
        "instance_supervision": args.instance_supervision,
        "instance_loss": args.instance_loss,
        "slide_loss_weight": args.slide_loss_weight,
        "projection_reg_weight": args.projection_reg_weight,
        "path_depth": args.path_depth,
        "path_neighbor_count": args.path_neighbor_count,
        "path_temperature": args.path_temperature,
    }
    return settings


def run_cross_validation(args, dataset) -> None:
    """Run the configured folds and continuously persist the summary."""
    import numpy as np
    import pandas as pd

    from utils.core_utils import train_fold
    from utils.file_utils import save_pkl

    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start, end)

    save_name = "summary.csv" if len(folds) == args.k else f"summary_partial_{start}_{end}.csv"
    summary_path = Path(args.results_dir) / save_name

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []

    for fold in folds:
        seed_torch(args.seed)
        train_dataset, validation_dataset, test_dataset = dataset.return_splits(
            from_id=False,
            csv_path=str(Path(args.split_dir) / f"splits_{fold}.csv"),
        )
        results, test_auc, val_auc, test_acc, val_acc = train_fold(
            (train_dataset, validation_dataset, test_dataset),
            fold,
            args,
        )
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        save_pkl(
            str(Path(args.results_dir) / f"split_{fold}_results.pkl"),
            results,
        )

        completed_folds = folds[: len(all_test_auc)]
        summary = pd.DataFrame(
            {
                "folds": completed_folds,
                "test_auc": all_test_auc,
                "val_auc": all_val_auc,
                "test_acc": all_test_acc,
                "val_acc": all_val_acc,
            }
        )
        summary.to_csv(summary_path, index=False)


def run_training(args) -> None:
    """Initialize the experiment, record configuration and start training."""
    print("\nLoad Dataset")
    dataset = build_dataset(args)
    settings = build_settings(args)

    experiment_dir = Path(args.results_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with (experiment_dir / f"experiment_{args.exp_code}.txt").open(
        "w", encoding="utf-8"
    ) as stream:
        print(settings, file=stream)
    with (experiment_dir / "config_effective.yaml").open(
        "w", encoding="utf-8"
    ) as stream:
        stream.write(args.effective_config_yaml)

    print("\nSettings")
    for key, value in settings.items():
        print(f"{key}:  {value}")

    seed_torch(args.seed)
    run_cross_validation(args, dataset)
    print("finished!")
    print("end script")


def cli(argv=None) -> None:
    args = parse_training_args(
        argv=argv,
        default_config=PROJECT_ROOT / "configs" / "train_camelyon16.yaml",
    )
    try:
        validate_runtime_config(args)
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    if args.print_config:
        print(dump_config(args), end="")
        return

    args.effective_config_yaml = dump_config(args)
    prepare_paths(args)

    if not Path(args.csv_path).is_file():
        raise SystemExit(f"Input error: dataset CSV does not exist: {args.csv_path}")
    if not Path(args.feature_dir).is_dir():
        raise SystemExit(f"Input error: feature directory does not exist: {args.feature_dir}")
    if not Path(args.split_dir).is_dir():
        raise SystemExit(f"Input error: split directory does not exist: {args.split_dir}")

    run_training(args)


if __name__ == "__main__":
    cli()
