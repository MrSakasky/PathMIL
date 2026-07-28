"""YAML configuration loading and command-line overrides for training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


CONFIG_SECTIONS = {
    "experiment": ("exp_code", "results_dir", "seed", "log_data"),
    "data": (
        "task",
        "data_root_dir",
        "csv_path",
        "feature_dir",
        "label_dict",
        "patient_strat",
        "ignore",
        "label_col",
    ),
    "splits": ("k", "k_start", "k_end", "split_dir", "label_frac"),
    "training": (
        "max_epochs",
        "lr",
        "reg",
        "opt",
        "classification_loss",
        "weighted_sample",
        "testing",
        "early_stopping",
        "early_stopping_patience",
        "early_stopping_min_epoch",
        "print_every",
    ),
    "model": (
        "embed_dim",
        "hidden_dim",
        "attention_dim",
        "drop_out",
    ),
    "pathmil": (
        "instance_supervision",
        "instance_loss",
        "include_negative_instances",
        "slide_loss_weight",
        "instance_sample_count",
        "projection_reg_weight",
        "path_depth",
        "path_attention_dim",
        "path_spatial_dim",
        "path_neighbor_count",
        "path_temperature",
        "path_local_weight",
        "path_spatial_weight",
        "path_global_weight",
    ),
}

DEFAULTS = {
    "exp_code": "pathmil_experiment",
    "results_dir": "./results",
    "seed": 1,
    "log_data": True,
    "task": "task_1_tumor_vs_normal",
    "data_root_dir": "Extracted_feature",
    "csv_path": "dataset_csv/train_camelyon16_all.csv",
    "feature_dir": "tumor_vs_normal_camelyon16_level0_patch256_features_resnet50",
    "label_dict": {"normal_tissue": 0, "tumor_tissue": 1},
    "patient_strat": False,
    "ignore": [],
    "label_col": "label",
    "k": 5,
    "k_start": -1,
    "k_end": -1,
    "split_dir": None,
    "label_frac": 1.0,
    "max_epochs": 200,
    "lr": 2e-4,
    "reg": 1e-5,
    "opt": "adam",
    "classification_loss": "ce",
    "weighted_sample": True,
    "testing": False,
    "early_stopping": False,
    "early_stopping_patience": 30,
    "early_stopping_min_epoch": 120,
    "print_every": 20,
    "embed_dim": 1024,
    "hidden_dim": 512,
    "attention_dim": 256,
    "drop_out": 0.25,
    "instance_supervision": True,
    "instance_loss": "svm",
    "include_negative_instances": False,
    "slide_loss_weight": 0.7,
    "instance_sample_count": 8,
    "projection_reg_weight": 1e-5,
    "path_depth": 2,
    "path_attention_dim": 256,
    "path_spatial_dim": 32,
    "path_neighbor_count": 8,
    "path_temperature": 1.0,
    "path_local_weight": 1.0,
    "path_spatial_weight": 0.25,
    "path_global_weight": 0.5,
}

KNOWN_KEYS = {key for keys in CONFIG_SECTIONS.values() for key in keys}


class ConfigError(ValueError):
    """Raised when a training configuration is invalid."""


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw_config, Mapping):
        raise ConfigError("The top level of the configuration must be a mapping.")

    unknown_sections = set(raw_config) - set(CONFIG_SECTIONS)
    if unknown_sections:
        names = ", ".join(sorted(str(name) for name in unknown_sections))
        raise ConfigError(f"Unknown configuration section(s): {names}")

    flat_config: dict[str, Any] = {}
    for section, allowed_keys in CONFIG_SECTIONS.items():
        values = raw_config.get(section, {})
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ConfigError(f"Configuration section '{section}' must be a mapping.")
        unknown_keys = set(values) - set(allowed_keys)
        if unknown_keys:
            names = ", ".join(sorted(str(name) for name in unknown_keys))
            raise ConfigError(f"Unknown key(s) in section '{section}': {names}")
        flat_config.update(values)

    validate_config(flat_config)
    return flat_config


def _validate_scalar_types(config: Mapping[str, Any]) -> None:
    integer_fields = (
        "seed",
        "k",
        "k_start",
        "k_end",
        "max_epochs",
        "early_stopping_patience",
        "early_stopping_min_epoch",
        "print_every",
        "embed_dim",
        "hidden_dim",
        "attention_dim",
        "instance_sample_count",
        "path_depth",
        "path_attention_dim",
        "path_spatial_dim",
        "path_neighbor_count",
    )
    numeric_fields = (
        "label_frac",
        "lr",
        "reg",
        "drop_out",
        "slide_loss_weight",
        "projection_reg_weight",
        "path_local_weight",
        "path_spatial_weight",
        "path_global_weight",
        "path_temperature",
    )
    boolean_fields = (
        "log_data",
        "patient_strat",
        "weighted_sample",
        "testing",
        "early_stopping",
        "instance_supervision",
        "include_negative_instances",
    )

    for key in integer_fields:
        if key in config and (
            not isinstance(config[key], int) or isinstance(config[key], bool)
        ):
            raise ConfigError(f"'{key}' must be an integer.")
    for key in numeric_fields:
        if key in config and (
            not isinstance(config[key], (int, float))
            or isinstance(config[key], bool)
        ):
            raise ConfigError(f"'{key}' must be a number.")
    for key in boolean_fields:
        if key in config and not isinstance(config[key], bool):
            raise ConfigError(f"'{key}' must be true or false.")


def validate_config(config: Mapping[str, Any]) -> None:
    unknown_keys = set(config) - KNOWN_KEYS
    if unknown_keys:
        names = ", ".join(sorted(unknown_keys))
        raise ConfigError(f"Unknown configuration key(s): {names}")

    _validate_scalar_types(config)
    path_fields = (
        "data_root_dir",
        "csv_path",
        "feature_dir",
        "results_dir",
        "split_dir",
    )
    for key in path_fields:
        if key not in config or config[key] is None:
            continue
        if not isinstance(config[key], str):
            raise ConfigError(f"'{key}' must be a path string.")
        value = config[key].strip()
        if (
            len(value) >= 3
            and value[0].lower() == "r"
            and value[1] in {"'", '"'}
            and value[-1] == value[1]
        ):
            raise ConfigError(
                f"'{key}' uses Python raw-string syntax. In YAML, write the "
                "Windows path in single quotes without the r prefix."
            )
    choices = {
        "opt": {"adam", "sgd"},
        "classification_loss": {"svm", "ce"},
        "instance_loss": {"svm", "ce"},
    }
    for key, valid_values in choices.items():
        if key in config and config[key] not in valid_values:
            expected = ", ".join(sorted(valid_values))
            raise ConfigError(
                f"Invalid '{key}': {config[key]!r}. Expected one of: {expected}."
            )

    if "label_dict" in config:
        label_dict = config["label_dict"]
        if not isinstance(label_dict, Mapping) or not label_dict:
            raise ConfigError("'label_dict' must be a non-empty mapping.")
        label_ids = list(label_dict.values())
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in label_ids
        ):
            raise ConfigError("Every value in 'label_dict' must be an integer.")
        if sorted(set(label_ids)) != list(range(len(set(label_ids)))):
            raise ConfigError(
                "'label_dict' values must be contiguous class IDs beginning at 0."
            )

    if "ignore" in config and not isinstance(config["ignore"], list):
        raise ConfigError("'ignore' must be a list.")

    positive_fields = (
        "k",
        "max_epochs",
        "early_stopping_patience",
        "embed_dim",
        "hidden_dim",
        "attention_dim",
        "instance_sample_count",
        "path_attention_dim",
        "path_spatial_dim",
        "path_neighbor_count",
        "path_temperature",
    )
    for key in positive_fields:
        if key in config and config[key] <= 0:
            raise ConfigError(f"'{key}' must be greater than 0.")

    non_negative_fields = (
        "early_stopping_min_epoch",
        "print_every",
        "path_depth",
        "reg",
        "projection_reg_weight",
    )
    for key in non_negative_fields:
        if key in config and config[key] < 0:
            raise ConfigError(f"'{key}' must be non-negative.")

    bounded_fields = {
        "label_frac": (0, 1, False, True),
        "drop_out": (0, 1, True, False),
        "slide_loss_weight": (0, 1, True, True),
    }
    for key, (lower, upper, include_lower, include_upper) in bounded_fields.items():
        if key not in config:
            continue
        value = config[key]
        lower_valid = value >= lower if include_lower else value > lower
        upper_valid = value <= upper if include_upper else value < upper
        if not lower_valid or not upper_valid:
            left = "[" if include_lower else "("
            right = "]" if include_upper else ")"
            raise ConfigError(f"'{key}' must be in the interval {left}{lower}, {upper}{right}.")


def _add_boolean_argument(
    parser: argparse._ActionsContainer,
    name: str,
    help_text: str,
) -> None:
    option = name.replace("_", "-")
    parser.add_argument(
        f"--{option}",
        dest=name,
        action="store_true",
        default=argparse.SUPPRESS,
        help=help_text,
    )
    parser.add_argument(
        f"--no-{option}",
        dest=name,
        action="store_false",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


def build_parser(default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PathMIL using a YAML configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="training YAML file",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the merged configuration and exit",
    )

    common = parser.add_argument_group("common overrides")
    common.add_argument("--exp-code", dest="exp_code", default=argparse.SUPPRESS)
    common.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    common.add_argument("--k-start", dest="k_start", type=int, default=argparse.SUPPRESS)
    common.add_argument("--k-end", dest="k_end", type=int, default=argparse.SUPPRESS)
    common.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    common.add_argument("--max-epochs", dest="max_epochs", type=int, default=argparse.SUPPRESS)

    advanced = parser.add_argument_group("advanced overrides")
    advanced.add_argument("--data-root-dir", dest="data_root_dir", default=argparse.SUPPRESS)
    advanced.add_argument("--csv-path", dest="csv_path", default=argparse.SUPPRESS)
    advanced.add_argument("--feature-dir", dest="feature_dir", default=argparse.SUPPRESS)
    advanced.add_argument("--results-dir", dest="results_dir", default=argparse.SUPPRESS)
    advanced.add_argument("--split-dir", dest="split_dir", default=argparse.SUPPRESS)
    advanced.add_argument(
        "--projection-reg-weight",
        dest="projection_reg_weight",
        type=float,
        default=argparse.SUPPRESS,
    )
    _add_boolean_argument(
        advanced,
        "instance_supervision",
        "enable instance-level supervision",
    )
    return parser


def parse_training_args(
    argv: Sequence[str] | None = None,
    default_config: Path | None = None,
) -> argparse.Namespace:
    if default_config is None:
        default_config = Path("configs/train_camelyon16.yaml")

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=default_config)
    config_only, _ = config_parser.parse_known_args(argv)
    try:
        yaml_config = load_yaml_config(config_only.config)
    except ConfigError as exc:
        config_parser.error(str(exc))

    parser = build_parser(default_config)
    merged = dict(DEFAULTS)
    merged.update(yaml_config)
    parser.set_defaults(**merged)
    args = parser.parse_args(argv)

    merged_values = {
        key: value for key, value in vars(args).items() if key in KNOWN_KEYS
    }
    try:
        validate_config(merged_values)
    except ConfigError as exc:
        parser.error(str(exc))
    return args


def to_sectioned_config(
    values: Mapping[str, Any] | argparse.Namespace,
) -> dict[str, Any]:
    if isinstance(values, argparse.Namespace):
        values = vars(values)
    return {
        section: {key: values[key] for key in keys if key in values}
        for section, keys in CONFIG_SECTIONS.items()
    }


def dump_config(values: Mapping[str, Any] | argparse.Namespace) -> str:
    return yaml.safe_dump(
        to_sectioned_config(values),
        allow_unicode=True,
        sort_keys=False,
    )
