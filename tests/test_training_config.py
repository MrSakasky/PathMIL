from pathlib import Path
import unittest

import yaml

import main
from utils.config_utils import ConfigError, dump_config, parse_training_args, validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_camelyon16.yaml"


class TrainingConfigTests(unittest.TestCase):
    def test_default_yaml_is_loaded(self):
        args = parse_training_args([], default_config=DEFAULT_CONFIG)

        self.assertEqual(args.exp_code, "pathmil_camelyon16")
        self.assertEqual(args.embed_dim, 1024)
        self.assertEqual(args.label_dict["tumor_tissue"], 1)

    def test_cli_overrides_yaml(self):
        args = parse_training_args(
            [
                "--seed",
                "7",
                "--lr",
                "0.001",
                "--k-start",
                "1",
                "--k-end",
                "2",
                "--projection-reg-weight",
                "0.001",
            ],
            default_config=DEFAULT_CONFIG,
        )

        self.assertEqual(args.seed, 7)
        self.assertEqual(args.lr, 0.001)
        self.assertEqual((args.k_start, args.k_end), (1, 2))
        self.assertEqual(args.projection_reg_weight, 0.001)

    def test_dumped_config_uses_public_sections(self):
        args = parse_training_args([], default_config=DEFAULT_CONFIG)
        dumped = yaml.safe_load(dump_config(args))

        self.assertEqual(
            set(dumped),
            {"experiment", "data", "splits", "training", "model", "pathmil"},
        )
        self.assertNotIn("config", dumped["experiment"])

    def test_invalid_label_ids_are_rejected(self):
        with self.assertRaises(ConfigError):
            validate_config({"label_dict": {"normal": 0, "tumor": 2}})

    def test_invalid_fold_range_is_rejected(self):
        args = parse_training_args(
            ["--k-start", "4", "--k-end", "2"],
            default_config=DEFAULT_CONFIG,
        )

        with self.assertRaises(ValueError):
            main.validate_runtime_config(args)

    def test_python_raw_string_path_is_rejected(self):
        with self.assertRaises(ConfigError):
            validate_config({"feature_dir": 'r"G:\\features"'})

    def test_path_temperature_must_be_positive(self):
        with self.assertRaises(ConfigError):
            validate_config({"path_temperature": 0.0})


if __name__ == "__main__":
    unittest.main()
