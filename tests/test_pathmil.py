from argparse import Namespace
import tempfile
import unittest

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from models import PathMIL
from models.model_PathMIL import DynamicPathEncoder
from utils.core_utils import (
    DEVICE,
    create_loss,
    evaluate_model,
    train_fold,
    train_one_epoch,
)
from utils.eval_utils import evaluate_dataset, normalize_checkpoint_keys


def build_small_model() -> PathMIL:
    return PathMIL(
        embed_dim=8,
        hidden_dim=8,
        attention_dim=4,
        dropout=0.0,
        n_classes=2,
        instance_sample_count=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        path_depth=1,
        path_attention_dim=4,
        path_spatial_dim=4,
        path_neighbor_count=3,
    )


def build_args() -> Namespace:
    return Namespace(
        n_classes=2,
        instance_supervision=True,
        slide_loss_weight=0.7,
        projection_reg_weight=1e-5,
        print_every=0,
    )


def build_batch():
    features = torch.randn(6, 8)
    labels = torch.tensor([1])
    coordinates = torch.tensor(
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    )
    return features, labels, coordinates


class SyntheticSplit(Dataset):
    def __init__(self):
        self.samples = [
            (torch.randn(6, 8), 0, torch.arange(12).reshape(6, 2)),
            (torch.randn(6, 8), 1, torch.arange(12).reshape(6, 2)),
        ]
        self.slide_data = pd.DataFrame(
            {
                "slide_id": ["slide_0", "slide_1"],
                "label": [0, 1],
            }
        )
        self.slide_cls_ids = [torch.tensor([0]), torch.tensor([1])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def getlabel(self, index):
        return self.samples[index][1]


class ScalarRecorder:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)


class PathMILTests(unittest.TestCase):
    def test_dynamic_paths_mask_visited_nodes_and_use_straight_through_gradients(self):
        torch.manual_seed(3)
        encoder = DynamicPathEncoder(
            embed_dim=8,
            attention_dim=4,
            spatial_dim=4,
            depth=3,
            neighbor_count=4,
            temperature=0.7,
        )
        features = torch.randn(9, 8, requires_grad=True)
        coordinates = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [0, 1],
                [1, 1],
                [2, 1],
                [0, 2],
                [1, 2],
                [2, 2],
            ]
        )

        encoded_paths, path_indices = encoder(
            features,
            coordinates,
            return_indices=True,
        )
        projected_values = encoder.value_projection(features)
        selected_values = projected_values[path_indices]

        for indices in path_indices:
            self.assertEqual(len(indices.unique()), indices.numel())
        self.assertTrue(torch.allclose(encoded_paths, selected_values, atol=1e-6))

        encoded_paths.square().mean().backward()
        self.assertGreater(
            encoder.query_projection.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            encoder.key_projection.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_path_dead_ends_expand_to_unvisited_candidates(self):
        encoder = DynamicPathEncoder(
            embed_dim=4,
            attention_dim=2,
            spatial_dim=2,
            depth=10,
            neighbor_count=1,
        )
        features = torch.randn(5, 4)
        coordinates = torch.tensor(
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
        )

        encoded_paths, path_indices = encoder(
            features,
            coordinates,
            return_indices=True,
        )

        self.assertEqual(encoded_paths.shape, (5, 5, 4))
        self.assertEqual(path_indices.shape, (5, 5))
        for indices in path_indices:
            self.assertEqual(len(indices.unique()), indices.numel())

    def test_svm_loss_uses_the_logits_device(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            logits = torch.tensor(
                [[1.0, -0.5], [-0.25, 0.75]],
                device=device,
                requires_grad=True,
            )
            targets = torch.tensor([0, 1], device=device)
            loss = create_loss("svm", 2)(logits, targets)
            loss.backward()

            self.assertEqual(loss.device, logits.device)
            self.assertEqual(logits.grad.device, logits.device)
            self.assertTrue(torch.isfinite(loss))

    def test_historical_checkpoint_keys_are_migrated(self):
        state = {
            "path_encoder.Wq.weight": torch.randn(4, 8),
            "attention_net.3.attention_c.bias": torch.randn(1),
            "classifiers.weight": torch.randn(2, 8),
        }

        normalized = normalize_checkpoint_keys(state)

        self.assertIn("path_encoder.query_projection.weight", normalized)
        self.assertIn("slide_attention.output.bias", normalized)
        self.assertIn("slide_classifier.weight", normalized)

    def test_forward_output_and_instance_supervision(self):
        model = build_small_model()
        features, labels, coordinates = build_batch()

        output = model(
            features,
            coordinates,
            label=labels,
            evaluate_instances=True,
        )

        self.assertEqual(output["logits"].shape, (1, 2))
        self.assertEqual(output["probabilities"].shape, (1, 2))
        self.assertEqual(output["predictions"].shape, (1, 1))
        self.assertTrue(torch.isfinite(output["instance_loss"]))

    def test_train_and_evaluate_share_the_same_model_contract(self):
        model = build_small_model().to(DEVICE)
        args = build_args()
        batch = build_batch()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss = nn.CrossEntropyLoss()
        writer = ScalarRecorder()

        metrics = train_one_epoch(
            0,
            model,
            [batch],
            optimizer,
            loss,
            args,
            writer,
        )
        result = evaluate_model(
            model,
            [batch],
            loss,
            args,
            split_name="validation",
            epoch=0,
            writer=writer,
        )

        self.assertIn("total_loss", metrics)
        self.assertIn("gradient_norm", metrics)
        self.assertIn("learning_rate", metrics)
        self.assertIn("epoch_seconds", metrics)
        self.assertIn("train/total_loss", writer.scalars)
        self.assertIn("train/gradient_norm", writer.scalars)
        self.assertIn("train/learning_rate", writer.scalars)
        self.assertIn("validation/loss", writer.scalars)
        self.assertIn("validation/accuracy", writer.scalars)
        self.assertIn("validation/auc", writer.scalars)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)))
        self.assertEqual(result.class_accuracy.count.sum(), 1)

    def test_single_fold_training_orchestration(self):
        with tempfile.TemporaryDirectory() as result_dir:
            args = Namespace(
                results_dir=result_dir,
                log_data=False,
                classification_loss="ce",
                instance_loss="ce",
                embed_dim=8,
                hidden_dim=8,
                attention_dim=4,
                drop_out=0.0,
                n_classes=2,
                instance_sample_count=2,
                include_negative_instances=False,
                path_depth=1,
                path_attention_dim=4,
                path_spatial_dim=4,
                path_neighbor_count=3,
                path_local_weight=1.0,
                path_spatial_weight=0.25,
                path_global_weight=0.5,
                lr=1e-4,
                reg=0.0,
                opt="adam",
                testing=False,
                weighted_sample=False,
                early_stopping=False,
                early_stopping_patience=2,
                early_stopping_min_epoch=0,
                max_epochs=1,
                instance_supervision=True,
                slide_loss_weight=0.7,
                projection_reg_weight=1e-5,
                print_every=0,
            )
            split = SyntheticSplit()
            records, test_auc, validation_auc, test_accuracy, validation_accuracy = (
                train_fold((split, split, split), 0, args)
            )

            self.assertEqual(set(records), {"slide_0", "slide_1"})
            self.assertTrue(torch.isfinite(torch.tensor(test_auc)))
            self.assertTrue(torch.isfinite(torch.tensor(validation_auc)))
            self.assertGreaterEqual(test_accuracy, 0.0)
            self.assertGreaterEqual(validation_accuracy, 0.0)

            _, _, evaluation_error, evaluation_auc, predictions = (
                evaluate_dataset(
                    split,
                    args,
                    f"{result_dir}/s_0_checkpoint.pt",
                )
            )
            self.assertEqual(len(predictions), 2)
            self.assertTrue(torch.isfinite(torch.tensor(evaluation_auc)))
            self.assertGreaterEqual(evaluation_error, 0.0)


if __name__ == "__main__":
    unittest.main()
