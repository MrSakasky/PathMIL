"""Shared data-loading, optimization and dataset-splitting utilities."""

from __future__ import annotations

from collections import deque
from itertools import islice
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
    WeightedRandomSampler,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices) -> None:
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def collate_mil(batch):
    """Collate a one-bag batch with either labels or coordinates."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    first = batch[0]
    if not isinstance(first, (list, tuple)):
        raise ValueError(f"Unexpected batch element type: {type(first)}")

    if len(first) == 3:
        features = torch.cat([item[0] for item in batch], dim=0)
        labels = torch.as_tensor([int(item[1]) for item in batch], dtype=torch.long)
        coordinates = torch.cat(
            [torch.as_tensor(item[2]) for item in batch],
            dim=0,
        )
        return features, labels, coordinates

    if len(first) == 2:
        features = torch.cat([item[0] for item in batch], dim=0)
        coordinates = []
        for _, item_coordinates in batch:
            tensor = torch.as_tensor(item_coordinates)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            coordinates.append(tensor)
        return features, torch.cat(coordinates, dim=0)

    raise ValueError(f"Unexpected batch element length: {len(first)}")


def collate_features(batch):
    features = torch.cat([item[0] for item in batch], dim=0)
    coordinates = np.vstack([item[1] for item in batch])
    return features, coordinates


def build_sequential_loader(dataset, batch_size: int = 1, num_workers: int = 1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collate_mil,
        num_workers=num_workers if DEVICE.type == "cuda" else 0,
        pin_memory=DEVICE.type == "cuda",
    )


def build_split_loader(
    dataset,
    training: bool = False,
    testing: bool = False,
    weighted: bool = False,
):
    """Build the sampler and loader for a training, validation or test split."""
    if dataset is None:
        raise ValueError("Cannot create a loader for an empty split.")

    if testing:
        sample_count = max(1, int(len(dataset) * 0.1))
        indices = np.random.choice(len(dataset), sample_count, replace=False)
        selected_sampler = SubsetSequentialSampler(indices)
    elif training and weighted:
        weights = balanced_sample_weights(dataset)
        selected_sampler = WeightedRandomSampler(weights, len(weights))
    elif training:
        selected_sampler = RandomSampler(dataset)
    else:
        selected_sampler = SequentialSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=1,
        sampler=selected_sampler,
        collate_fn=collate_mil,
        num_workers=1 if DEVICE.type == "cuda" else 0,
        pin_memory=DEVICE.type == "cuda",
    )


def create_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
    if args.opt == "adam":
        return torch.optim.Adam(
            parameters,
            lr=args.lr,
            weight_decay=args.reg,
        )
    if args.opt == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.reg,
        )
    raise ValueError(f"Unsupported optimizer: {args.opt}")


def print_model_summary(model: nn.Module) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(model)
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")


def generate_split(
    cls_ids,
    val_num,
    test_num,
    samples,
    n_splits: int = 5,
    seed: int = 7,
    label_frac: float = 1.0,
    custom_test_ids=None,
):
    if not 0 < label_frac <= 1:
        raise ValueError("label_frac must be in (0, 1]")
    indices = np.arange(samples, dtype=int)
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    random = np.random.RandomState(seed)
    for _ in range(n_splits):
        validation_ids = []
        test_ids = list(custom_test_ids) if custom_test_ids is not None else []
        training_ids = []

        for class_index in range(len(val_num)):
            available = np.intersect1d(cls_ids[class_index], indices)
            required = int(val_num[class_index])
            if custom_test_ids is None:
                required += int(test_num[class_index])
            if required > len(available):
                raise ValueError(
                    f"Class {class_index} has {len(available)} samples, "
                    f"but {required} validation/test samples were requested"
                )
            class_validation_ids = random.choice(
                available,
                val_num[class_index],
                replace=False,
            )
            remaining = np.setdiff1d(available, class_validation_ids)
            validation_ids.extend(class_validation_ids)

            if custom_test_ids is None:
                class_test_ids = random.choice(
                    remaining,
                    test_num[class_index],
                    replace=False,
                )
                remaining = np.setdiff1d(remaining, class_test_ids)
                test_ids.extend(class_test_ids)

            sample_count = (
                len(remaining)
                if label_frac == 1
                else math.ceil(len(remaining) * label_frac)
            )
            if sample_count == len(remaining):
                selected_training_ids = remaining
            else:
                selected_training_ids = random.choice(
                    remaining,
                    sample_count,
                    replace=False,
                )
            training_ids.extend(selected_training_ids)

        yield training_ids, validation_ids, test_ids


def nth(iterator, n, default=None):
    if n is None:
        return deque(iterator, maxlen=0)
    return next(islice(iterator, n, None), default)


def classification_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return 1.0 - predictions.float().eq(targets.float()).float().mean().item()


def balanced_sample_weights(dataset) -> torch.DoubleTensor:
    sample_count = len(dataset)
    class_weights = []
    for class_indices in dataset.slide_cls_ids:
        if len(class_indices) == 0:
            raise ValueError("Weighted sampling requires every class in the split.")
        class_weights.append(sample_count / len(class_indices))

    weights = [
        class_weights[int(dataset.getlabel(index))]
        for index in range(sample_count)
    ]
    return torch.as_tensor(weights, dtype=torch.double)


def initialize_weights(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm1d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
