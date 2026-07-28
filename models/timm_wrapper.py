"""Thin wrappers around timm feature encoders."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn


class TimmCNNEncoder(nn.Module):
    """Return one pooled feature vector per input image."""

    def __init__(
        self,
        model_name: str = "resnet50.tv_in1k",
        *,
        pretrained: bool = True,
        checkpoint_path: str | Path | None = None,
        checkpoint_format: str = "state_dict",
        pool: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        kwargs = {
            "features_only": True,
            "out_indices": (3,),
            "num_classes": 0,
            "pretrained": pretrained,
            **(model_kwargs or {}),
        }
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        self.pool = nn.AdaptiveAvgPool2d(1) if pool else None
        if checkpoint_path is not None:
            self._load_checkpoint(Path(checkpoint_path), checkpoint_format)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.model(inputs)
        if isinstance(features, (list, tuple)):
            if len(features) != 1:
                raise RuntimeError(
                    f"Expected one feature map, received {len(features)}"
                )
            features = features[0]
        if self.pool is not None:
            features = self.pool(features).flatten(1)
        return features

    def _load_checkpoint(self, path: Path, checkpoint_format: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Encoder checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, nn.Module):
            state_dict = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if checkpoint_format == "state_dict":
            prepared = state_dict
        elif checkpoint_format == "backbone":
            prepared = {
                key.removeprefix("backbone."): value
                for key, value in state_dict.items()
                if key.startswith("backbone.")
            }
        elif checkpoint_format == "torchvision_partial":
            prepared = _convert_torchvision_resnet(state_dict, self.model.state_dict())
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")

        if not prepared:
            raise ValueError(
                f"No compatible encoder weights were found in checkpoint: {path}"
            )
        incompatible = self.model.load_state_dict(prepared, strict=False)
        print(
            f"Loaded encoder checkpoint {path}; "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )


def _convert_torchvision_resnet(
    state_dict: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Map the first four torchvision ResNet stages to timm feature keys."""

    prefix_map = {
        "conv1": "conv1",
        "bn1": "bn1",
        "layer1": "layer1",
        "layer2": "layer2",
        "layer3": "layer3",
    }
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.").removeprefix("backbone.")
        for source_prefix, target_prefix in prefix_map.items():
            if clean_key == source_prefix or clean_key.startswith(f"{source_prefix}."):
                target_key = clean_key.replace(source_prefix, target_prefix, 1)
                if (
                    target_key in target_state
                    and target_state[target_key].shape == value.shape
                ):
                    converted[target_key] = value
                break
    return converted
