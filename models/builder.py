"""Build image encoders used during patch feature extraction."""

from __future__ import annotations

import importlib.util
import os
from functools import partial
from pathlib import Path

import timm
import torch

from models.timm_wrapper import TimmCNNEncoder
from utils.constants import MODEL_NORMALIZATION
from utils.transform_utils import get_eval_transforms


SUPPORTED_ENCODERS = (
    "resnet50_trunc",
    "resnet50_brain",
    "resnet50_brain2",
    "uni_v1",
    "conch_v1",
)


def _conch_availability() -> tuple[bool, str]:
    checkpoint = os.environ.get("CONCH_CKPT_PATH", "")
    try:
        installed = importlib.util.find_spec("conch.open_clip_custom") is not None
    except ModuleNotFoundError:
        installed = False
    return installed and bool(checkpoint), checkpoint


def _uni_checkpoint() -> tuple[bool, str]:
    checkpoint = os.environ.get("UNI_CKPT_PATH", "")
    return bool(checkpoint), checkpoint


def _require_checkpoint(
    model_name: str, checkpoint_path: str | Path | None
) -> Path:
    if checkpoint_path is None:
        raise ValueError(f"{model_name} requires --encoder-checkpoint")
    return Path(checkpoint_path)


def _build_encoder(
    model_name: str,
    checkpoint_path: str | Path | None,
):
    if model_name == "resnet50_trunc":
        return TimmCNNEncoder()
    if model_name == "resnet50_brain":
        return TimmCNNEncoder(
            checkpoint_path=_require_checkpoint(model_name, checkpoint_path),
            checkpoint_format="torchvision_partial",
        )
    if model_name == "resnet50_brain2":
        return TimmCNNEncoder(
            checkpoint_path=_require_checkpoint(model_name, checkpoint_path),
            checkpoint_format="backbone",
        )
    if model_name == "uni_v1":
        available, env_checkpoint = _uni_checkpoint()
        path = Path(checkpoint_path) if checkpoint_path else Path(env_checkpoint)
        if not available and checkpoint_path is None:
            raise RuntimeError(
                "UNI requires --encoder-checkpoint or UNI_CKPT_PATH"
            )
        model = timm.create_model(
            "vit_large_patch16_224",
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        return model
    if model_name == "conch_v1":
        available, env_checkpoint = _conch_availability()
        try:
            installed = importlib.util.find_spec("conch.open_clip_custom") is not None
        except ModuleNotFoundError:
            installed = False
        path = Path(checkpoint_path) if checkpoint_path else Path(env_checkpoint)
        if not installed:
            raise RuntimeError("CONCH package is not installed")
        if not available and checkpoint_path is None:
            raise RuntimeError(
                "CONCH requires its package and --encoder-checkpoint "
                "or CONCH_CKPT_PATH"
            )
        from conch.open_clip_custom import create_model_from_pretrained

        model, _ = create_model_from_pretrained("conch_ViT-B-16", str(path))
        model.forward = partial(
            model.encode_image,
            proj_contrast=False,
            normalize=False,
        )
        return model
    raise ValueError(
        f"Unknown encoder '{model_name}'. Choose from {SUPPORTED_ENCODERS}"
    )


def get_encoder(
    model_name: str,
    target_img_size: int = 224,
    checkpoint_path: str | Path | None = None,
):
    model = _build_encoder(model_name, checkpoint_path)
    normalization = MODEL_NORMALIZATION[model_name]
    transform = get_eval_transforms(
        mean=normalization["mean"],
        std=normalization["std"],
        target_img_size=target_img_size,
    )
    return model, transform
