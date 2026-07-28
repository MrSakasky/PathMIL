"""Checkpoint loading and dataset evaluation for PathMIL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from utils.core_utils import build_model, create_loss, evaluate_model
from utils.utils import build_sequential_loader


CHECKPOINT_KEY_RENAMES = (
    ("path_encoder.Wq.", "path_encoder.query_projection."),
    ("path_encoder.Wk.", "path_encoder.key_projection."),
    ("path_encoder.Wv.", "path_encoder.value_projection."),
    ("path_encoder.W_global.", "path_encoder.global_projection."),
    ("path_encoder.spatial_mlp.", "path_encoder.spatial_projection."),
    ("path_intra_attn.attention_a.", "path_attention.attention."),
    ("path_intra_attn.attention_b.", "path_attention.gate."),
    ("path_intra_attn.attention_c.", "path_attention.output."),
    ("patch_proj.", "patch_projection."),
    ("path_proj.", "path_projection."),
    ("fusion_mlp.", "feature_fusion."),
    ("pos_layer.proj7.", "positional_embedding.convolution_7."),
    ("pos_layer.proj5.", "positional_embedding.convolution_5."),
    ("pos_layer.proj3.", "positional_embedding.convolution_3."),
    ("pos_layer.token_gate.", "positional_embedding.token_gate."),
    ("attention_net.0.", "attention_projection.0."),
    ("attention_net.3.attention_a.", "slide_attention.attention."),
    ("attention_net.3.attention_b.", "slide_attention.gate."),
    ("attention_net.3.attention_c.", "slide_attention.output."),
    ("classifiers.", "slide_classifier."),
)


def normalize_checkpoint_keys(state_dict):
    """Translate historical parameter names to the current PathMIL modules."""
    normalized = {}
    for original_key, value in state_dict.items():
        key = original_key[7:] if original_key.startswith("module.") else original_key
        key = key.replace(".module.", ".")
        if "instance_loss_fn" in key:
            continue
        for old_prefix, new_prefix in CHECKPOINT_KEY_RENAMES:
            if key.startswith(old_prefix):
                key = new_prefix + key[len(old_prefix):]
                break
        normalized[key] = value
    return normalized


def load_trained_model(args, checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args, device=device)
    state_dict = torch.load(
        Path(checkpoint_path),
        map_location=device,
    )
    model.load_state_dict(normalize_checkpoint_keys(state_dict), strict=True)
    model.eval()
    return model


def evaluate_dataset(dataset, args, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args, checkpoint_path, device)
    loader = build_sequential_loader(dataset)
    loss = create_loss(args.classification_loss, args.n_classes)
    result = evaluate_model(
        model,
        loader,
        loss,
        args,
        split_name="evaluation",
        collect_records=True,
    )

    rows = []
    for record in (result.records or {}).values():
        row = {
            "slide_id": record["slide_id"],
            "label": record["label"],
        }
        row.update(
            {
                f"probability_{class_index}": probability
                for class_index, probability in enumerate(record["prob"])
            }
        )
        rows.append(row)

    return model, result.records, result.error, result.auc, pd.DataFrame(rows)
