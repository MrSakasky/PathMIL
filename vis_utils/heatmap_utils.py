"""Utilities for extracting PathMIL attention and rendering WSI heatmaps."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import percentileofscore
from tqdm import tqdm

from dataset_modules.wsi_dataset import WholeSlideRegionDataset
from utils.file_utils import save_hdf5
from utils.utils import build_sequential_loader
from wsi_core.WholeSlideImage import WholeSlideImage


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_to_percentile(score: float, reference: np.ndarray) -> float:
    return percentileofscore(reference.reshape(-1), score)


def draw_heatmap(
    scores,
    coordinates,
    slide_path=None,
    wsi_object=None,
    vis_level: int = -1,
    **kwargs,
):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
    if vis_level < 0:
        vis_level = wsi_object.getOpenSlide().get_best_level_for_downsample(32)
    return wsi_object.visHeatmap(
        scores=scores,
        coords=coordinates,
        vis_level=vis_level,
        **kwargs,
    )


def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params["seg_level"] < 0:
        seg_params["seg_level"] = (
            wsi_object.wsi.get_best_level_for_downsample(32)
        )
    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object


def compute_from_patches(
    wsi_object,
    img_transforms,
    feature_extractor,
    model=None,
    batch_size: int = 512,
    attention_save_path=None,
    reference_scores=None,
    feature_save_path=None,
    **wsi_kwargs,
):
    """Extract patch features and optionally persist PathMIL attention."""
    region_dataset = WholeSlideRegionDataset(
        wsi_object,
        transform=img_transforms,
        **wsi_kwargs,
    )
    loader = build_sequential_loader(
        region_dataset,
        batch_size=batch_size,
        num_workers=0,
    )
    print(f"Patches to process: {len(region_dataset)}; batches: {len(loader)}")

    mode = "w"
    for images, coordinates in tqdm(loader):
        images = images.to(DEVICE)
        coordinates_array = (
            coordinates.detach().cpu().numpy()
            if torch.is_tensor(coordinates)
            else np.asarray(coordinates)
        )
        coordinates_tensor = torch.as_tensor(
            coordinates_array,
            dtype=torch.long,
            device=DEVICE,
        )

        with torch.inference_mode():
            features = feature_extractor(images)
            if attention_save_path is not None and model is not None:
                attention = model(
                    features,
                    coordinates_tensor,
                    attention_only=True,
                )
                attention = attention.reshape(-1, 1).cpu().numpy()
                if reference_scores is not None:
                    attention = np.asarray(
                        [
                            score_to_percentile(value.item(), reference_scores)
                            for value in attention
                        ]
                    ).reshape(-1, 1)
                save_hdf5(
                    attention_save_path,
                    {
                        "attention_scores": attention,
                        "coords": coordinates_array,
                    },
                    mode=mode,
                )

        if feature_save_path is not None:
            save_hdf5(
                feature_save_path,
                {
                    "features": features.cpu().numpy(),
                    "coords": coordinates_array,
                },
                mode=mode,
            )
        mode = "a"

    return attention_save_path, feature_save_path, wsi_object
