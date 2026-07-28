"""On-demand patch datasets for heatmap generation."""

from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from wsi_core.util_classes import (
    isInContourV1,
    isInContourV2,
    isInContourV3_Easy,
    isInContourV3_Hard,
)


def build_contour_checker(
    method="four_pt_hard",
    contour=None,
    patch_size=None,
    center_shift=None,
):
    if method == "four_pt_hard":
        return isInContourV3_Hard(
            contour=contour,
            patch_size=patch_size,
            center_shift=center_shift,
        )
    if method == "four_pt_easy":
        return isInContourV3_Easy(
            contour=contour,
            patch_size=patch_size,
            center_shift=0.5,
        )
    if method == "center":
        return isInContourV2(contour=contour, patch_size=patch_size)
    if method == "basic":
        return isInContourV1(contour=contour)
    raise ValueError(f"Unknown contour checking method: {method}")


class WholeSlideRegionDataset(Dataset):
    """Enumerate valid patches inside the segmented tissue contours."""

    def __init__(
        self,
        wsi_object,
        top_left=None,
        bot_right=None,
        level=0,
        patch_size=(256, 256),
        step_size=(256, 256),
        contour_fn="four_pt_hard",
        transform=None,
        custom_downsample=1,
        use_center_shift=False,
    ):
        if transform is None:
            raise ValueError("A patch transform is required")
        if custom_downsample < 1:
            raise ValueError("custom_downsample must be at least 1")

        level_downsample = np.asarray(wsi_object.level_downsamples[level])
        patch_size = np.asarray(patch_size, dtype=int)
        step_size = np.asarray(step_size, dtype=int)
        reference_size = tuple((patch_size * level_downsample).astype(int))

        self.custom_downsample = custom_downsample
        self.target_patch_size = tuple(patch_size)
        if custom_downsample > 1:
            read_patch_size = tuple(
                (patch_size * level_downsample * custom_downsample).astype(int)
            )
            step_size = step_size * custom_downsample
            reference_size = read_patch_size
        else:
            read_patch_size = tuple(patch_size)

        center_shift = _center_shift(
            step_size[0] / patch_size[0], use_center_shift
        )
        coordinates = []
        for index, contour in enumerate(wsi_object.contours_tissue):
            print(f"Processing contour {index + 1}/{len(wsi_object.contours_tissue)}")
            checker = build_contour_checker(
                contour_fn,
                contour,
                reference_size[0],
                center_shift,
            )
            result, _ = wsi_object.process_contour(
                contour,
                wsi_object.holes_tissue[index],
                level,
                "",
                patch_size=read_patch_size[0],
                step_size=int(step_size[0]),
                contour_fn=checker,
                use_padding=True,
                top_left=top_left,
                bot_right=bot_right,
            )
            if len(result) > 0:
                coordinates.append(result["coords"])

        self.coords = (
            np.vstack(coordinates)
            if coordinates
            else np.empty((0, 2), dtype=np.int64)
        )
        self.wsi = wsi_object.wsi
        self.level = level
        self.patch_size = read_patch_size
        self.transform = transform
        print(f"Filtered a total of {len(self.coords)} coordinates")

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int):
        coordinate = self.coords[index]
        patch = self.wsi.read_region(
            tuple(coordinate),
            self.level,
            self.patch_size,
        ).convert("RGB")
        if self.custom_downsample > 1:
            patch = patch.resize(self.target_patch_size)
        return self.transform(patch).unsqueeze(0), coordinate


def _center_shift(step_ratio: float, enabled: bool) -> float:
    if not enabled:
        return 0.0
    overlap = 1.0 - float(step_ratio)
    if overlap < 0.25:
        return 0.375
    if overlap < 0.95:
        return 0.5
    return 0.625
