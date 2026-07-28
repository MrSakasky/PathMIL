"""Datasets used by the whole-slide feature extraction pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class SlideListDataset(Dataset):
    """Read slide identifiers from the ``slide_id`` column of a CSV file."""

    def __init__(self, csv_path: str | Path):
        frame = pd.read_csv(csv_path, dtype={"slide_id": str})
        if "slide_id" not in frame:
            raise ValueError(f"{csv_path} must contain a 'slide_id' column")
        self.slide_ids = frame["slide_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, index: int) -> str:
        return self.slide_ids[index]


class MaterializedPatchDataset(Dataset):
    """Read patch pixels and their coordinates from an HDF5 image bag."""

    def __init__(self, file_path: str | Path, transform: Callable):
        self.file_path = str(file_path)
        self.transform = transform
        with h5py.File(self.file_path, "r") as handle:
            self.length = len(handle["imgs"])
            self.metadata = dict(handle["imgs"].attrs)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        with h5py.File(self.file_path, "r") as handle:
            image = Image.fromarray(handle["imgs"][index]).convert("RGB")
            coordinate = handle["coords"][index]
        return {"img": self.transform(image), "coord": coordinate}


class CoordinatePatchDataset(Dataset):
    """Read coordinates from HDF5 and fetch patch pixels directly from a WSI."""

    def __init__(
        self,
        file_path: str | Path,
        wsi,
        transform: Callable,
        patch_level: int | None = None,
        patch_size: int | None = None,
    ):
        self.file_path = str(file_path)
        self.wsi = wsi
        self.transform = transform
        with h5py.File(self.file_path, "r") as handle:
            coordinates = handle["coords"]
            self.length = len(coordinates)
            self.metadata = dict(coordinates.attrs)

        stored_level = self.metadata.get("patch_level")
        stored_size = self.metadata.get("patch_size")
        self.patch_level = _resolve_patch_value(
            "patch_level", patch_level, stored_level
        )
        self.patch_size = _resolve_patch_value("patch_size", patch_size, stored_size)
        if self.patch_level >= self.wsi.level_count:
            raise ValueError(
                f"patch_level={self.patch_level} is unavailable; "
                f"slide has {self.wsi.level_count} levels"
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        with h5py.File(self.file_path, "r") as handle:
            coordinate = handle["coords"][index]
        image = self.wsi.read_region(
            coordinate,
            self.patch_level,
            (self.patch_size, self.patch_size),
        ).convert("RGB")
        return {"img": self.transform(image), "coord": coordinate}


def _resolve_patch_value(
    name: str, override: int | None, stored_value
) -> int:
    value = override if override is not None else stored_value
    if value is None:
        raise ValueError(
            f"{name} is absent from the coordinate bag; provide it explicitly"
        )
    value = int(value)
    if value < 0 or (name == "patch_size" and value == 0):
        raise ValueError(f"{name} must be positive" if name == "patch_size" else f"{name} must be non-negative")
    return value
