"""Small geometry and visualization helper classes."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


class Mosaic_Canvas:
    def __init__(
        self,
        patch_size=256,
        n=100,
        downscale=4,
        n_per_row=10,
        bg_color=(0, 0, 0),
        alpha=-1,
    ):
        self.patch_size = patch_size
        self.downscaled_patch_size = int(np.ceil(patch_size / downscale))
        self.n_rows = int(np.ceil(n / n_per_row))
        self.n_cols = n_per_row
        width = self.n_cols * self.downscaled_patch_size
        height = self.n_rows * self.downscaled_patch_size
        if alpha < 0:
            self.canvas = Image.new("RGB", (width, height), bg_color)
        else:
            self.canvas = Image.new(
                "RGBA",
                (width, height),
                bg_color + (int(255 * alpha),),
            )
        self.dimensions = np.array([width, height])
        self.reset_coord()

    def reset_coord(self):
        self.coord = np.array([0, 0])

    def increment_coord(self):
        if self.coord[0] + self.downscaled_patch_size < self.dimensions[0]:
            self.coord[0] += self.downscaled_patch_size
        else:
            self.coord[0] = 0
            self.coord[1] += self.downscaled_patch_size

    def save(self, save_path, **kwargs):
        self.canvas.save(save_path, **kwargs)

    def paste_patch(self, patch):
        if patch.size != (self.patch_size, self.patch_size):
            raise ValueError(
                f"Expected {(self.patch_size, self.patch_size)}, got {patch.size}"
            )
        size = (self.downscaled_patch_size, self.downscaled_patch_size)
        self.canvas.paste(patch.resize(size), tuple(self.coord))
        self.increment_coord()

    def get_painting(self):
        return self.canvas


class Contour_Checking_fn:
    def __call__(self, point):
        raise NotImplementedError


class isInContourV1(Contour_Checking_fn):
    def __init__(self, contour):
        self.contour = contour

    def __call__(self, point):
        return int(_inside(self.contour, point))


class isInContourV2(Contour_Checking_fn):
    def __init__(self, contour, patch_size):
        self.contour = contour
        self.patch_size = patch_size

    def __call__(self, point):
        center = (
            point[0] + self.patch_size // 2,
            point[1] + self.patch_size // 2,
        )
        return int(_inside(self.contour, center))


class isInContourV3_Easy(Contour_Checking_fn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.contour = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, point):
        return int(any(_inside(self.contour, item) for item in self._points(point)))

    def _points(self, point):
        return _shifted_points(point, self.patch_size, self.shift)


class isInContourV3_Hard(Contour_Checking_fn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.contour = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, point):
        return int(all(_inside(self.contour, item) for item in self._points(point)))

    def _points(self, point):
        return _shifted_points(point, self.patch_size, self.shift)


def _inside(contour, point) -> bool:
    return cv2.pointPolygonTest(contour, tuple(np.asarray(point, dtype=float)), False) >= 0


def _shifted_points(point, patch_size, shift):
    center_x = point[0] + patch_size // 2
    center_y = point[1] + patch_size // 2
    if shift <= 0:
        return [(center_x, center_y)]
    return [
        (center_x - shift, center_y - shift),
        (center_x + shift, center_y + shift),
        (center_x + shift, center_y - shift),
        (center_x - shift, center_y + shift),
    ]
