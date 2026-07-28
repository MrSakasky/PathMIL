"""Build and update per-slide preprocessing manifests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def initialize_df(
    slides,
    seg_params,
    filter_params,
    vis_params,
    patch_params,
    use_heatmap_args=False,
    save_patches=False,
):
    """Fill missing manifest columns with the supplied preprocessing defaults."""

    if isinstance(slides, pd.DataFrame):
        frame = slides.copy()
        if "slide_id" not in frame:
            raise ValueError("Process list must contain a 'slide_id' column")
        slide_ids = frame["slide_id"].to_numpy()
    else:
        slide_ids = np.asarray(list(slides))
        frame = pd.DataFrame({"slide_id": slide_ids})

    total = len(frame)
    defaults = {
        "slide_id": slide_ids,
        "process": np.full(total, 1, dtype=np.uint8),
        "status": np.full(total, "tbp", dtype=object),
        "seg_level": np.full(total, int(seg_params["seg_level"]), dtype=np.int16),
        "sthresh": np.full(total, int(seg_params["sthresh"]), dtype=np.uint8),
        "mthresh": np.full(total, int(seg_params["mthresh"]), dtype=np.uint8),
        "close": np.full(total, int(seg_params["close"]), dtype=np.uint32),
        "use_otsu": np.full(total, bool(seg_params["use_otsu"]), dtype=bool),
        "keep_ids": np.full(total, seg_params["keep_ids"], dtype=object),
        "exclude_ids": np.full(total, seg_params["exclude_ids"], dtype=object),
        "a_t": np.full(total, float(filter_params["a_t"]), dtype=np.float32),
        "a_h": np.full(total, float(filter_params["a_h"]), dtype=np.float32),
        "max_n_holes": np.full(
            total, int(filter_params["max_n_holes"]), dtype=np.uint32
        ),
        "vis_level": np.full(total, int(vis_params["vis_level"]), dtype=np.int16),
        "line_thickness": np.full(
            total, int(vis_params["line_thickness"]), dtype=np.uint32
        ),
        "use_padding": np.full(
            total, bool(patch_params["use_padding"]), dtype=bool
        ),
        "contour_fn": np.full(total, patch_params["contour_fn"], dtype=object),
    }
    if save_patches:
        defaults.update(
            {
                "white_thresh": np.full(
                    total, int(patch_params["white_thresh"]), dtype=np.uint8
                ),
                "black_thresh": np.full(
                    total, int(patch_params["black_thresh"]), dtype=np.uint8
                ),
            }
        )
    if use_heatmap_args:
        defaults.update(
            {
                "label": np.full(total, -1, dtype=int),
                "x1": np.full(total, np.nan),
                "x2": np.full(total, np.nan),
                "y1": np.full(total, np.nan),
                "y2": np.full(total, np.nan),
            }
        )

    default_frame = pd.DataFrame(defaults)
    for column, values in defaults.items():
        if column in frame:
            frame[column] = frame[column].fillna(default_frame[column])
        else:
            frame[column] = values
    return frame
