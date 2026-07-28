"""Segment whole-slide images and create patch-coordinate or patch-image bags."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from wsi_core.batch_process_utils import initialize_df

if TYPE_CHECKING:
    from wsi_core.WholeSlideImage import WholeSlideImage


DEFAULT_SEGMENTATION = {
    "seg_level": -1,
    "sthresh": 8,
    "mthresh": 7,
    "close": 4,
    "use_otsu": False,
    "keep_ids": "none",
    "exclude_ids": "none",
}
DEFAULT_FILTER = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
DEFAULT_VISUALIZATION = {"vis_level": -1, "line_thickness": 250}
DEFAULT_PATCHING = {"use_padding": True, "contour_fn": "four_pt"}
IMAGE_BAG_FILTER = {"white_thresh": 5, "black_thresh": 40}


def _timed(callable_, *args, **kwargs):
    start = time.perf_counter()
    result = callable_(*args, **kwargs)
    return result, time.perf_counter() - start


def _parse_ids(value: Any) -> list[int]:
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    return [int(item) for item in text.split(",")]


def _resolve_level(wsi: WholeSlideImage, requested_level: int) -> int:
    if requested_level >= 0 or len(wsi.level_dim) == 1:
        return max(requested_level, 0)
    return wsi.getOpenSlide().get_best_level_for_downsample(64)


def _parameters_for_slide(
    frame: pd.DataFrame,
    index: int,
    wsi: WholeSlideImage,
    defaults: dict[str, dict[str, Any]],
    use_defaults: bool,
) -> dict[str, dict[str, Any]]:
    parameters = {
        group: values.copy() if use_defaults else {
            key: frame.loc[index, key] for key in values
        }
        for group, values in defaults.items()
    }
    parameters["vis"]["vis_level"] = _resolve_level(
        wsi, int(parameters["vis"]["vis_level"])
    )
    parameters["seg"]["seg_level"] = _resolve_level(
        wsi, int(parameters["seg"]["seg_level"])
    )
    parameters["seg"]["keep_ids"] = _parse_ids(parameters["seg"]["keep_ids"])
    parameters["seg"]["exclude_ids"] = _parse_ids(parameters["seg"]["exclude_ids"])
    return parameters


def _load_process_frame(
    source: Path,
    process_list: Path | None,
    defaults: dict[str, dict[str, Any]],
    pipeline: str,
) -> pd.DataFrame:
    slides = sorted(path.name for path in source.iterdir() if path.is_file())
    source_data: list[str] | pd.DataFrame
    source_data = pd.read_csv(process_list) if process_list else slides
    patch_defaults = defaults["patch"].copy()
    if pipeline == "images":
        patch_defaults.update(IMAGE_BAG_FILTER)
    return initialize_df(
        source_data,
        defaults["seg"],
        defaults["filter"],
        defaults["vis"],
        patch_defaults,
        save_patches=pipeline == "images",
    )


def _create_patch_bag(
    wsi: WholeSlideImage,
    pipeline: str,
    patch_params: dict[str, Any],
    patch_level: int,
    patch_size: int,
    step_size: int,
    patch_dir: Path,
    custom_downsample: int,
) -> tuple[str, float]:
    common = {
        **patch_params,
        "patch_level": patch_level,
        "patch_size": patch_size,
        "step_size": step_size,
        "save_path": str(patch_dir),
    }
    if pipeline == "coordinates":
        return _timed(wsi.process_contours, **common)
    return _timed(
        wsi.createPatches_bag_hdf5,
        **common,
        save_coord=True,
        custom_downsample=custom_downsample,
    )


def _create_stitch(
    bag_path: Path,
    wsi: WholeSlideImage,
    pipeline: str,
    save_patch_preview: bool,
):
    from wsi_core.wsi_utils import StitchCoords, StitchPatches

    if pipeline == "coordinates":
        return _timed(
            StitchCoords,
            str(bag_path),
            wsi,
            downscale=64,
            bg_color=(0, 0, 0),
            alpha=-1,
            draw_grid=False,
            save_patch=save_patch_preview,
        )
    return _timed(
        StitchPatches,
        str(bag_path),
        downscale=64,
        bg_color=(0, 0, 0),
        alpha=-1,
        draw_grid=False,
    )


def process_slides(
    source: Path,
    save_dir: Path,
    defaults: dict[str, dict[str, Any]],
    *,
    pipeline: str = "coordinates",
    patch_size: int = 224,
    step_size: int = 224,
    patch_level: int = 0,
    custom_downsample: int = 1,
    segment: bool = True,
    patch: bool = True,
    stitch: bool = False,
    save_mask: bool = True,
    save_patch_preview: bool = False,
    auto_skip: bool = True,
    process_list: Path | None = None,
    use_default_params: bool = False,
) -> dict[str, float]:
    from wsi_core.WholeSlideImage import WholeSlideImage

    patch_dir = save_dir / "patches"
    mask_dir = save_dir / "masks"
    stitch_dir = save_dir / "stitches"
    for directory in (save_dir, patch_dir, mask_dir, stitch_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame = _load_process_frame(source, process_list, defaults, pipeline)
    pending_indices = frame.index[frame["process"] == 1].tolist()
    timings = {"segmentation": [], "patching": [], "stitching": []}
    process_log = save_dir / "process_list_autogen.csv"

    for position, index in enumerate(tqdm(pending_indices), start=1):
        frame.to_csv(process_log, index=False)
        slide_name = str(frame.loc[index, "slide_id"])
        slide_id = Path(slide_name).stem
        bag_path = patch_dir / f"{slide_id}.h5"
        print(f"\n[{position}/{len(pending_indices)}] Processing {slide_name}")
        frame.loc[index, "process"] = 0

        if auto_skip and bag_path.is_file():
            frame.loc[index, "status"] = "already_exists"
            print(f"Skipping {slide_id}: output already exists")
            continue

        wsi = WholeSlideImage(str(source / slide_name))
        current = _parameters_for_slide(
            frame, index, wsi, defaults, use_default_params
        )
        segmentation_level = current["seg"]["seg_level"]
        width, height = wsi.level_dim[segmentation_level]
        if width * height > 1e8:
            frame.loc[index, "status"] = "failed_segmentation"
            print(f"Skipping {slide_id}: segmentation level is too large")
            continue

        frame.loc[index, "vis_level"] = current["vis"]["vis_level"]
        frame.loc[index, "seg_level"] = segmentation_level

        if segment:
            _, elapsed = _timed(
                wsi.segmentTissue,
                **current["seg"],
                filter_params=current["filter"],
            )
            timings["segmentation"].append(elapsed)

        if save_mask:
            mask = wsi.visWSI(**current["vis"])
            mask.save(mask_dir / f"{slide_id}.png")

        if patch:
            _, elapsed = _create_patch_bag(
                wsi,
                pipeline,
                current["patch"],
                patch_level,
                patch_size,
                step_size,
                patch_dir,
                custom_downsample,
            )
            timings["patching"].append(elapsed)

        if stitch and bag_path.is_file():
            preview, elapsed = _create_stitch(
                bag_path, wsi, pipeline, save_patch_preview
            )
            preview.save(stitch_dir / f"{slide_id}.png")
            timings["stitching"].append(elapsed)

        frame.loc[index, "status"] = "processed"

    frame.to_csv(process_log, index=False)
    averages = {
        name: float(np.mean(values)) if values else 0.0
        for name, values in timings.items()
    }
    for name, elapsed in averages.items():
        print(f"Average {name} time: {elapsed:.2f} seconds per slide")
    return averages


def _add_boolean_flag(
    parser: argparse.ArgumentParser, name: str, default: bool, help_text: str
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action="store_true",
        help=help_text,
    )
    group.add_argument(
        f"--no-{name}",
        dest=name.replace("-", "_"),
        action="store_false",
    )
    parser.set_defaults(**{name.replace("-", "_"): default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment slides and create HDF5 patch bags."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--pipeline",
        choices=("coordinates", "images"),
        default="coordinates",
        help="Store coordinates only (recommended) or materialized patch images.",
    )
    parser.add_argument("--preset", type=Path)
    parser.add_argument("--process-list", type=Path)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--step-size", type=int, default=224)
    parser.add_argument("--patch-level", type=int, default=0)
    parser.add_argument("--custom-downsample", type=int, choices=(1, 2), default=1)
    parser.add_argument("--save-patch-preview", action="store_true")
    _add_boolean_flag(parser, "segment", True, "Run tissue segmentation.")
    _add_boolean_flag(parser, "patch", True, "Create patch bags.")
    _add_boolean_flag(parser, "stitch", False, "Create stitched previews.")
    _add_boolean_flag(parser, "save-mask", True, "Save segmentation masks.")
    _add_boolean_flag(parser, "auto-skip", True, "Skip existing patch bags.")
    return parser


def _load_defaults(preset: Path | None, pipeline: str) -> dict[str, dict[str, Any]]:
    defaults = {
        "seg": DEFAULT_SEGMENTATION.copy(),
        "filter": DEFAULT_FILTER.copy(),
        "vis": DEFAULT_VISUALIZATION.copy(),
        "patch": DEFAULT_PATCHING.copy(),
    }
    if pipeline == "images":
        defaults["patch"].update(IMAGE_BAG_FILTER)
    if preset is None:
        return defaults

    preset_path = preset if preset.is_file() else Path("presets") / preset
    row = pd.read_csv(preset_path).iloc[0]
    for values in defaults.values():
        for key in values:
            if key in row:
                values[key] = row[key]
    return defaults


def main() -> None:
    args = build_parser().parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"Slide directory does not exist: {args.source}")
    process_list = args.process_list
    if process_list and not process_list.is_file():
        candidate = args.save_dir / process_list
        process_list = candidate if candidate.is_file() else process_list
    process_slides(
        args.source,
        args.save_dir,
        _load_defaults(args.preset, args.pipeline),
        pipeline=args.pipeline,
        patch_size=args.patch_size,
        step_size=args.step_size,
        patch_level=args.patch_level,
        custom_downsample=args.custom_downsample,
        segment=args.segment,
        patch=args.patch,
        stitch=args.stitch,
        save_mask=args.save_mask,
        save_patch_preview=args.save_patch_preview,
        auto_skip=args.auto_skip,
        process_list=process_list,
    )


if __name__ == "__main__":
    main()
