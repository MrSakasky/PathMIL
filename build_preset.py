"""Create a reusable CSV preset for slide segmentation and patching."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("presets"))
    parser.add_argument("--seg-level", type=int, default=-1)
    parser.add_argument("--sthresh", type=int, default=8)
    parser.add_argument("--mthresh", type=int, default=7)
    parser.add_argument("--use-otsu", action="store_true")
    parser.add_argument("--close", type=int, default=4)
    parser.add_argument("--a-t", type=int, default=100)
    parser.add_argument("--a-h", type=int, default=16)
    parser.add_argument("--max-n-holes", type=int, default=8)
    parser.add_argument("--vis-level", type=int, default=-1)
    parser.add_argument("--line-thickness", type=int, default=250)
    parser.add_argument("--white-thresh", type=int, default=5)
    parser.add_argument("--black-thresh", type=int, default=50)
    parser.add_argument("--no-padding", dest="use_padding", action="store_false")
    parser.add_argument(
        "--contour-fn",
        choices=("four_pt", "center", "basic", "four_pt_hard"),
        default="four_pt",
    )
    parser.set_defaults(use_padding=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    values = {
        "seg_level": args.seg_level,
        "sthresh": args.sthresh,
        "mthresh": args.mthresh,
        "close": args.close,
        "use_otsu": args.use_otsu,
        "keep_ids": "none",
        "exclude_ids": "none",
        "a_t": args.a_t,
        "a_h": args.a_h,
        "max_n_holes": args.max_n_holes,
        "vis_level": args.vis_level,
        "line_thickness": args.line_thickness,
        "white_thresh": args.white_thresh,
        "black_thresh": args.black_thresh,
        "use_padding": args.use_padding,
        "contour_fn": args.contour_fn,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        args.preset_name
        if args.preset_name.endswith(".csv")
        else f"{args.preset_name}.csv"
    )
    output_path = args.output_dir / filename
    pd.DataFrame([values]).to_csv(output_path, index=False)
    print(f"Saved preset to {output_path}")


if __name__ == "__main__":
    main()
