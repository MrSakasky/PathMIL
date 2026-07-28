"""Extract patch embeddings from either coordinate bags or image bags."""

from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_modules.dataset_h5 import (
    CoordinatePatchDataset,
    MaterializedPatchDataset,
    SlideListDataset,
)
from models import get_encoder
from utils.file_utils import save_hdf5


def _import_openslide():
    openslide_path = os.environ.get("OPENSLIDE_PATH")
    dll_context = (
        os.add_dll_directory(openslide_path)
        if openslide_path and hasattr(os, "add_dll_directory")
        else nullcontext()
    )
    with dll_context:
        import openslide

    return openslide


def extract_batches(
    output_path: Path,
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    mode = "w"
    for batch_data in tqdm(loader, desc="Batches", leave=False):
        images = batch_data["img"].to(device, non_blocking=True)
        coordinates = batch_data["coord"].numpy().astype(np.int32)
        with torch.inference_mode():
            features = model(images).detach().cpu().numpy().astype(np.float32)
        save_hdf5(
            str(output_path),
            {"features": features, "coords": coordinates},
            mode=mode,
        )
        mode = "a"


def _slide_id(csv_value: str, slide_extension: str) -> str:
    name = Path(csv_value).name
    return name[: -len(slide_extension)] if name.endswith(slide_extension) else Path(name).stem


def extract_features(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_h5_dir = args.feature_dir / "h5_files"
    output_pt_dir = args.feature_dir / "pt_files"
    output_h5_dir.mkdir(parents=True, exist_ok=True)
    output_pt_dir.mkdir(parents=True, exist_ok=True)

    slide_list = SlideListDataset(args.csv_path)
    model, transform = get_encoder(
        args.model_name,
        target_img_size=args.target_patch_size,
        checkpoint_path=args.encoder_checkpoint,
    )
    model = model.eval().to(device)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    openslide = _import_openslide() if args.pipeline == "coordinates" else None

    for index, csv_value in enumerate(slide_list, start=1):
        slide_id = _slide_id(csv_value, args.slide_ext)
        bag_path = args.data_h5_dir / "patches" / f"{slide_id}.h5"
        output_h5 = output_h5_dir / f"{slide_id}.h5"
        output_pt = output_pt_dir / f"{slide_id}.pt"
        print(f"\n[{index}/{len(slide_list)}] Processing {slide_id}")

        if args.auto_skip and output_pt.is_file():
            print("Skipping: feature tensor already exists")
            continue
        if not bag_path.is_file():
            raise FileNotFoundError(f"Patch bag does not exist: {bag_path}")

        start = time.perf_counter()
        wsi = None
        try:
            if args.pipeline == "coordinates":
                slide_path = args.data_slide_dir / f"{slide_id}{args.slide_ext}"
                if not slide_path.is_file():
                    raise FileNotFoundError(f"Slide does not exist: {slide_path}")
                wsi = openslide.open_slide(str(slide_path))
                dataset = CoordinatePatchDataset(
                    bag_path,
                    wsi,
                    transform,
                    patch_level=args.patch_level,
                    patch_size=args.read_patch_size,
                )
            else:
                dataset = MaterializedPatchDataset(bag_path, transform)

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                **loader_kwargs,
            )
            extract_batches(output_h5, loader, model, device)
        finally:
            if wsi is not None:
                wsi.close()

        with h5py.File(output_h5, "r") as handle:
            features = handle["features"][:]
            coordinate_shape = handle["coords"].shape
        torch.save(torch.from_numpy(features), output_pt)
        elapsed = time.perf_counter() - start
        print(
            f"Saved {features.shape} features and {coordinate_shape} coordinates "
            f"in {elapsed:.2f} seconds"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--data-h5-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--pipeline",
        choices=("coordinates", "images"),
        default="coordinates",
    )
    parser.add_argument(
        "--data-slide-dir",
        type=Path,
        help="Required for the coordinate pipeline.",
    )
    parser.add_argument("--slide-ext", default=".svs")
    parser.add_argument("--model-name", default="resnet50_trunc")
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-patch-size", type=int, default=224)
    parser.add_argument(
        "--patch-level",
        type=int,
        help="Override the level stored in coordinate bags.",
    )
    parser.add_argument(
        "--read-patch-size",
        type=int,
        help="Override the patch size stored in coordinate bags.",
    )
    parser.add_argument("--no-auto-skip", dest="auto_skip", action="store_false")
    parser.set_defaults(auto_skip=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pipeline == "coordinates" and args.data_slide_dir is None:
        raise ValueError("--data-slide-dir is required for the coordinate pipeline")
    extract_features(args)


if __name__ == "__main__":
    main()
