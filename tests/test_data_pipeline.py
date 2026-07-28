import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset
from dataset_modules.dataset_h5 import (
    CoordinatePatchDataset,
    MaterializedPatchDataset,
    SlideListDataset,
)
from wsi_core.batch_process_utils import initialize_df


class _FakeSlide:
    level_count = 2

    def read_region(self, coordinate, level, size):
        value = int(coordinate[0]) + level
        return Image.fromarray(np.full((*size, 3), value, dtype=np.uint8))


class DataPipelineTests(unittest.TestCase):
    def test_manifest_defaults_fill_missing_values(self):
        frame = pd.DataFrame({"slide_id": ["a.svs"], "x1": [np.nan]})
        result = initialize_df(
            frame,
            {
                "seg_level": -1,
                "sthresh": 8,
                "mthresh": 7,
                "close": 4,
                "use_otsu": False,
                "keep_ids": "none",
                "exclude_ids": "none",
            },
            {"a_t": 100, "a_h": 16, "max_n_holes": 8},
            {"vis_level": -1, "line_thickness": 250},
            {"use_padding": True, "contour_fn": "four_pt"},
            use_heatmap_args=True,
        )

        self.assertEqual(result.loc[0, "process"], 1)
        self.assertTrue(np.isnan(result.loc[0, "x1"]))
        self.assertEqual(result.loc[0, "status"], "tbp")

    def test_both_patch_bag_datasets_share_the_same_sample_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            bag_path = Path(directory) / "bag.h5"
            with h5py.File(bag_path, "w") as handle:
                images = handle.create_dataset(
                    "imgs",
                    data=np.zeros((2, 4, 4, 3), dtype=np.uint8),
                )
                images.attrs["patch_size"] = 4
                coordinates = handle.create_dataset(
                    "coords",
                    data=np.asarray([[1, 2], [3, 4]], dtype=np.int32),
                )
                coordinates.attrs["patch_level"] = 0
                coordinates.attrs["patch_size"] = 4

            transform = lambda image: np.asarray(image)
            image_dataset = MaterializedPatchDataset(bag_path, transform)
            coordinate_dataset = CoordinatePatchDataset(
                bag_path,
                _FakeSlide(),
                transform,
                patch_level=1,
            )

            self.assertEqual(image_dataset[0]["coord"].tolist(), [1, 2])
            self.assertEqual(coordinate_dataset[0]["img"].shape, (4, 4, 3))
            self.assertEqual(coordinate_dataset.patch_level, 1)

    def test_slide_list_and_metadata_split_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "slides.csv"
            pd.DataFrame(
                {
                    "slide_id": ["001", "002", "003", "004"],
                    "case_id": ["p1", "p2", "p3", "p4"],
                    "label": ["normal", "normal", "tumor", "tumor"],
                }
            ).to_csv(csv_path, index=False)

            slide_list = SlideListDataset(csv_path)
            dataset = Generic_WSI_Classification_Dataset(
                csv_path,
                label_dict={"normal": 0, "tumor": 1},
                print_info=False,
            )

            self.assertEqual(slide_list[0], "001")
            self.assertEqual(dataset.slide_cls_ids[0].tolist(), [0, 1])
            self.assertEqual(dataset.slide_cls_ids[1].tolist(), [2, 3])

            dataset.create_splits(
                k=1,
                val_num=np.asarray([1, 1]),
                test_num=np.asarray([0, 0]),
                label_frac=1.0,
            )
            dataset.set_splits()
            train, validation, test = dataset.return_splits(from_id=True)
            self.assertEqual((len(train), len(validation), test), (2, 2, None))


if __name__ == "__main__":
    unittest.main()
