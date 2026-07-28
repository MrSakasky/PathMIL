"""Slide-level datasets and split management."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth


def save_splits(
    split_datasets,
    column_keys: Sequence[str],
    filename: str | Path,
    boolean_style: bool = False,
) -> None:
    slide_columns = [
        split.slide_data["slide_id"] if split is not None else pd.Series(dtype=str)
        for split in split_datasets
    ]
    if boolean_style:
        slide_ids = pd.concat(slide_columns, ignore_index=True)
        membership = np.repeat(
            np.eye(len(slide_columns), dtype=bool),
            [len(column) for column in slide_columns],
            axis=0,
        )
        frame = pd.DataFrame(membership, index=slide_ids, columns=column_keys)
    else:
        frame = pd.concat(slide_columns, ignore_index=True, axis=1)
        frame.columns = column_keys
    frame.to_csv(filename)


class Generic_WSI_Classification_Dataset(Dataset):
    """Dataset metadata shared by split creation, training and evaluation."""

    def __init__(
        self,
        csv_path: str | Path,
        shuffle: bool = False,
        seed: int = 7,
        print_info: bool = True,
        label_dict: dict | None = None,
        filter_dict: dict | None = None,
        ignore: Iterable | None = None,
        patient_strat: bool = False,
        label_col: str | None = None,
        patient_voting: str = "max",
    ):
        self.label_dict = dict(label_dict or {})
        if not self.label_dict:
            raise ValueError("label_dict must define at least one class")
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.label_col = label_col or "label"
        self.train_ids = self.val_ids = self.test_ids = None
        self.data_dir = None

        slide_data = pd.read_csv(
            csv_path,
            encoding="utf-8",
            dtype={"slide_id": str, "case_id": str},
        )
        self._validate_columns(slide_data)
        slide_data = self.filter_df(slide_data, filter_dict or {})
        slide_data = self.df_prep(
            slide_data,
            self.label_dict,
            list(ignore or []),
            self.label_col,
        )
        if shuffle:
            slide_data = slide_data.sample(frac=1, random_state=seed)
        self.slide_data = slide_data.reset_index(drop=True)

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()
        if print_info:
            self.summarize()

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        required = {"slide_id", "case_id", self.label_col}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Dataset CSV is missing columns: {sorted(missing)}")

    def cls_ids_prep(self) -> None:
        self.patient_cls_ids = [
            np.flatnonzero(self.patient_data["label"] == class_id)
            for class_id in range(self.num_classes)
        ]
        slide_labels = self.slide_data["label"].to_numpy(dtype=int)
        self.slide_cls_ids = [
            np.flatnonzero(slide_labels == class_id)
            for class_id in range(self.num_classes)
        ]

    def patient_data_prep(self, patient_voting: str = "max") -> None:
        case_ids = []
        labels = []
        for case_id, group in self.slide_data.groupby("case_id", sort=False):
            values = group["label"].to_numpy(dtype=int)
            if patient_voting == "max":
                patient_label = int(values.max())
            elif patient_voting == "maj":
                patient_label = int(np.bincount(values).argmax())
            else:
                raise ValueError("patient_voting must be 'max' or 'maj'")
            case_ids.append(case_id)
            labels.append(patient_label)
        self.patient_data = {
            "case_id": np.asarray(case_ids),
            "label": np.asarray(labels, dtype=int),
        }

    @staticmethod
    def df_prep(
        data: pd.DataFrame,
        label_dict: dict,
        ignore: Iterable,
        label_col: str,
    ) -> pd.DataFrame:
        prepared = data.copy()
        prepared["label"] = prepared[label_col]
        prepared = prepared[~prepared["label"].isin(ignore)].copy()
        unknown = set(prepared["label"]).difference(label_dict)
        if unknown:
            raise ValueError(f"Labels are missing from label_dict: {sorted(unknown)}")
        prepared["label"] = prepared["label"].map(label_dict).astype(int)
        return prepared.reset_index(drop=True)

    @staticmethod
    def filter_df(frame: pd.DataFrame, filter_dict: dict | None = None):
        filtered = frame
        for column, accepted_values in (filter_dict or {}).items():
            if column not in filtered:
                raise ValueError(f"Unknown filter column: {column}")
            filtered = filtered[filtered[column].isin(accepted_values)]
        return filtered.copy()

    def __len__(self) -> int:
        key = "case_id" if self.patient_strat else None
        return len(self.patient_data[key]) if key else len(self.slide_data)

    def __getitem__(self, index):
        raise NotImplementedError

    def summarize(self) -> None:
        print(f"Label column: {self.label_col}")
        print(f"Label mapping: {self.label_dict}")
        print(f"Number of classes: {self.num_classes}")
        print("Slide-level counts:")
        print(self.slide_data["label"].value_counts(sort=False))
        for class_id in range(self.num_classes):
            print(
                f"Class {class_id}: {len(self.patient_cls_ids[class_id])} patients, "
                f"{len(self.slide_cls_ids[class_id])} slides"
            )

    def create_splits(
        self,
        k: int = 3,
        val_num=(25, 25),
        test_num=(40, 40),
        label_frac: float = 1.0,
        custom_test_ids=None,
    ) -> None:
        class_ids = self.patient_cls_ids if self.patient_strat else self.slide_cls_ids
        sample_count = (
            len(self.patient_data["case_id"])
            if self.patient_strat
            else len(self.slide_data)
        )
        self.split_gen = generate_split(
            cls_ids=class_ids,
            val_num=val_num,
            test_num=test_num,
            samples=sample_count,
            n_splits=k,
            label_frac=label_frac,
            seed=self.seed,
            custom_test_ids=custom_test_ids,
        )

    def set_splits(self, start_from: int | None = None) -> None:
        ids = nth(self.split_gen, start_from) if start_from is not None else next(self.split_gen)
        if ids is None:
            raise IndexError("Requested split does not exist")
        if self.patient_strat:
            slide_ids = []
            for split_ids in ids:
                selected = []
                for patient_index in split_ids:
                    case_id = self.patient_data["case_id"][patient_index]
                    selected.extend(
                        np.flatnonzero(
                            self.slide_data["case_id"].to_numpy() == case_id
                        ).tolist()
                    )
                slide_ids.append(selected)
            ids = slide_ids
        self.train_ids, self.val_ids, self.test_ids = (list(values) for values in ids)

    def _split_from_slide_ids(self, slide_ids: Iterable[str]):
        selected = list(slide_ids)
        if not selected:
            return None
        frame = self.slide_data[
            self.slide_data["slide_id"].isin(selected)
        ].reset_index(drop=True)
        return Generic_Split(frame, self.data_dir, self.num_classes)

    def get_split_from_df(self, all_splits, split_key: str = "train"):
        return self._split_from_slide_ids(all_splits[split_key].dropna().tolist())

    def get_merged_split_from_df(self, all_splits, split_keys=("train",)):
        merged = []
        for split_key in split_keys:
            merged.extend(all_splits[split_key].dropna().tolist())
        return self._split_from_slide_ids(merged)

    def return_splits(
        self,
        from_id: bool = True,
        csv_path: str | Path | None = None,
    ):
        if from_id:
            if any(ids is None for ids in (self.train_ids, self.val_ids, self.test_ids)):
                raise RuntimeError("Call set_splits before return_splits(from_id=True)")
            return tuple(
                Generic_Split(
                    self.slide_data.iloc[ids].reset_index(drop=True),
                    self.data_dir,
                    self.num_classes,
                )
                if ids
                else None
                for ids in (self.train_ids, self.val_ids, self.test_ids)
            )
        if csv_path is None:
            raise ValueError("csv_path is required when from_id=False")
        all_splits = pd.read_csv(csv_path, dtype={"train": str, "val": str, "test": str})
        return tuple(
            self.get_split_from_df(all_splits, key)
            for key in ("train", "val", "test")
        )

    def get_list(self, ids):
        return self.slide_data.iloc[ids]["slide_id"]

    def getlabel(self, ids):
        return self.slide_data.iloc[ids]["label"]

    def test_split_gen(self, return_descriptor: bool = False):
        ids_by_name = {
            "train": self.train_ids,
            "val": self.val_ids,
            "test": self.test_ids,
        }
        if any(ids is None for ids in ids_by_name.values()):
            raise RuntimeError("No active split")
        class_names = {
            class_id: name for name, class_id in self.label_dict.items()
        }
        descriptor = pd.DataFrame(
            0,
            index=[class_names[index] for index in range(self.num_classes)],
            columns=ids_by_name,
            dtype=int,
        )
        for split_name, ids in ids_by_name.items():
            counts = self.getlabel(ids).value_counts()
            print(f"{split_name}: {len(ids)} samples")
            for class_id, count in counts.items():
                descriptor.loc[class_names[int(class_id)], split_name] = int(count)

        split_sets = [set(ids) for ids in ids_by_name.values()]
        if any(
            left.intersection(right)
            for index, left in enumerate(split_sets)
            for right in split_sets[index + 1 :]
        ):
            raise RuntimeError("Generated splits overlap")
        return descriptor if return_descriptor else None

    def save_split(self, filename: str | Path) -> None:
        frame = pd.concat(
            [
                pd.Series(self.get_list(ids).tolist(), name=name)
                for name, ids in (
                    ("train", self.train_ids),
                    ("val", self.val_ids),
                    ("test", self.test_ids),
                )
            ],
            axis=1,
        )
        frame.to_csv(filename, index=False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False

    def load_from_h5(self, toggle: bool) -> None:
        self.use_h5 = toggle

    def __getitem__(self, index: int):
        row = self.slide_data.iloc[index]
        slide_id = row["slide_id"]
        label = int(row["label"])
        data_dir = (
            self.data_dir[row["source"]]
            if isinstance(self.data_dir, dict)
            else self.data_dir
        )
        if not data_dir:
            return slide_id, label

        if not self.use_h5:
            features = torch.load(
                Path(data_dir) / "pt_files" / f"{slide_id}.pt",
                map_location="cpu",
            )
            return features, label

        with h5py.File(
            Path(data_dir) / "h5_files" / f"{slide_id}.h5",
            "r",
        ) as handle:
            features = torch.from_numpy(handle["features"][:])
            coordinates = handle["coords"][:]
        return features, label, coordinates


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, data_dir=None, num_classes: int = 2):
        self.use_h5 = True
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        labels = self.slide_data["label"].to_numpy(dtype=int)
        self.slide_cls_ids = [
            np.flatnonzero(labels == class_id)
            for class_id in range(self.num_classes)
        ]

    def __len__(self) -> int:
        return len(self.slide_data)
