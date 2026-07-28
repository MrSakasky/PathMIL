"""Serialization helpers."""

from __future__ import annotations

import pickle

import h5py


def save_pkl(filename, save_object) -> None:
    with open(filename, "wb") as handle:
        pickle.dump(save_object, handle)


def load_pkl(filename):
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def save_hdf5(
    output_path,
    asset_dict,
    attr_dict=None,
    mode="a",
    chunk_size=32,
):
    """Create or append aligned arrays in an HDF5 file."""

    with h5py.File(output_path, mode) as handle:
        for key, values in asset_dict.items():
            if values.ndim == 0:
                raise ValueError(f"HDF5 asset '{key}' must have a batch dimension")
            if key not in handle:
                chunk_length = max(1, min(chunk_size, max(1, values.shape[0])))
                dataset = handle.create_dataset(
                    key,
                    shape=values.shape,
                    maxshape=(None, *values.shape[1:]),
                    chunks=(chunk_length, *values.shape[1:]),
                    dtype=values.dtype,
                )
                dataset[:] = values
                for attr_key, attr_value in (attr_dict or {}).get(key, {}).items():
                    dataset.attrs[attr_key] = attr_value
            elif values.shape[0] > 0:
                dataset = handle[key]
                original_length = len(dataset)
                dataset.resize(original_length + values.shape[0], axis=0)
                dataset[original_length:] = values
    return output_path
