from __future__ import annotations

import numpy as np

from .data_loader import (
    dataset_error,
    list_files,
    load_np_files_dataset,
    load_table_dataset,
    require_raw_dir,
)


def load_tsa_continuous(config):
    """Load the self-collected TSA dataset as a subject-wise HAR dataset."""
    raw_dir = require_raw_dir(config, "TSA")

    np_files = (
        list_files(raw_dir, config["dataset"].get("npz_patterns", ["*.npz"]))
        + list_files(raw_dir, config["dataset"].get("npy_patterns", ["*.npy"]))
        + list_files(raw_dir, config["dataset"].get("mat_patterns", ["*.mat"]))
    )
    table_files = list_files(raw_dir, config["dataset"].get("file_patterns", ["*.csv", "*.txt", "*.dat"]))
    if np_files:
        return load_np_files_dataset(config, "TSA")
    if table_files:
        return load_table_dataset(
            config,
            dataset_name="TSA",
            default_patterns=["*.csv", "*.txt", "*.dat"],
            default_subject_patterns=[r"subject[_-]?(\d+)", r"subj[_-]?(\d+)", r"S(\d+)"],
        )
    raise dataset_error("TSA", raw_dir, str(config["dataset"].get("config_hint", "configs/datasets/tsa.yaml")))
