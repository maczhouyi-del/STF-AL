from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.base import load_continuous_arrays, remap_labels
from src.datasets.splits import subject_partitions, validate_no_subject_leakage
from src.datasets.windowing import create_sliding_windows
from src.utils.config import load_config


def build_parser(dataset_name: str):
    parser = argparse.ArgumentParser(description=f"Inspect {dataset_name} data format and fold definitions.")
    parser.add_argument("--config", required=True)
    return parser


def prepare_dataset(config_path: str, expected_name: str) -> None:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    dataset_name = str(dataset_cfg["name"]).lower()
    if dataset_name != expected_name:
        raise ValueError(f"Expected dataset.name={expected_name}, got {dataset_name} in {config_path}")

    raw_dir = Path(dataset_cfg["raw_dir"])
    print(f"dataset: {dataset_name}")
    print(f"raw_dir: {raw_dir}")

    x_raw, y_raw, subject_raw, session_raw = load_continuous_arrays(config)
    raw_files = []
    if raw_dir.exists():
        raw_files = [path for path in raw_dir.rglob("*") if path.is_file()]
    print(f"raw_file_count: {len(raw_files)}")
    print(f"raw_shape: {tuple(x_raw.shape)}")
    print(f"subject_count: {len(np.unique(subject_raw))}")
    print(f"activity_class_count_raw: {len(np.unique(y_raw))}")

    invalid_labels = set(int(v) for v in dataset_cfg.get("invalid_labels", []))
    x_win, y_win, subject_win, window_meta = create_sliding_windows(
        x=x_raw,
        y=y_raw,
        subject_ids=subject_raw,
        group_ids=session_raw,
        window_length=int(dataset_cfg["window_length"]),
        overlap=float(dataset_cfg["overlap"]),
        invalid_labels=invalid_labels,
        max_invalid_ratio=float(dataset_cfg.get("max_invalid_ratio", 1.0)),
    )
    y_win, label_mapping = remap_labels(y_win)
    print(f"windowed_shape: {tuple(x_win.shape)}")
    print(f"subject_window_counts: {window_meta['subject_window_counts']}")
    print(f"class_window_counts: {window_meta['class_window_counts']}")

    partitions = dataset_cfg.get("subject_partitions")
    if partitions is None:
        partitions = subject_partitions(
            subject_win,
            dataset_name=dataset_name,
            num_partitions=int(dataset_cfg.get("num_partitions", 4)),
        )
    else:
        partitions = [[int(subject) for subject in partition] for partition in partitions]

    for fold_idx, target_subjects in enumerate(partitions):
        source_subjects = [
            int(subject)
            for idx, partition in enumerate(partitions)
            if idx != fold_idx
            for subject in partition
        ]
        validate_no_subject_leakage(source_subjects, target_subjects)
        print(f"fold_{fold_idx}: source_subjects={source_subjects}, target_subjects={target_subjects}, leakage=false")

    print(f"label_mapping: {label_mapping}")
    print("saved_files: none")
    print("inspection_status: ok")
