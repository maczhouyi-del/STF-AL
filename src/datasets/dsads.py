from __future__ import annotations

from pathlib import Path

import numpy as np

from .data_loader import (
    dataset_error,
    list_files,
    parse_id_from_path,
    read_table_file,
    require_raw_dir,
    resolve_sensor_columns,
    validate_continuous_arrays,
)
from .transforms import impute_missing_values


def load_dsads_continuous(config):
    raw_dir = require_raw_dir(config, "DSADS")
    patterns = config["dataset"].get("file_patterns", ["*.txt", "*.dat", "*.csv"])
    files = list_files(raw_dir, patterns)
    if not files:
        raise dataset_error("DSADS", raw_dir, str(config["dataset"].get("config_hint", "configs/datasets/dsads.yaml")))

    subject_patterns = config["dataset"].get("subject_patterns", [r"p(\d+)", r"subject[_-]?(\d+)", r"subj[_-]?(\d+)"])
    activity_patterns = config["dataset"].get("activity_patterns", [r"a(\d+)", r"activity[_-]?(\d+)", r"act[_-]?(\d+)"])
    trial_patterns = config["dataset"].get("trial_patterns", [r"s(\d+)", r"trial[_-]?(\d+)", r"segment[_-]?(\d+)"])
    has_header = bool(config["dataset"].get("has_header", False))
    missing_strategy = config["dataset"].get("missing_value_strategy", "interpolate")

    xs, ys, subjects, sessions = [], [], [], []
    for file_index, path in enumerate(files):
        table, columns = read_table_file(path, has_header=has_header)
        if table.ndim != 2:
            raise ValueError(f"DSADS file {path} did not parse to a 2D table.")
        subject_id = parse_id_from_path(path, subject_patterns, "subject_id")
        activity_id = parse_id_from_path(path, activity_patterns, "activity_id")
        try:
            trial_id = parse_id_from_path(path, trial_patterns, "trial_id")
        except ValueError:
            trial_id = file_index
        sensor_cols = resolve_sensor_columns(config, columns, table.shape[1], reserved=[])
        x = impute_missing_values(table[:, sensor_cols].astype(np.float32), strategy=missing_strategy)
        y = np.full(len(x), activity_id, dtype=np.int64)
        xs.append(x)
        ys.append(y)
        subjects.append(np.full(len(x), subject_id, dtype=np.int64))
        sessions.append(np.full(len(x), trial_id, dtype=np.int64))

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subject_id = np.concatenate(subjects, axis=0)
    session_id = np.concatenate(sessions, axis=0)
    validate_continuous_arrays(x, y, subject_id, session_id, "DSADS")
    return x, y, subject_id, session_id
