from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .transforms import impute_missing_values


def dataset_error(dataset_name: str, raw_dir: Path, config_hint: str) -> FileNotFoundError:
    return FileNotFoundError(
        "\n".join(
            [
                f"Missing raw data for dataset: {dataset_name}",
                f"Current raw_dir: {raw_dir}",
                "Place the dataset files under this directory, or edit the corresponding YAML config.",
                f"Config to check: {config_hint}",
            ]
        )
    )


def require_raw_dir(config: Dict, dataset_name: str) -> Path:
    raw_dir = Path(config["dataset"]["raw_dir"])
    config_hint = str(config["dataset"].get("config_hint", f"configs/datasets/{dataset_name.lower()}.yaml"))
    if not raw_dir.exists():
        raise dataset_error(dataset_name, raw_dir, config_hint)
    return raw_dir


def list_files(raw_dir: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(raw_dir.rglob(pattern))
    return sorted({path for path in files if path.is_file()})


def read_table_file(path: Path, has_header: bool = False) -> Tuple[np.ndarray, List[str] | None]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        if has_header:
            df = pd.read_csv(path)
            return df.to_numpy(), [str(col) for col in df.columns]
        return pd.read_csv(path, header=None).to_numpy(), None
    if suffix in {".dat", ".txt", ".data"}:
        data = np.genfromtxt(path, delimiter=None, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data, None
    raise ValueError(f"Unsupported table file extension for {path}")


def resolve_column(column, columns: List[str] | None, total_cols: int, name: str) -> int:
    if column is None:
        raise ValueError(f"{name} must be configured when it cannot be inferred.")
    if isinstance(column, str):
        if columns is None:
            raise ValueError(f"{name}='{column}' requires a headered CSV file.")
        if column not in columns:
            raise ValueError(f"{name}='{column}' not found in columns: {columns}")
        return int(columns.index(column))
    idx = int(column)
    if idx < 0:
        idx = total_cols + idx
    if idx < 0 or idx >= total_cols:
        raise ValueError(f"{name} index {column} is out of range for {total_cols} columns.")
    return idx


def resolve_sensor_columns(config: Dict, columns: List[str] | None, total_cols: int, reserved=None) -> List[int]:
    sensor_columns = config["dataset"].get("sensor_columns")
    if not isinstance(sensor_columns, list) or not sensor_columns:
        raise ValueError("dataset.sensor_columns must be a non-empty list.")
    resolved = [resolve_column(col, columns, total_cols, "sensor_columns") for col in sensor_columns]
    if not resolved:
        raise ValueError("sensor_columns resolved to an empty list.")
    return resolved


def parse_id_from_path(path: Path, patterns: Sequence[str], field_name: str) -> int:
    text = "/".join(path.parts)
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not parse {field_name} from path: {path}. Configure {field_name}_patterns.")


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = ("X", "y", "subject_id")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}. Required arrays: X, y, subject_id.")
    x = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    subject_id = np.asarray(data["subject_id"], dtype=np.int64)
    session_id = np.asarray(data["session_id"] if "session_id" in data else subject_id, dtype=np.int64)
    if x.ndim == 3:
        raise ValueError(
            f"{path} contains pre-windowed X with shape {x.shape}. Provide continuous X [time, channels] for this pipeline."
        )
    return x, y, subject_id, session_id


def load_npy(path: Path, config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        item = data.item()
        return (
            np.asarray(item["X"], dtype=np.float32),
            np.asarray(item["y"], dtype=np.int64),
            np.asarray(item["subject_id"], dtype=np.int64),
            np.asarray(item.get("session_id", item["subject_id"]), dtype=np.int64),
        )
    raise ValueError(f"{path} must contain a dict-like object with X, y, subject_id, and optional session_id.")


def load_mat(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required for MAT files. Install scipy or convert TSA data to NPZ/CSV.") from exc
    data = loadmat(path)
    aliases = {
        "X": ["X", "data", "signals", "features"],
        "y": ["y", "labels", "label", "activity"],
        "subject_id": ["subject_id", "subjects", "subject"],
        "session_id": ["session_id", "sessions", "trial_id", "trial"],
    }

    def first_available(names):
        for name in names:
            if name in data:
                return np.asarray(data[name]).squeeze()
        return None

    x = first_available(aliases["X"])
    y = first_available(aliases["y"])
    subject_id = first_available(aliases["subject_id"])
    session_id = first_available(aliases["session_id"])
    if x is None or y is None or subject_id is None:
        raise ValueError(
            f"{path} does not contain recognizable MAT variables. Required aliases: {aliases}"
        )
    if session_id is None:
        session_id = subject_id
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(subject_id, dtype=np.int64),
        np.asarray(session_id, dtype=np.int64),
    )


def validate_continuous_arrays(x: np.ndarray, y: np.ndarray, subject_id: np.ndarray, session_id: np.ndarray, dataset_name: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{dataset_name}: X must have shape [time, channels], got {x.shape}")
    n = len(x)
    if len(y) != n or len(subject_id) != n or len(session_id) != n:
        raise ValueError(
            f"{dataset_name}: X, y, subject_id, session_id must share time length. "
            f"Got {len(x)}, {len(y)}, {len(subject_id)}, {len(session_id)}."
        )


def apply_label_filters(x, y, subject_id, session_id, config: Dict):
    invalid_labels = set(int(v) for v in config["dataset"].get("invalid_labels", []))
    if invalid_labels:
        valid = ~np.isin(y, list(invalid_labels))
        x, y, subject_id, session_id = x[valid], y[valid], subject_id[valid], session_id[valid]
    return x, y, subject_id, session_id


def load_table_dataset(config: Dict, dataset_name: str, default_patterns: Sequence[str], default_subject_patterns: Sequence[str]):
    raw_dir = require_raw_dir(config, dataset_name)
    patterns = config["dataset"].get("file_patterns", list(default_patterns))
    files = list_files(raw_dir, patterns)
    if not files:
        raise dataset_error(dataset_name, raw_dir, str(config["dataset"].get("config_hint", "")))

    xs, ys, subjects, sessions = [], [], [], []
    has_header = bool(config["dataset"].get("has_header", False))
    label_column = config["dataset"].get("label_column")
    subject_column = config["dataset"].get("subject_column")
    session_column = config["dataset"].get("session_column")
    timestamp_column = config["dataset"].get("timestamp_column")
    subject_patterns = config["dataset"].get("subject_patterns", list(default_subject_patterns))
    session_patterns = config["dataset"].get("session_patterns", [r"trial[_-]?(\d+)", r"run[_-]?(\d+)", r"session[_-]?(\d+)"])
    missing_strategy = config["dataset"].get("missing_value_strategy", "interpolate")

    for file_index, path in enumerate(files):
        table, columns = read_table_file(path, has_header=has_header)
        total_cols = table.shape[1]
        label_idx = resolve_column(label_column, columns, total_cols, "label_column")
        subject_idx = resolve_column(subject_column, columns, total_cols, "subject_column") if subject_column is not None else None
        session_idx = resolve_column(session_column, columns, total_cols, "session_column") if session_column is not None else None
        timestamp_idx = resolve_column(timestamp_column, columns, total_cols, "timestamp_column") if timestamp_column is not None else None
        reserved = [label_idx]
        if subject_idx is not None:
            reserved.append(subject_idx)
        if session_idx is not None:
            reserved.append(session_idx)
        if timestamp_idx is not None:
            reserved.append(timestamp_idx)
        sensor_cols = resolve_sensor_columns(config, columns, total_cols, reserved)
        x = table[:, sensor_cols].astype(np.float32)
        y = table[:, label_idx].astype(np.int64)
        if subject_idx is None:
            subject_value = parse_id_from_path(path, subject_patterns, "subject_id")
            subject_id = np.full(len(y), subject_value, dtype=np.int64)
        else:
            subject_id = table[:, subject_idx].astype(np.int64)
        if session_idx is None:
            try:
                session_value = parse_id_from_path(path, session_patterns, "session_id")
            except ValueError:
                session_value = file_index
            session_id = np.full(len(y), session_value, dtype=np.int64)
        else:
            session_id = table[:, session_idx].astype(np.int64)
        x = impute_missing_values(x, strategy=missing_strategy)
        xs.append(x)
        ys.append(y)
        subjects.append(subject_id)
        sessions.append(session_id)

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subject_id = np.concatenate(subjects, axis=0)
    session_id = np.concatenate(sessions, axis=0)
    validate_continuous_arrays(x, y, subject_id, session_id, dataset_name)
    return x, y, subject_id, session_id


def load_np_files_dataset(config: Dict, dataset_name: str):
    raw_dir = require_raw_dir(config, dataset_name)
    npz_files = list_files(raw_dir, config["dataset"].get("npz_patterns", ["*.npz"]))
    npy_files = list_files(raw_dir, config["dataset"].get("npy_patterns", ["*.npy"]))
    mat_files = list_files(raw_dir, config["dataset"].get("mat_patterns", ["*.mat"]))
    if len(npz_files) == 1:
        arrays = load_npz(npz_files[0])
    elif len(npy_files) == 1:
        arrays = load_npy(npy_files[0], config)
    elif len(mat_files) == 1:
        arrays = load_mat(mat_files[0])
    else:
        raise dataset_error(dataset_name, raw_dir, str(config["dataset"].get("config_hint", "")))
    x, y, subject_id, session_id = arrays
    x = impute_missing_values(x, strategy=config["dataset"].get("missing_value_strategy", "interpolate"))
    validate_continuous_arrays(x, y, subject_id, session_id, dataset_name)
    return x, y, subject_id, session_id
