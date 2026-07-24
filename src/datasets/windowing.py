from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import numpy as np


def majority_label(labels: np.ndarray, invalid_labels: set[int] | None = None) -> int:
    """Return the majority label for one window."""
    if invalid_labels:
        labels = np.asarray([label for label in labels.tolist() if int(label) not in invalid_labels])
    if len(labels) == 0:
        raise ValueError("Cannot compute majority label for a window with no valid labels.")
    counts = Counter(labels.tolist())
    return int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])


def sliding_window_sequence(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: int,
    window_length: int,
    overlap: float,
    invalid_labels: set[int] | None = None,
    max_invalid_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [time, channels], got {x.shape}")
    if len(x) != len(y):
        raise ValueError("x and y must have the same time dimension.")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    step = max(1, int(round(window_length * (1.0 - overlap))))
    xs, ys, subjects = [], [], []
    for start in range(0, len(x) - window_length + 1, step):
        end = start + window_length
        window_labels = y[start:end]
        if invalid_labels:
            invalid_ratio = float(np.isin(window_labels, list(invalid_labels)).mean())
            if invalid_ratio > max_invalid_ratio:
                continue
        xs.append(x[start:end])
        ys.append(majority_label(window_labels, invalid_labels=invalid_labels))
        subjects.append(subject_id)
    if not xs:
        return (
            np.empty((0, window_length, x.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(subjects, dtype=np.int64),
    )


def create_sliding_windows(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    window_length: int,
    overlap: float,
    group_ids: np.ndarray | None = None,
    invalid_labels: set[int] | None = None,
    max_invalid_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Window a continuous dataset without crossing subject/session boundaries."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    if group_ids is None:
        group_ids = subject_ids
    group_ids = np.asarray(group_ids, dtype=np.int64)
    if len(group_ids) != len(subject_ids):
        raise ValueError("group_ids and subject_ids must have the same length.")
    windows, labels, subjects = [], [], []
    group_window_counts: Dict[str, int] = {}
    keys = sorted({(int(s), int(g)) for s, g in zip(subject_ids, group_ids)})
    for subject, group in keys:
        mask = (subject_ids == subject) & (group_ids == group)
        wx, wy, ws = sliding_window_sequence(
            x[mask],
            y[mask],
            int(subject),
            window_length,
            overlap,
            invalid_labels=invalid_labels,
            max_invalid_ratio=max_invalid_ratio,
        )
        if len(wx):
            windows.append(wx)
            labels.append(wy)
            subjects.append(ws)
            group_window_counts[f"subject={subject},group={group}"] = int(len(wx))
    if not windows:
        raise ValueError("Sliding window generation produced zero windows.")
    x_windows = np.concatenate(windows, axis=0)
    y_windows = np.concatenate(labels, axis=0)
    subject_windows = np.concatenate(subjects, axis=0)
    metadata = {
        "num_windows": int(len(x_windows)),
        "window_length": int(window_length),
        "overlap": float(overlap),
        "group_window_counts": group_window_counts,
        "class_window_counts": {
            int(label): int((y_windows == label).sum()) for label in sorted(np.unique(y_windows).tolist())
        },
        "subject_window_counts": {
            int(subject): int((subject_windows == subject).sum())
            for subject in sorted(np.unique(subject_windows).tolist())
        },
    }
    return x_windows, y_windows, subject_windows, metadata
