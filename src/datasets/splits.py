from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

ALLOWED_DATASETS = {"pamap2", "dsads", "opportunity", "tsa"}


def build_subject_partitions(
    subject_ids: Sequence[int],
    dataset_name: str,
    num_partitions: int = 4,
) -> List[List[int]]:
    dataset_name = dataset_name.lower()
    if dataset_name not in ALLOWED_DATASETS:
        raise ValueError(f"Unsupported dataset.name='{dataset_name}'. Allowed values: {sorted(ALLOWED_DATASETS)}")
    subjects = sorted({int(s) for s in subject_ids})
    if dataset_name in {"pamap2", "dsads"} and len(subjects) >= 8:
        return [subjects[i : i + 2] for i in range(0, 8, 2)]
    if dataset_name == "opportunity" and len(subjects) >= 4:
        return [[s] for s in subjects[:4]]

    partitions = [[] for _ in range(num_partitions)]
    for idx, subject in enumerate(subjects):
        partitions[idx % num_partitions].append(subject)
    return partitions


def subject_partitions(
    subject_ids: Sequence[int],
    dataset_name: str,
    num_partitions: int = 4,
) -> List[List[int]]:
    return build_subject_partitions(subject_ids, dataset_name, num_partitions)


def validate_no_subject_leakage(source_subjects: Sequence[int], target_subjects: Sequence[int]) -> None:
    overlap = set(int(s) for s in source_subjects) & set(int(s) for s in target_subjects)
    if overlap:
        raise ValueError(f"Subject leakage detected between source and target subjects: {sorted(overlap)}")


def leave_one_partition_out_split(
    partitions: Sequence[Sequence[int]],
    target_partition: int,
) -> Tuple[List[int], List[int]]:
    if target_partition < 0 or target_partition >= len(partitions):
        raise ValueError(
            f"target_partition={target_partition} is invalid for {len(partitions)} partitions."
        )
    target_subjects = [int(s) for s in partitions[target_partition]]
    source_subjects = [
        int(s)
        for idx, partition in enumerate(partitions)
        if idx != target_partition
        for s in partition
    ]
    validate_no_subject_leakage(source_subjects, target_subjects)
    return source_subjects, target_subjects


def get_source_target_indices(
    subject_ids: np.ndarray,
    partitions: Sequence[Sequence[int]],
    target_partition: int,
) -> Tuple[np.ndarray, np.ndarray]:
    source_subjects, target_subjects = leave_one_partition_out_split(partitions, target_partition)
    target_subjects = set(target_subjects)
    target_mask = np.asarray([int(s) in target_subjects for s in subject_ids], dtype=bool)
    source_mask = ~target_mask
    validate_no_subject_leakage(
        np.unique(subject_ids[source_mask]).tolist(),
        np.unique(subject_ids[target_mask]).tolist(),
    )
    if not source_mask.any() or not target_mask.any():
        raise ValueError("Source and target splits must both contain at least one sample.")
    return source_mask, target_mask


def leave_one_partition_out_masks(
    subject_ids: np.ndarray,
    partitions: Sequence[Sequence[int]],
    target_partition: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return get_source_target_indices(subject_ids, partitions, target_partition)
