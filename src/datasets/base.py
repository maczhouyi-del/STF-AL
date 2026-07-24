from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .splits import ALLOWED_DATASETS, leave_one_partition_out_masks, subject_partitions
from .transforms import ChannelNormalizer
from .windowing import create_sliding_windows


class WindowedHARDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, domain_id: int):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.subject_ids = torch.as_tensor(subject_ids, dtype=torch.long)
        self.domain_id = int(domain_id)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "subject_id": self.subject_ids[idx],
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
        }


def load_continuous_arrays(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    name = str(config["dataset"]["name"]).lower()
    if name not in ALLOWED_DATASETS:
        raise ValueError(f"Unsupported dataset.name='{name}'. Allowed values: {sorted(ALLOWED_DATASETS)}")
    if name == "pamap2":
        from .pamap2 import load_pamap2_continuous

        return load_pamap2_continuous(config)
    if name == "dsads":
        from .dsads import load_dsads_continuous

        return load_dsads_continuous(config)
    if name == "opportunity":
        from .opportunity import load_opportunity_continuous

        return load_opportunity_continuous(config)
    if name == "tsa":
        from .tsa import load_tsa_continuous

        return load_tsa_continuous(config)
    raise ValueError(f"Unsupported dataset.name='{name}'. Allowed values: {sorted(ALLOWED_DATASETS)}")


def load_windowed_arrays(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    loaded = load_continuous_arrays(config)
    if len(loaded) != 4:
        raise ValueError("Dataset loader must return X, y, subject_id, session_id.")
    x, y, subjects, groups = loaded
    invalid_labels = set(int(v) for v in config["dataset"].get("invalid_labels", []))
    return create_sliding_windows(
        x=x,
        y=y,
        subject_ids=subjects,
        window_length=int(config["dataset"]["window_length"]),
        overlap=float(config["dataset"]["overlap"]),
        group_ids=groups,
        invalid_labels=invalid_labels,
        max_invalid_ratio=float(config["dataset"].get("max_invalid_ratio", 1.0)),
    )


def remap_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    unique_labels = sorted(int(label) for label in np.unique(y).tolist())
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    mapped = np.asarray([label_to_index[int(label)] for label in y], dtype=np.int64)
    return mapped, label_to_index


def split_and_normalize(
    config: Dict,
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    window_metadata: Dict,
    normalization_stats: Dict | None = None,
) -> Tuple[WindowedHARDataset, WindowedHARDataset, Dict]:
    dataset_name = str(config["dataset"]["name"]).lower()
    y, label_mapping = remap_labels(y)
    partitions = config["dataset"].get("subject_partitions")
    if partitions is None:
        partitions = subject_partitions(
            subjects,
            dataset_name=dataset_name,
            num_partitions=int(config["dataset"].get("num_partitions", 4)),
        )
    else:
        partitions = [[int(subject) for subject in partition] for partition in partitions]
    source_mask, target_mask = leave_one_partition_out_masks(
        subjects,
        partitions,
        target_partition=int(config["dataset"].get("target_partition", 0)),
    )
    x_source, y_source, s_source = x[source_mask], y[source_mask], subjects[source_mask]
    x_target, y_target, s_target = x[target_mask], y[target_mask], subjects[target_mask]
    if str(config["dataset"].get("normalize", "source")).lower() == "source":
        normalizer = ChannelNormalizer.from_dict(normalization_stats) if normalization_stats else ChannelNormalizer.fit(x_source)
        x_source = normalizer.transform(x_source)
        x_target = normalizer.transform(x_target)
        norm_stats = normalizer.to_dict()
    else:
        norm_stats = None
    max_windows_per_subject = int(config["dataset"].get("max_windows_per_subject", 0) or 0)
    if max_windows_per_subject > 0:
        x_source, y_source, s_source = limit_windows_per_subject(x_source, y_source, s_source, max_windows_per_subject)
        x_target, y_target, s_target = limit_windows_per_subject(x_target, y_target, s_target, max_windows_per_subject)
    metadata = {
        "partitions": partitions,
        "source_samples": int(len(x_source)),
        "target_samples": int(len(x_target)),
        "input_channels": int(x.shape[-1]),
        "num_classes": int(max(int(config.get("model", {}).get("num_classes", 0)), len(label_mapping))),
        "label_mapping": label_mapping,
        "windowing": window_metadata,
        "normalization": norm_stats,
    }
    return (
        WindowedHARDataset(x_source, y_source, s_source, domain_id=1),
        WindowedHARDataset(x_target, y_target, s_target, domain_id=0),
        metadata,
    )


def limit_windows_per_subject(x: np.ndarray, y: np.ndarray, subjects: np.ndarray, max_windows: int):
    indices = []
    for subject in sorted(np.unique(subjects).tolist()):
        subject_indices = np.flatnonzero(subjects == subject)
        indices.extend(subject_indices[:max_windows].tolist())
    indices = np.asarray(indices, dtype=np.int64)
    return x[indices], y[indices], subjects[indices]


def split_source_train_val(source_dataset: Dataset, config: Dict) -> Tuple[Dataset, Dataset]:
    split = float(config["training"].get("source_validation_split", 0.1))
    if split <= 0.0 or len(source_dataset) < 2:
        return source_dataset, source_dataset
    val_size = max(1, int(round(len(source_dataset) * split)))
    val_size = min(val_size, len(source_dataset) - 1)
    generator = torch.Generator().manual_seed(int(config.get("runtime", {}).get("seed", 42)))
    indices = torch.randperm(len(source_dataset), generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(source_dataset, train_indices), Subset(source_dataset, val_indices)


def build_dataloaders(config: Dict, normalization_stats: Dict | None = None):
    x, y, subjects, window_metadata = load_windowed_arrays(config)
    source_dataset, target_dataset, metadata = split_and_normalize(
        config,
        x,
        y,
        subjects,
        window_metadata=window_metadata,
        normalization_stats=normalization_stats,
    )
    source_train_dataset, source_val_dataset = split_source_train_val(source_dataset, config)
    metadata["source_train_samples"] = int(len(source_train_dataset))
    metadata["source_val_samples"] = int(len(source_val_dataset))
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    source_loader = DataLoader(
        source_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    source_val_loader = DataLoader(
        source_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    eval_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return source_loader, target_loader, source_val_loader, eval_loader, metadata


def build_eval_dataloader(config: Dict, normalization_stats: Dict | None = None):
    _, _, _, eval_loader, metadata = build_dataloaders(config, normalization_stats=normalization_stats)
    return eval_loader, metadata
