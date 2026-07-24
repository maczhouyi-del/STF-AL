from __future__ import annotations

from .data_loader import load_table_dataset


def load_opportunity_continuous(config):
    """Load Opportunity continuous samples from .dat/.data text files."""
    return load_table_dataset(
        config,
        dataset_name="Opportunity",
        default_patterns=["*.dat", "*.data", "*.txt"],
        default_subject_patterns=[r"S(\d+)", r"subject[_-]?(\d+)", r"user[_-]?(\d+)"],
    )
