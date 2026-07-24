from __future__ import annotations

from .data_loader import load_table_dataset


def load_pamap2_continuous(config):
    """Load PAMAP2/PAMAP continuous samples.

    Feature columns are controlled by YAML. For official PAMAP2 Protocol files,
    the default label column is column 1 and the subject id is parsed from
    filenames such as subject101.dat.
    """
    return load_table_dataset(
        config,
        dataset_name="PAMAP2",
        default_patterns=["*.dat", "*.txt"],
        default_subject_patterns=[r"subject10?(\d+)", r"subject[_-]?(\d+)", r"subj[_-]?(\d+)"],
    )
