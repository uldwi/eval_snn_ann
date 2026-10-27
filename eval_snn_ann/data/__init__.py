"""Dataset registry, preprocessing, and loading APIs."""
from .hub import get_dataset
from .registry import (
    AVAILABLE_DATASETS, DATASET_ALIASES, DatasetInfo, list_available_datasets,
    normalize_dataset_name, print_available_datasets_table,
)
from .speech_commands import SpeechCommandsDataset

__all__ = [
    "AVAILABLE_DATASETS", "DATASET_ALIASES", "DatasetInfo",
    "SpeechCommandsDataset", "get_dataset", "list_available_datasets",
    "normalize_dataset_name", "print_available_datasets_table",
]
