#!/usr/bin/env python3
"""Backward-compatible imports for the dataset package."""
from eval_snn_ann.data import *  # noqa: F401,F403
from eval_snn_ann.data.synthetic import (
    _generate_synthetic_ecg,
    _generate_synthetic_eeg,
    _generate_synthetic_general_time_series,
    _generate_synthetic_har,
)


if __name__ == "__main__":
    print_available_datasets_table()
