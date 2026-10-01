#!/usr/bin/env python3
"""
Unit-Tests und Validierung für den Datensatz-Hub und die Trainings-/Export-Pipeline.
"""

import sys
import unittest
import numpy as np

from datasets_hub import (
    AVAILABLE_DATASETS,
    DatasetInfo,
    list_available_datasets,
    _generate_synthetic_speech_commands,
    _generate_synthetic_har,
    _generate_synthetic_eeg,
    _generate_synthetic_ecg,
    _generate_synthetic_general_time_series,
)


class TestDatasetsHub(unittest.TestCase):
    def test_list_datasets(self):
        datasets = list_available_datasets()
        self.assertGreaterEqual(len(datasets), 5)
        names = [d.name for d in datasets]
        self.assertIn("mnist", names)
        self.assertIn("fashion_mnist", names)
        self.assertIn("cifar10", names)
        self.assertIn("speech_commands", names)
        self.assertIn("har_time_series", names)
        self.assertIn("eeg_data", names)
        self.assertIn("ecg_data", names)

    def test_synthetic_speech_commands_shapes(self):
        train_ds, test_ds = _generate_synthetic_speech_commands(num_train=20, num_test=10, seed=42)
        self.assertEqual(len(train_ds), 20)
        self.assertEqual(len(test_ds), 10)
        x0, y0 = train_ds[0]
        # x shape: (1, 16, 16) -> 256
        self.assertEqual(tuple(x0.shape), (1, 16, 16))
        self.assertEqual(x0.numel(), 256)
        self.assertTrue(0 <= int(y0) < 10)

    def test_synthetic_har_shapes(self):
        train_ds, test_ds = _generate_synthetic_har(num_train=24, num_test=12, seed=42)
        self.assertEqual(len(train_ds), 24)
        self.assertEqual(len(test_ds), 12)
        x0, y0 = train_ds[0]
        # x shape: (1, 6, 20) -> 120
        self.assertEqual(tuple(x0.shape), (1, 6, 20))
        self.assertEqual(x0.numel(), 120)
        self.assertTrue(0 <= int(y0) < 6)

    def test_synthetic_eeg_shapes(self):
        train_ds, test_ds = _generate_synthetic_eeg(num_train=16, num_test=8, seed=42)
        self.assertEqual(len(train_ds), 16)
        self.assertEqual(len(test_ds), 8)
        x0, y0 = train_ds[0]
        # x shape: (1, 8, 16) -> 128
        self.assertEqual(tuple(x0.shape), (1, 8, 16))
        self.assertEqual(x0.numel(), 128)
        self.assertTrue(0 <= int(y0) < 4)

    def test_synthetic_ecg_shapes(self):
        train_ds, test_ds = _generate_synthetic_ecg(num_train=20, num_test=10, seed=42)
        self.assertEqual(len(train_ds), 20)
        self.assertEqual(len(test_ds), 10)
        x0, y0 = train_ds[0]
        # x shape: (1, 1, 128) -> 128
        self.assertEqual(tuple(x0.shape), (1, 1, 128))
        self.assertEqual(x0.numel(), 128)
        self.assertTrue(0 <= int(y0) < 5)

    def test_synthetic_general_time_series_shapes(self):
        train_ds, test_ds = _generate_synthetic_general_time_series(num_train=16, num_test=8, seed=42)
        self.assertEqual(len(train_ds), 16)
        self.assertEqual(len(test_ds), 8)
        x0, y0 = train_ds[0]
        # x shape: (1, 1, 128) -> 128
        self.assertEqual(tuple(x0.shape), (1, 1, 128))
        self.assertEqual(x0.numel(), 128)
        self.assertTrue(0 <= int(y0) < 4)


if __name__ == "__main__":
    unittest.main()
