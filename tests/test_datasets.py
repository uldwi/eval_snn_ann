#!/usr/bin/env python3
"""
Unit-Tests und Validierung für den Datensatz-Hub und die Trainings-/Export-Pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np

from datasets_hub import (
    AVAILABLE_DATASETS,
    DatasetInfo,
    list_available_datasets,
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

    def test_alias_normalization(self):
        from datasets_hub import normalize_dataset_name
        self.assertEqual(normalize_dataset_name("kws"), "speech_commands")
        self.assertEqual(normalize_dataset_name("uci_har"), "har_time_series")
        self.assertEqual(normalize_dataset_name("eeg_bands"), "eeg_data")
        self.assertEqual(normalize_dataset_name("ecg_heartbeat"), "ecg_data")
        self.assertEqual(normalize_dataset_name("mnist"), "mnist")

    def test_config_models_dimensions_match_datasets(self):
        import torch
        from train_mnist import load_config, get_model_configs, ConfigurableANN, ConfigurableSpikingJellySNN
        from datasets_hub import get_dataset, normalize_dataset_name

        cfg = load_config("config.yaml")
        all_models = get_model_configs(cfg)
        self.assertGreaterEqual(len(all_models), 18)

        for m_cfg in all_models:
            ds_name = m_cfg["dataset"]
            canon = normalize_dataset_name(ds_name)
            train_ds, _, dinfo = get_dataset(canon)
            batch_x = torch.stack([train_ds[i][0] for i in range(4)])
            m_dict = m_cfg["model"]
            if m_cfg["type"] == "snn":
                net = ConfigurableSpikingJellySNN(
                    input_size=dinfo.input_size,
                    conv_layers=m_dict.get("conv_layers", []),
                    hidden_layers=m_dict.get("hidden_layers", []),
                    output_size=dinfo.num_classes,
                    in_channels=dinfo.in_channels,
                    in_height=dinfo.in_height,
                    in_width=dinfo.in_width,
                    num_steps=2,
                )
            else:
                net = ConfigurableANN(
                    input_size=dinfo.input_size,
                    conv_layers=m_dict.get("conv_layers", []),
                    hidden_layers=m_dict.get("hidden_layers", []),
                    output_size=dinfo.num_classes,
                    in_channels=dinfo.in_channels,
                    in_height=dinfo.in_height,
                    in_width=dinfo.in_width,
                )
            out = net(batch_x)
            self.assertEqual(out.shape, (4, dinfo.num_classes))


if __name__ == "__main__":
    unittest.main()
