#!/usr/bin/env python3
"""
Unit tests for event-based SNN implementation (host and C export).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from train_mnist import (
    ConfigurableSpikingJellySNN,
    compute_model_signature,
    build_model_from_config,
    export_c_files,
)
from datasets_hub import DatasetInfo


class TestEventBasedSNN(unittest.TestCase):
    def test_snn_initialization_with_execution_mode(self):
        snn_dense = ConfigurableSpikingJellySNN(
            input_size=784,
            hidden_layers=[64, 32],
            output_size=10,
            execution_mode="dense",
            encoding_type="ttfs",
        )
        self.assertEqual(snn_dense.execution_mode, "dense")

        snn_event = ConfigurableSpikingJellySNN(
            input_size=784,
            hidden_layers=[64, 32],
            output_size=10,
            execution_mode="event_based",
            encoding_type="ttfs",
        )
        self.assertEqual(snn_event.execution_mode, "event_based")

    def test_event_simulation_and_synops(self):
        snn = ConfigurableSpikingJellySNN(
            input_size=64,
            hidden_layers=[32, 10],
            output_size=10,
            num_steps=5,
            encoding_type="ttfs",
            execution_mode="event_based",
        )
        x = torch.rand(4, 64)
        out_spikes, stats = snn.simulate_event_based(x)
        self.assertEqual(out_spikes.shape, (4, 10))
        self.assertIn("total_spikes", stats)
        self.assertIn("synops", stats)
        self.assertIn("possible_ops", stats)
        self.assertIn("synops_sparsity_pct", stats)
        self.assertGreaterEqual(stats["synops"], 0)
        self.assertGreaterEqual(stats["synops_sparsity_pct"], 0.0)
        self.assertLessEqual(stats["synops_sparsity_pct"], 100.0)

    def test_conv_snn_event_simulation(self):
        conv_cfg = [
            {"out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1, "pool_size": 2}
        ]
        snn = ConfigurableSpikingJellySNN(
            input_size=16 * 16,
            in_channels=1,
            in_height=16,
            in_width=16,
            conv_layers=conv_cfg,
            hidden_layers=[16],
            output_size=10,
            num_steps=5,
            encoding_type="ttfs",
            execution_mode="event_based",
        )
        x = torch.rand(2, 1, 16, 16)
        out_spikes, stats = snn.simulate_event_based(x)
        self.assertEqual(out_spikes.shape, (2, 10))
        self.assertGreaterEqual(stats["synops"], 0)

    def test_signature_hashing_distinguishes_execution_mode(self):
        cfg_dense = {
            "name": "SNN_Test",
            "model": {
                "type": "snn",
                "neuron_type": "lif",
                "encoding_type": "ttfs",
                "execution_mode": "dense",
                "hidden_layers": [64],
                "num_steps": 10,
            },
            "training": {"epochs": 5, "batch_size": 64},
        }
        cfg_event = {
            "name": "SNN_Test",
            "model": {
                "type": "snn",
                "neuron_type": "lif",
                "encoding_type": "ttfs",
                "execution_mode": "event_based",
                "hidden_layers": [64],
                "num_steps": 10,
            },
            "training": {"epochs": 5, "batch_size": 64},
        }

        sig1, hash1 = compute_model_signature(cfg_dense)
        sig2, hash2 = compute_model_signature(cfg_event)

        self.assertNotEqual(hash1, hash2)
        self.assertEqual(sig1["snn_params"]["execution_mode"], "dense")
        self.assertEqual(sig2["snn_params"]["execution_mode"], "event_based")

    def test_c_export_contains_execution_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_info = DatasetInfo("test_ds", "Vision", "Desc", 784, 1, 28, 28, 10)
            cfg = {
                "name": "SNN_Event_Test",
                "model": {
                    "type": "snn",
                    "neuron_type": "lif",
                    "encoding_type": "ttfs",
                    "execution_mode": "event_based",
                    "hidden_layers": [32],
                    "num_steps": 5,
                    "input_size": 784,
                    "output_size": 10,
                },
                "training": {"dataset": "mnist"},
                "evaluation": {"num_test_samples": 5},
            }
            model = build_model_from_config(cfg, ds_info)
            dummy_x = torch.rand(5, 784)
            dummy_y = torch.tensor([0, 1, 2, 3, 4])
            dataset = TensorDataset(dummy_x, dummy_y)

            entry = {
                "name": "SNN_Event_Test",
                "type": "snn",
                "config": cfg,
                "model": model,
                "test_dataset": dataset,
                "model_size_bytes": 1024,
            }

            export_c_files(entry, output_dir=tmpdir)

            header_file = Path(tmpdir) / "model_data.h"
            cc_file = Path(tmpdir) / "model_data.cc"

            self.assertTrue(header_file.exists())
            self.assertTrue(cc_file.exists())

            header_content = header_file.read_text(encoding="utf-8")
            cc_content = cc_file.read_text(encoding="utf-8")

            self.assertIn("#define EXECUTION_MODE_DENSE", header_content)
            self.assertIn("#define EXECUTION_MODE_EVENT_BASED 1", header_content)
            self.assertIn("int execution_mode;", header_content)
            self.assertIn(".execution_mode = 1", cc_content)


if __name__ == "__main__":
    unittest.main()
