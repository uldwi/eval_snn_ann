#!/usr/bin/env python3
"""
Test post-training optimization (pruning, quantization, centroid clustering, sparsity) through train_mnist.py.
"""
import copy
import json
import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets_hub import DatasetInfo
from train_mnist import (
    ConfigurableANN,
    ConfigurableSpikingJellySNN,
    build_model_from_config,
    compute_model_signature,
    compute_model_sparsity,
    apply_pruning,
    apply_quantization,
    apply_weight_clustering,
    optimize_model,
    export_pytorch_to_tflite,
    export_c_files,
    HAS_SPIKINGJELLY,
)


class TestModelOptimization(unittest.TestCase):
    def setUp(self):
        self.dataset_info = DatasetInfo(
            name="mnist",
            input_size=784,
            num_classes=10,
            in_channels=1,
            in_height=28,
            in_width=28,
            domain="vision",
            description="MNIST",
        )

        self.ann_cfg = {
            "name": "TestANN",
            "type": "ann",
            "model": {
                "name": "TestANN",
                "type": "ann",
                "dataset": "mnist",
                "conv_layers": [
                    {"out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1, "pool_size": 2}
                ],
                "hidden_layers": [32],
                "activation": "relu",
            },
            "training": {
                "epochs": 1,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
            "optimization": {
                "pruning": {"enabled": False, "amount": 0.4},
                "quantization": {"enabled": False, "type": "int8"},
                "clustering": {"enabled": False, "num_clusters": 16},
            },
        }

    def test_sparsity_calculation(self):
        model = ConfigurableANN(input_size=784, conv_layers=[], hidden_layers=[32], output_size=10)
        # Set some weights to zero
        with torch.no_grad():
            for m in model.fc_layers:
                m.weight.data[:, :10] = 0.0

        sparsity = compute_model_sparsity(model)
        self.assertGreater(sparsity["total_weights"], 0)
        self.assertGreater(sparsity["zero_weights"], 0)
        self.assertGreater(sparsity["sparsity_pct"], 0.0)
        self.assertEqual(sparsity["zero_weights"] + sparsity["non_zero_weights"], sparsity["total_weights"])

    def test_pruning_l1_unstructured(self):
        model = build_model_from_config(self.ann_cfg, self.dataset_info)
        prune_cfg = {"enabled": True, "method": "l1_unstructured", "amount": 0.5, "layers": ["conv", "fc"]}
        res = apply_pruning(model, prune_cfg)
        self.assertTrue(res["applied"])
        self.assertGreater(len(res["pruned_modules"]), 0)

        sparsity = compute_model_sparsity(model)
        # Sollte nahe bei 50% liegen (mindestens 45%)
        self.assertGreaterEqual(sparsity["sparsity_pct"], 45.0)

    def test_pruning_threshold(self):
        model = build_model_from_config(self.ann_cfg, self.dataset_info)
        prune_cfg = {"enabled": True, "method": "threshold", "threshold": 0.05, "layers": ["conv", "fc"]}
        res = apply_pruning(model, prune_cfg)
        self.assertTrue(res["applied"])

        # Check that no weight magnitude lies between zero and the threshold.
        for _, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                w_abs = module.weight.data.abs()
                nonzero = w_abs[w_abs > 0]
                if len(nonzero) > 0:
                    self.assertTrue((nonzero >= 0.05).all().item())

    def test_quantization_int8_symmetric(self):
        model = build_model_from_config(self.ann_cfg, self.dataset_info)
        quant_cfg = {"enabled": True, "type": "int8", "mode": "symmetric", "per_channel": True}
        res = apply_quantization(model, quant_cfg)
        self.assertTrue(res["applied"])
        self.assertGreater(len(res["quantized_modules"]), 0)

    def test_quantization_int4(self):
        model = build_model_from_config(self.ann_cfg, self.dataset_info)
        quant_cfg = {"enabled": True, "type": "int4", "mode": "symmetric", "per_channel": False}
        res = apply_quantization(model, quant_cfg)
        self.assertTrue(res["applied"])

    def test_weight_clustering(self):
        model = build_model_from_config(self.ann_cfg, self.dataset_info)
        cluster_cfg = {"enabled": True, "num_clusters": 16}
        res = apply_weight_clustering(model, cluster_cfg)
        self.assertTrue(res["applied"])

    def test_full_optimization_pipeline(self):
        cfg = copy.deepcopy(self.ann_cfg)
        cfg["optimization"] = {
            "pruning": {"enabled": True, "amount": 0.3},
            "quantization": {"enabled": True, "type": "int8"},
        }
        model = build_model_from_config(cfg, self.dataset_info)
        opt_model, summary = optimize_model(model, cfg)
        self.assertTrue(summary["applied"])
        self.assertIn("post_sparsity", summary)
        self.assertGreaterEqual(summary["post_sparsity"]["sparsity_pct"], 25.0)

    def test_signature_changes_with_optimization(self):
        cfg1 = copy.deepcopy(self.ann_cfg)
        cfg2 = copy.deepcopy(self.ann_cfg)
        cfg2["optimization"]["pruning"] = {"enabled": True, "amount": 0.5}

        sig1, hash1 = compute_model_signature(cfg1, self.dataset_info)
        sig2, hash2 = compute_model_signature(cfg2, self.dataset_info)

        self.assertNotEqual(hash1, hash2)
        self.assertFalse(sig1["optimization"]["pruning"]["enabled"])
        self.assertTrue(sig2["optimization"]["pruning"]["enabled"])

    def test_snn_optimization_and_export(self):
        if not HAS_SPIKINGJELLY:
            self.skipTest("SpikingJelly is not available")

        snn_cfg = {
            "name": "TestSNN",
            "type": "snn",
            "model": {
                "name": "TestSNN",
                "type": "snn",
                "dataset": "mnist",
                "neuron_type": "lif",
                "encoding_type": "ttfs",
                "conv_layers": [],
                "hidden_layers": [32],
                "num_steps": 5,
                "beta": 0.9,
                "threshold": 1.0,
            },
            "training": {"epochs": 1, "batch_size": 32},
            "optimization": {
                "pruning": {"enabled": True, "amount": 0.4},
                "quantization": {"enabled": True, "type": "int8"},
            },
        }

        model = build_model_from_config(snn_cfg, self.dataset_info)
        opt_model, summary = optimize_model(model, snn_cfg)
        self.assertTrue(summary["applied"])

        # Test C-Export des optimierten SNN
        with tempfile.TemporaryDirectory() as tmpdir:
            model_entry = {
                "name": "TestSNN",
                "type": "snn",
                "dataset": "mnist",
                "config": snn_cfg,
                "model": opt_model,
                "test_dataset": [(torch.zeros(1, 28, 28), 0)],
                "tflite_bytes": b"",
                "model_size_bytes": 1000,
            }
            export_c_files(model_entry, output_dir=tmpdir)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "model_data.h")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "model_data.cc")))


if __name__ == "__main__":
    unittest.main()
