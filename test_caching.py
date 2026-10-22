#!/usr/bin/env python3
"""
Testet das automatische Speichern und Caching der Modelle in train_mnist.py.
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

# Stelle sicher, dass das aktuelle Verzeichnis im Modulpfad ist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_mnist import compute_model_signature, DEFAULT_CONFIG


class TestModelCaching(unittest.TestCase):
    def test_signature_deterministic(self):
        cfg1 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2 = copy.deepcopy(DEFAULT_CONFIG)
        sig1, hash1 = compute_model_signature(cfg1)
        sig2, hash2 = compute_model_signature(cfg2)
        self.assertEqual(sig1, sig2)
        self.assertEqual(hash1, hash2)

    def test_signature_changes_on_epochs(self):
        cfg1 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2["training"]["epochs"] = 10
        _, hash1 = compute_model_signature(cfg1)
        _, hash2 = compute_model_signature(cfg2)
        self.assertNotEqual(hash1, hash2)

    def test_signature_changes_on_lr(self):
        cfg1 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2["training"]["learning_rate"] = 0.01
        _, hash1 = compute_model_signature(cfg1)
        _, hash2 = compute_model_signature(cfg2)
        self.assertNotEqual(hash1, hash2)

    def test_signature_changes_on_architecture(self):
        cfg1 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2 = copy.deepcopy(DEFAULT_CONFIG)
        cfg2["model"]["hidden_layers"] = [128, 64]
        _, hash1 = compute_model_signature(cfg1)
        _, hash2 = compute_model_signature(cfg2)
        self.assertNotEqual(hash1, hash2)

    def test_signature_snn_params(self):
        cfg_snn1 = {
            "name": "SNN_Test",
            "type": "snn",
            "dataset": "mnist",
            "model": {
                "type": "snn",
                "conv_layers": [],
                "hidden_layers": [64],
                "neuron_type": "lif",
                "encoding_type": "ttfs",
                "num_steps": 10,
                "beta": 0.9,
                "threshold": 1.0,
            },
            "training": {"epochs": 3, "batch_size": 64, "learning_rate": 0.005},
        }
        cfg_snn2 = copy.deepcopy(cfg_snn1)
        cfg_snn2["model"]["beta"] = 0.8
        _, hash1 = compute_model_signature(cfg_snn1)
        _, hash2 = compute_model_signature(cfg_snn2)
        self.assertNotEqual(hash1, hash2)


if __name__ == "__main__":
    unittest.main()
