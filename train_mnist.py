#!/usr/bin/env python3
"""
train_mnist.py

Trainiert ein konfigurierbares künstliches neuronales Netz (ANN / CNN mit PyTorch)
oder ein Spiking Neural Network (SNN / CSNN mit SpikingJelly) auf dem MNIST-Datensatz
und exportiert das Modell wahlweise für TensorFlow Lite Micro oder die C++ SNN Inferenz-Engine
auf dem ESP32-C6.

Unterstützte Funktionen:
- Flexible Schichtarchitekturen (Conv2D, MaxPool2D, AvgPool2D, Flatten, Linear/FullyConnected)
- Frei konfigurierbare Hidden Layers (Tiefe und Breiten)
- Wählbare Aktivierungsfunktionen (ReLU, None) für ANNs
- SpikingJelly SNN-Unterstützung mit LIFNode (Leaky-Integrate-and-Fire) und IFNode (Integrate-and-Fire)
- Wählbare SNN-Eingangs-Encodings: Direct, TTFS (Time-To-First-Spike), Poisson Rate, Periodic Rate
- Individuelle Trainingshyperparameter pro Modell (Epochen, Batch-Größe, Lernrate, Optimierer)
- Early Stopping mit konfigurierbarer Patience, Mindestverbesserung und automatischem Best-Model-Restore
- TFLite FlatBuffer Generator für beliebige FC- und Conv-Netze
- Automatische Erzeugung von model_data.h/cc und mnist_test_data.h für den ESP32
"""

import os
import sys
import copy
import argparse
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

import numpy as np
import flatbuffers
import tflite

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as torch_prune
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from datasets_hub import (
    AVAILABLE_DATASETS,
    DATASET_ALIASES,
    DatasetInfo,
    get_dataset,
    list_available_datasets,
    normalize_dataset_name,
    print_available_datasets_table,
)

try:
    import spikingjelly.activation_based as sj_activation
    from spikingjelly.activation_based import neuron, layer as sj_layer, functional, surrogate
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False

# Standardkonfiguration
DEFAULT_CONFIG = {
    "optimization": {
        "pruning": {
            "enabled": False,
            "method": "l1_unstructured",
            "amount": 0.3,
            "threshold": 0.01,
            "layers": ["conv", "fc"],
            "fine_tune_epochs": 0,
            "fine_tune_lr": 0.0001,
        },
        "quantization": {
            "enabled": False,
            "type": "int8",
            "mode": "symmetric",
            "per_channel": True,
            "layers": ["conv", "fc"],
        },
        "clustering": {
            "enabled": False,
            "num_clusters": 16,
        },
    },
    "training": {
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
        # Optimierer: "adam", "sgd", "adamw", "adagrad", "rmsprop" (pro Modell konfigurierbar)
        "optimizer": "adam",
        "early_stopping": False,
        "patience": 3,
        "min_delta": 0.001,
        "monitor": "test_acc",
        "restore_best_weights": True,
        "data_dir": "./data",
        "save_dir": "./saved_models",
        "auto_save": True,
        "random_seed": 42,
    },
    "evaluation": {
        "num_test_samples": 500,
    },
    "esp32": {
        "trigger_pin": 4,
    },
    "joulescope": {
        "trigger_pin": "in1",
        "timeout": 60.0,
    },
    "model": {
        "name": "ANN_64",
        "type": "ann",
        "input_size": 784,
        "conv_layers": [],
        "hidden_layers": [64],
        "activation": "relu",
        "output_size": 10,
        "dropout_rate": 0.0,
    },
}


def load_config(config_path: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Lädt eine Konfigurationsdatei (YAML oder JSON).
    """
    if config_path is None:
        candidates = ["config.yaml", "config.yml", "config.json"]
        for c in candidates:
            if os.path.isfile(c):
                config_path = c
                break

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))

    if config_path and os.path.isfile(config_path):
        print(f"[Config] Lade Konfiguration aus '{config_path}'...")
        path_str = str(config_path)
        with open(path_str, "r", encoding="utf-8") as f:
            if path_str.endswith((".yaml", ".yml")):
                if not HAS_YAML:
                    raise ImportError("PyYAML ist nicht installiert. Bitte 'pip install pyyaml' ausführen.")
                loaded = yaml.safe_load(f)
            else:
                loaded = json.load(f)

        if isinstance(loaded, dict):
            for section, values in loaded.items():
                if section in cfg and isinstance(values, dict):
                    cfg[section].update(values)
                else:
                    cfg[section] = values
    else:
        if config_path:
            print(f"[Config] Hinweis: Konfigurationsdatei '{config_path}' nicht gefunden. Verwende Standardwerte.")
        else:
            print("[Config] Keine Konfigurationsdatei angegeben. Verwende Standardwerte.")

    return cfg


def parse_hidden_layers(val: Any) -> List[int]:
    """
    Parst und normalisiert die Konfiguration der verborgenen Schichten flexibel.
    """
    if val is None:
        return []
    if isinstance(val, int):
        return [val] if val > 0 else []
    if isinstance(val, (list, tuple)):
        result = []
        for item in val:
            if isinstance(item, (list, tuple)):
                result.extend(parse_hidden_layers(item))
            elif isinstance(item, int):
                if item > 0:
                    result.append(item)
            elif isinstance(item, str):
                item_str = item.strip()
                if item_str:
                    for sub in item_str.replace(",", " ").replace("[", " ").replace("]", " ").split():
                        if sub.isdigit():
                            result.append(int(sub))
        return result
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str.lower() in ("none", "null", "[]"):
            return []
        try:
            parsed = json.loads(val_str)
            return parse_hidden_layers(parsed)
        except Exception:
            parts = val_str.replace(",", " ").replace("[", " ").replace("]", " ").split()
            return [int(p) for p in parts if p.isdigit()]
    return []


def parse_conv_layers(val: Any) -> List[Dict[str, Any]]:
    """
    Parst und normalisiert die Konfiguration der Convolutional Layers flexibel.
    Unterstützt verschiedene Formate wie:
      - [{'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool_size': 2}]
      - [{'channels': 8, 'kernel': 3, 'pool': 2}]
      - [[8, 3, 1, 1, 2]] (out_channels, kernel_size, stride, padding, pool_size)
    """
    if val is None:
        return []
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str.lower() in ("none", "null", "[]"):
            return []
        try:
            parsed = json.loads(val_str)
            return parse_conv_layers(parsed)
        except Exception:
            return []
    if isinstance(val, dict):
        val = [val]
    if isinstance(val, (list, tuple)):
        if val and all(isinstance(x, (int, float)) for x in val):
            val = [val]

        result = []
        for item in val:
            if isinstance(item, dict):
                out_channels = int(item.get("out_channels", item.get("channels", item.get("filters", 8))))
                kernel_size = int(item.get("kernel_size", item.get("kernel", 3)))
                stride = int(item.get("stride", 1))
                padding = item.get("padding", None)
                if padding is None:
                    padding = 1 if kernel_size == 3 else 0
                else:
                    padding = int(padding) if isinstance(padding, (int, str)) and str(padding).isdigit() else 0

                pool_size = item.get("pool_size", item.get("pool", item.get("max_pool", 0)))
                pool_size = int(pool_size) if pool_size else 0
                pool_stride = item.get("pool_stride", pool_size)
                pool_stride = int(pool_stride) if pool_stride else pool_size
                pool_type = str(item.get("pool_type", "max")).lower()

                result.append({
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "padding": padding,
                    "pool_size": pool_size,
                    "pool_stride": pool_stride,
                    "pool_type": pool_type,
                })
            elif isinstance(item, (list, tuple)):
                out_c = int(item[0]) if len(item) > 0 else 8
                k_size = int(item[1]) if len(item) > 1 else 3
                stride = int(item[2]) if len(item) > 2 else 1
                padding = int(item[3]) if len(item) > 3 else (1 if k_size == 3 else 0)
                pool_size = int(item[4]) if len(item) > 4 else 0
                pool_stride = int(item[5]) if len(item) > 5 else pool_size
                result.append({
                    "out_channels": out_c,
                    "kernel_size": k_size,
                    "stride": stride,
                    "padding": padding,
                    "pool_size": pool_size,
                    "pool_stride": pool_stride,
                    "pool_type": "max",
                })
        return result
    return []


def compute_conv_layer_dims(conv_layers: List[Dict[str, Any]], in_c: int = 1, in_h: int = 28, in_w: int = 28) -> Tuple[List[Dict[str, Any]], int]:
    """
    Berechnet die Ein- und Ausgangsdimensionen für jeden Convolutional Layer und den resultierenden Flat-Tensor.
    """
    cur_c, cur_h, cur_w = in_c, in_h, in_w
    resolved_layers = []
    for cl in conv_layers:
        out_c = cl["out_channels"]
        k_size = cl["kernel_size"]
        stride = cl["stride"]
        pad = cl["padding"]
        pool_size = cl["pool_size"]
        pool_stride = cl["pool_stride"]

        out_h = (cur_h + 2 * pad - k_size) // stride + 1
        out_w = (cur_w + 2 * pad - k_size) // stride + 1

        if pool_size > 1:
            pool_out_h = (out_h - pool_size) // pool_stride + 1
            pool_out_w = (out_w - pool_size) // pool_stride + 1
        else:
            pool_out_h = out_h
            pool_out_w = out_w

        resolved_layers.append({
            **cl,
            "in_channels": cur_c,
            "in_h": cur_h,
            "in_w": cur_w,
            "out_h": out_h,
            "out_w": out_w,
            "pool_out_h": pool_out_h,
            "pool_out_w": pool_out_w,
        })
        cur_c = out_c
        cur_h = pool_out_h
        cur_w = pool_out_w

    flat_dim = cur_c * cur_h * cur_w
    return resolved_layers, flat_dim


# -------------------------------------------------------------
# Konfigurierbare Modell-Architekturen (ANN & SNN)
# -------------------------------------------------------------
class ConfigurableANN(nn.Module):
    """
    Flexibles Neural Network (MLP oder CNN), dessen Faltungs- und Fully-Connected-Schichten,
    Eingangs-/Ausgangsgrößen und Aktivierungsfunktionen konfigurierbar sind.
    """
    def __init__(
        self,
        input_size: int = 784,
        conv_layers: Any = None,
        hidden_layers: Any = None,
        activation: str = "relu",
        output_size: int = 10,
        in_channels: int = 1,
        in_height: int = 28,
        in_width: int = 28,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        self.in_height = int(in_height)
        self.in_width = int(in_width)
        self.conv_cfg = parse_conv_layers(conv_layers)
        self.hidden_layers = parse_hidden_layers(hidden_layers)
        self.activation_name = (activation or "relu").lower()
        self.output_size = int(output_size)

        self.has_conv = len(self.conv_cfg) > 0
        if self.has_conv:
            self.resolved_conv, self.flat_size = compute_conv_layer_dims(
                self.conv_cfg, self.in_channels, self.in_height, self.in_width
            )
            self.conv_modules = nn.ModuleList()
            for cl in self.resolved_conv:
                mods = []
                mods.append(nn.Conv2d(
                    in_channels=cl["in_channels"],
                    out_channels=cl["out_channels"],
                    kernel_size=cl["kernel_size"],
                    stride=cl["stride"],
                    padding=cl["padding"],
                    bias=True,
                ))
                if self.activation_name == "relu":
                    mods.append(nn.ReLU())
                if cl["pool_size"] > 1:
                    if cl["pool_type"] == "avg":
                        mods.append(nn.AvgPool2d(cl["pool_size"], stride=cl["pool_stride"]))
                    else:
                        mods.append(nn.MaxPool2d(cl["pool_size"], stride=cl["pool_stride"]))
                self.conv_modules.append(nn.Sequential(*mods))
            self.flatten = nn.Flatten()
            fc_in = self.flat_size
        else:
            self.resolved_conv = []
            self.flat_size = self.input_size
            self.conv_modules = nn.ModuleList()
            self.flatten = nn.Flatten()
            fc_in = self.input_size

        all_fc_sizes = [fc_in] + self.hidden_layers + [self.output_size]
        self.fc_layers = nn.ModuleList()
        for i in range(len(all_fc_sizes) - 1):
            self.fc_layers.append(nn.Linear(all_fc_sizes[i], all_fc_sizes[i + 1]))

        self.dropout_rate = float(dropout_rate) if dropout_rate is not None else 0.0
        if self.dropout_rate > 0:
            self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(len(self.hidden_layers))])
        else:
            self.dropouts = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_conv:
            if x.dim() == 2:
                x = x.view(x.size(0), self.in_channels, self.in_height, self.in_width)
            elif x.dim() == 3:
                x = x.unsqueeze(1)
            out = x
            for block in self.conv_modules:
                out = block(out)
            out = self.flatten(out)
        else:
            out = x.view(x.size(0), -1)

        num_fc = len(self.fc_layers)
        for i, layer in enumerate(self.fc_layers):
            out = layer(out)
            if i < num_fc - 1:
                if self.activation_name == "relu":
                    out = F.relu(out)
                elif self.activation_name in ("none", "linear", "identity"):
                    pass
                else:
                    raise ValueError(f"Nicht unterstützte Aktivierungsfunktion: {self.activation_name}")
                if self.dropouts is not None:
                    out = self.dropouts[i](out)
        return out

    def get_structure_summary(self) -> str:
        parts = []
        if self.has_conv:
            for cl in self.resolved_conv:
                p_str = f",p{cl['pool_size']}" if cl['pool_size'] > 1 else ""
                parts.append(f"Conv({cl['in_channels']}->{cl['out_channels']},{cl['kernel_size']}x{cl['kernel_size']}{p_str})")
            parts.append(f"Flat({self.flat_size})")
        else:
            parts.append(str(self.input_size))

        for h in self.hidden_layers:
            parts.append(str(h))
        parts.append(str(self.output_size))

        act_str = f" mit {self.activation_name.upper()}" if (self.hidden_layers or self.has_conv) else ""
        return f"ANN [{' -> '.join(parts)}]{act_str}"


def encode_input(x: torch.Tensor, encoding_type: str, num_steps: int) -> torch.Tensor:
    """
    Kodiert Eingabedaten x (Batch, Features) oder (Batch, C, H, W) über num_steps Zeitschritte.
    Unterstützte Encoding-Typen:
      - 'direct' (Standard): Konstanter Eingangswert über alle Zeitschritte (T, Batch, ...)
      - 'ttfs' / 'latency': Time-To-First-Spike / Latency Coding.
      - 'poisson' / 'rate': Stochastisches Poisson-Rate-Coding mit Wahrscheinlichkeit proportional zu x.
      - 'periodic' / 'rate_periodic': Deterministisches Rate-Coding (k = round(x * T) Spikes).
    """
    enc = str(encoding_type).lower().strip()
    if enc in ("direct", "constant", "raw"):
        return x.unsqueeze(0).repeat(num_steps, *([1] * x.dim()))

    elif enc in ("ttfs", "latency", "time_to_first_spike"):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        t_f = ((num_steps - 1) * (1.0 - x_clamped)).round().long()
        t_indices = torch.arange(num_steps, device=x.device).view(num_steps, *([1] * x.dim()))
        spikes = (t_indices == t_f.unsqueeze(0)).to(x.dtype)
        mask = (x_clamped > 0.0).unsqueeze(0).to(x.dtype)
        return spikes * mask

    elif enc in ("poisson", "rate", "rate_poisson"):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        rand_tensor = torch.rand((num_steps, *x.shape), device=x.device, dtype=x.dtype)
        return (rand_tensor < x_clamped.unsqueeze(0)).to(x.dtype)

    elif enc in ("periodic", "rate_periodic", "phase"):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        k = (x_clamped * num_steps).round().long()
        t_indices = torch.arange(num_steps, device=x.device).view(num_steps, *([1] * x.dim()))
        return (t_indices < k.unsqueeze(0)).to(x.dtype)

    else:
        raise ValueError(f"Unbekannter Encoding-Algorithmus: '{encoding_type}'. Unterstützt: 'direct', 'ttfs', 'poisson', 'periodic'")


def get_encoding_code(encoding_name: str) -> int:
    enc = str(encoding_name).lower().strip()
    if enc in ("ttfs", "latency", "time_to_first_spike"):
        return 1
    elif enc in ("poisson", "rate", "rate_poisson"):
        return 2
    elif enc in ("periodic", "rate_periodic", "phase"):
        return 3
    else:
        return 0  # direct


class ConfigurableSpikingJellySNN(nn.Module):
    """
    Spiking Neural Network (SNN / CSNN) trainiert mit SpikingJelly.
    Unterstützt LIFNode (Leaky-Integrate-and-Fire) und IFNode (Integrate-and-Fire) Neuronen,
    Convolutional Layers, MaxPool2D, Flatten, Fully-Connected Layers und frei wählbare Encodings.
    """
    def __init__(
        self,
        input_size: int = 784,
        conv_layers: Any = None,
        hidden_layers: Any = None,
        output_size: int = 10,
        neuron_type: str = "lif",
        encoding_type: str = "direct",
        execution_mode: str = "dense",
        num_steps: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        in_channels: int = 1,
        in_height: int = 28,
        in_width: int = 28,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        self.in_height = int(in_height)
        self.in_width = int(in_width)
        self.conv_cfg = parse_conv_layers(conv_layers)
        self.hidden_layers = parse_hidden_layers(hidden_layers)
        self.output_size = int(output_size)
        self.neuron_type = str(neuron_type).lower().strip()
        if self.neuron_type not in ("lif", "if"):
            self.neuron_type = "lif"
        self.encoding_type = str(encoding_type).lower().strip() if encoding_type else "direct"
        if self.encoding_type not in ("direct", "ttfs", "latency", "time_to_first_spike", "poisson", "rate", "rate_poisson", "periodic", "rate_periodic", "phase"):
            self.encoding_type = "direct"
        self.execution_mode = str(execution_mode).lower().strip() if execution_mode else "dense"
        if self.execution_mode in ("event", "event_based", "event_driven"):
            self.execution_mode = "event_based"
        else:
            self.execution_mode = "dense"
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.dropout_rate = float(dropout_rate) if dropout_rate is not None else 0.0
        if self.dropout_rate > 0:
            self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(len(self.hidden_layers))])
        else:
            self.dropouts = None

        if self.neuron_type == "if":
            self.beta = 1.0
            self.tau = 1000.0
        else:
            self.beta = float(beta)
            if self.beta >= 1.0:
                self.tau = 1000.0
            elif self.beta <= 0.0:
                self.tau = 1.0
            else:
                self.tau = float(1.0 / (1.0 - self.beta))

        def create_neuron():
            if self.neuron_type == "if":
                return neuron.IFNode(
                    v_threshold=self.threshold,
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(),
                    detach_reset=True,
                )
            else:
                return neuron.LIFNode(
                    tau=self.tau,
                    decay_input=False,
                    v_threshold=self.threshold,
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(),
                    detach_reset=True,
                )

        self.has_conv = len(self.conv_cfg) > 0
        self.conv_layers_list = nn.ModuleList()
        self.conv_lif_list = nn.ModuleList()
        self.conv_pool_list = nn.ModuleList()

        if self.has_conv:
            self.resolved_conv, self.flat_size = compute_conv_layer_dims(
                self.conv_cfg, self.in_channels, self.in_height, self.in_width
            )
            for cl in self.resolved_conv:
                self.conv_layers_list.append(sj_layer.Conv2d(
                    in_channels=cl["in_channels"],
                    out_channels=cl["out_channels"],
                    kernel_size=cl["kernel_size"],
                    stride=cl["stride"],
                    padding=cl["padding"],
                    bias=True,
                ))
                self.conv_lif_list.append(create_neuron())
                if cl["pool_size"] > 1:
                    if cl["pool_type"] == "avg":
                        self.conv_pool_list.append(sj_layer.AvgPool2d(cl["pool_size"], stride=cl["pool_stride"]))
                    else:
                        self.conv_pool_list.append(sj_layer.MaxPool2d(cl["pool_size"], stride=cl["pool_stride"]))
                else:
                    self.conv_pool_list.append(nn.Identity())
            self.flatten = sj_layer.Flatten()
            fc_in = self.flat_size
        else:
            self.resolved_conv = []
            self.flat_size = self.input_size
            self.flatten = sj_layer.Flatten()
            fc_in = self.input_size

        all_fc_sizes = [fc_in] + self.hidden_layers + [self.output_size]
        self.fcs = nn.ModuleList([
            sj_layer.Linear(all_fc_sizes[i], all_fc_sizes[i + 1]) for i in range(len(all_fc_sizes) - 1)
        ])
        self.lif_layers = nn.ModuleList([create_neuron() for _ in range(len(all_fc_sizes) - 1)])

        if HAS_SPIKINGJELLY:
            functional.set_step_mode(self, 'm')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        if self.has_conv:
            if x.dim() == 2:
                x_4d = x.view(batch_size, self.in_channels, self.in_height, self.in_width)
            elif x.dim() == 3:
                x_4d = x.unsqueeze(1)
            else:
                x_4d = x
            x_seq = encode_input(x_4d, self.encoding_type, self.num_steps)
            cur = x_seq
            for conv, lif, pool in zip(self.conv_layers_list, self.conv_lif_list, self.conv_pool_list):
                cur = conv(cur)
                cur = lif(cur)
                cur = pool(cur)
            cur = self.flatten(cur)
        else:
            x_flat = x.view(batch_size, -1)
            x_seq = encode_input(x_flat, self.encoding_type, self.num_steps)
            cur = x_seq

        for i, (fc, lif) in enumerate(zip(self.fcs, self.lif_layers)):
            cur = fc(cur)
            cur = lif(cur)
            if self.dropouts is not None:
                cur = self.dropouts[i](cur)

        out_spikes = cur.sum(dim=0)
        if HAS_SPIKINGJELLY:
            functional.reset_net(self)
        return out_spikes

    def simulate_event_based(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Simuliert die SNN-Inferenz ereignisgesteuert (Event-by-Event) Zeitschritt für Zeitschritt
        und ermittelt Spike-Events, synaptische Operationen (SynOps) und Sparsity.
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.size(0)
            if self.has_conv:
                if x.dim() == 2:
                    x_4d = x.view(batch_size, self.in_channels, self.in_height, self.in_width)
                elif x.dim() == 3:
                    x_4d = x.unsqueeze(1)
                else:
                    x_4d = x
                x_seq = encode_input(x_4d, self.encoding_type, self.num_steps)
            else:
                x_flat = x.view(batch_size, -1)
                x_seq = encode_input(x_flat, self.encoding_type, self.num_steps)

            total_spikes = 0
            total_synops = 0
            total_possible_ops = 0

            # Forward über Schritte
            if self.has_conv:
                cur = x_seq
                for conv, lif, pool in zip(self.conv_layers_list, self.conv_lif_list, self.conv_pool_list):
                    in_spikes_count = (cur != 0).sum().item()
                    k_size = conv.kernel_size[0] * conv.kernel_size[1]
                    out_ch = conv.out_channels
                    total_synops += int(in_spikes_count * k_size * out_ch)
                    total_possible_ops += int(cur.numel() * k_size * out_ch)
                    cur = conv(cur)
                    cur = lif(cur)
                    total_spikes += int((cur != 0).sum().item())
                    cur = pool(cur)
                cur = self.flatten(cur)
            else:
                cur = x_seq

            for fc, lif in zip(self.fcs, self.lif_layers):
                in_spikes_count = (cur != 0).sum().item()
                out_dim = fc.out_features
                total_synops += int(in_spikes_count * out_dim)
                total_possible_ops += int(cur.numel() * out_dim)
                cur = fc(cur)
                cur = lif(cur)
                total_spikes += int((cur != 0).sum().item())

            out_spikes = cur.sum(dim=0)
            if HAS_SPIKINGJELLY:
                functional.reset_net(self)

            op_sparsity = 100.0 * (1.0 - (total_synops / max(1, total_possible_ops)))
            stats = {
                "total_spikes": total_spikes,
                "synops": total_synops,
                "possible_ops": total_possible_ops,
                "synops_sparsity_pct": op_sparsity,
            }
            return out_spikes, stats

    def get_structure_summary(self) -> str:
        parts = []
        if self.has_conv:
            for cl in self.resolved_conv:
                p_str = f",p{cl['pool_size']}" if cl['pool_size'] > 1 else ""
                parts.append(f"Conv({cl['in_channels']}->{cl['out_channels']},{cl['kernel_size']}x{cl['kernel_size']}{p_str})")
            parts.append(f"Flat({self.flat_size})")
        else:
            parts.append(str(self.input_size))

        for h in self.hidden_layers:
            parts.append(str(h))
        parts.append(str(self.output_size))

        n_label = "SNN-IF" if self.neuron_type == "if" else "SNN-LIF"
        enc_label = self.encoding_type.upper()
        if self.neuron_type == "if":
            param_str = f"T={self.num_steps}, Enc={enc_label}, θ={self.threshold:.1f}"
        else:
            param_str = f"T={self.num_steps}, Enc={enc_label}, β={self.beta:.2f}, θ={self.threshold:.1f}"
        return f"SpikingJelly {n_label} [{' -> '.join(parts)}] ({param_str})"


# -------------------------------------------------------------
# Optimierungs-Konfiguration & Hilfsfunktionen
# -------------------------------------------------------------
def get_model_optimization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrahiert und normalisiert die Optimierungskonfiguration für ein Modell.
    Kombiniert globale 'optimization'-Einstellungen mit modellindividuellen Einstellungen.
    """
    model_cfg = config.get("model", {})
    global_opt = config.get("optimization", {})
    if not isinstance(global_opt, dict):
        global_opt = {}

    model_opt = model_cfg.get("optimization", config.get("optimization", {}))
    if not isinstance(model_opt, dict):
        model_opt = {}

    merged = {
        "pruning": copy.deepcopy(global_opt.get("pruning", {})),
        "quantization": copy.deepcopy(global_opt.get("quantization", {})),
        "clustering": copy.deepcopy(global_opt.get("clustering", {})),
    }

    # Modell-spezifische Übersteuerungen
    if "pruning" in model_opt:
        if isinstance(model_opt["pruning"], dict):
            merged["pruning"].update(model_opt["pruning"])
        elif isinstance(model_opt["pruning"], (float, int)):
            merged["pruning"]["enabled"] = True
            merged["pruning"]["amount"] = float(model_opt["pruning"])
        elif isinstance(model_opt["pruning"], bool):
            merged["pruning"]["enabled"] = model_opt["pruning"]

    if "quantization" in model_opt:
        if isinstance(model_opt["quantization"], dict):
            merged["quantization"].update(model_opt["quantization"])
        elif isinstance(model_opt["quantization"], str):
            merged["quantization"]["enabled"] = True
            merged["quantization"]["type"] = model_opt["quantization"]
        elif isinstance(model_opt["quantization"], bool):
            merged["quantization"]["enabled"] = model_opt["quantization"]

    if "clustering" in model_opt:
        if isinstance(model_opt["clustering"], dict):
            merged["clustering"].update(model_opt["clustering"])
        elif isinstance(model_opt["clustering"], int):
            merged["clustering"]["enabled"] = True
            merged["clustering"]["num_clusters"] = model_opt["clustering"]
        elif isinstance(model_opt["clustering"], bool):
            merged["clustering"]["enabled"] = model_opt["clustering"]

    return merged


# -------------------------------------------------------------
# Modell-Signatur & Instanziierung
# -------------------------------------------------------------
def compute_model_signature(config: Dict[str, Any], dataset_info: Any = None) -> Tuple[Dict[str, Any], str]:
    """
    Erstellt eine kanonische Signatur und einen SHA256-Hash aller Architektur-,
    Datensatz-, Trainings- und Optimierungs-Hyperparameter eines Modells zur Erkennung von Änderungen.
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    model_type = str(model_cfg.get("type", config.get("type", "ann"))).lower()

    dataset_name = model_cfg.get("dataset", config.get("dataset", train_cfg.get("dataset", "mnist")))
    if isinstance(dataset_name, dict):
        dataset_name = dataset_name.get("name", "mnist")
    dataset_name = normalize_dataset_name(dataset_name)

    conv_layers = model_cfg.get("conv_layers", [])
    canonical_conv = []
    for cl in conv_layers:
        if isinstance(cl, dict):
            canonical_conv.append({k: cl[k] for k in sorted(cl.keys())})
        else:
            canonical_conv.append(cl)

    hidden_layers = [int(h) for h in model_cfg.get("hidden_layers", [64])]

    es_cfg = train_cfg.get("early_stopping", False)
    if isinstance(es_cfg, dict):
        es_info = {
            "enabled": bool(es_cfg.get("enabled", True)),
            "patience": int(es_cfg.get("patience", train_cfg.get("patience", 3))),
            "min_delta": float(es_cfg.get("min_delta", train_cfg.get("min_delta", 0.0))),
            "monitor": str(es_cfg.get("monitor", train_cfg.get("monitor", "test_acc"))).lower(),
            "restore_best_weights": bool(es_cfg.get("restore_best_weights", train_cfg.get("restore_best_weights", True))),
        }
    else:
        es_info = {
            "enabled": bool(es_cfg),
            "patience": int(train_cfg.get("patience", 3)),
            "min_delta": float(train_cfg.get("min_delta", 0.0)),
            "monitor": str(train_cfg.get("monitor", "test_acc")).lower(),
            "restore_best_weights": bool(train_cfg.get("restore_best_weights", True)),
        }

    opt_cfg = get_model_optimization_config(config)
    prune_cfg = opt_cfg.get("pruning", {})
    quant_cfg = opt_cfg.get("quantization", {})
    clust_cfg = opt_cfg.get("clustering", {})

    canonical_opt = {
        "pruning": {
            "enabled": bool(prune_cfg.get("enabled", False)),
            "method": str(prune_cfg.get("method", "l1_unstructured")).lower(),
            "amount": float(prune_cfg.get("amount", prune_cfg.get("sparsity", 0.0))),
            "threshold": float(prune_cfg.get("threshold", 0.0)),
            "layers": [str(l).lower() for l in prune_cfg.get("layers", ["conv", "fc"])],
            "fine_tune_epochs": int(prune_cfg.get("fine_tune_epochs", 0)),
            "fine_tune_lr": float(prune_cfg.get("fine_tune_lr", 0.0001)),
        },
        "quantization": {
            "enabled": bool(quant_cfg.get("enabled", False)),
            "type": str(quant_cfg.get("type", "int8")).lower(),
            "mode": str(quant_cfg.get("mode", "symmetric")).lower(),
            "per_channel": bool(quant_cfg.get("per_channel", True)),
            "layers": [str(l).lower() for l in quant_cfg.get("layers", ["conv", "fc"])],
        },
        "clustering": {
            "enabled": bool(clust_cfg.get("enabled", False)),
            "num_clusters": int(clust_cfg.get("num_clusters", 16)),
        },
    }

    signature = {
        "name": str(config.get("name", model_cfg.get("name", "model"))),
        "type": model_type,
        "dataset": dataset_name,
        "conv_layers": canonical_conv,
        "hidden_layers": hidden_layers,
        "optimization": canonical_opt,
        "training": {
            "epochs": int(train_cfg.get("epochs", 3)),
            "batch_size": int(train_cfg.get("batch_size", 64)),
            "learning_rate": float(train_cfg.get("learning_rate", 0.001)),
            "optimizer": str(train_cfg.get("optimizer", "adam")).lower(),
            "early_stopping": es_info,
            "random_seed": train_cfg.get("random_seed", 42),
        },
    }

    if dataset_info is not None:
        signature["dataset_dims"] = {
            "in_channels": int(dataset_info.in_channels),
            "in_height": int(dataset_info.in_height),
            "in_width": int(dataset_info.in_width),
            "input_size": int(dataset_info.input_size),
            "num_classes": int(dataset_info.num_classes),
        }

    if model_type == "snn":
        neuron_type = str(model_cfg.get("neuron_type", model_cfg.get("neuron", "lif"))).lower()
        exec_mode = str(model_cfg.get("execution_mode", model_cfg.get("execution", "dense"))).lower()
        if exec_mode in ("event", "event_based", "event_driven"):
            exec_mode = "event_based"
        else:
            exec_mode = "dense"
        signature["snn_params"] = {
            "neuron_type": neuron_type,
            "encoding_type": str(model_cfg.get("encoding_type", model_cfg.get("encoding", "direct"))).lower(),
            "execution_mode": exec_mode,
            "num_steps": int(model_cfg.get("num_steps", model_cfg.get("steps", 10))),
            "beta": float(model_cfg.get("beta", 0.9)) if neuron_type == "lif" else 1.0,
            "threshold": float(model_cfg.get("threshold", 1.0)),
        }
    else:
        signature["ann_params"] = {
            "activation": str(model_cfg.get("activation", "relu")).lower(),
        }

    signature["dropout_rate"] = float(model_cfg.get("dropout_rate", model_cfg.get("dropout", 0.0)))

    sig_json = json.dumps(signature, sort_keys=True)
    sig_hash = hashlib.sha256(sig_json.encode("utf-8")).hexdigest()
    return signature, sig_hash


def transfer_ann_to_snn(
    ann_model: ConfigurableANN,
    snn_model: ConfigurableSpikingJellySNN,
) -> bool:
    """
    Übertragt die Gewichte eines ANN auf ein SNN mit identischer Architektur.
    Die Conv- und FC-Gewichte sowie Bias-Werte werden kopiert.
    """
    try:
        # Conv Layers: ANN conv_modules[i][0] -> SNN conv_layers_list[i]
        if ann_model.has_conv and snn_model.has_conv:
            for i in range(len(ann_model.conv_modules)):
                ann_conv = ann_model.conv_modules[i][0]
                snn_conv = snn_model.conv_layers_list[i]
                if ann_conv.weight.shape == snn_conv.weight.shape:
                    snn_conv.weight.data.copy_(ann_conv.weight.data)
                    if ann_conv.bias is not None and snn_conv.bias is not None:
                        snn_conv.bias.data.copy_(ann_conv.bias.data)
                else:
                    print(f"[Transfer] Warnung: Conv-Layer {i} Form unvereinbar: ANN {ann_conv.weight.shape} vs SNN {snn_conv.weight.shape}")
                    return False

        # FC Layers: ANN fc_layers[i] -> SNN fcs[i]
        for i in range(len(ann_model.fc_layers)):
            ann_fc = ann_model.fc_layers[i]
            snn_fc = snn_model.fcs[i]
            if ann_fc.weight.shape == snn_fc.weight.shape:
                snn_fc.weight.data.copy_(ann_fc.weight.data)
                if ann_fc.bias is not None and snn_fc.bias is not None:
                    snn_fc.bias.data.copy_(ann_fc.bias.data)
            else:
                print(f"[Transfer] Warnung: FC-Layer {i} Form unvereinbar: ANN {ann_fc.weight.shape} vs SNN {snn_fc.weight.shape}")
                return False

        print("[Transfer] Alle Gewichte erfolgreich von ANN auf SNN übertragen.")
        return True
    except Exception as e:
        print(f"[Transfer] Fehler: {e}")
        return False


def build_model_from_config(
    config: Dict[str, Any],
    dataset_info: DatasetInfo,
    device: str = "cpu",
) -> Union[ConfigurableANN, ConfigurableSpikingJellySNN]:
    """
    Erzeugt ein ConfigurableANN- oder ConfigurableSpikingJellySNN-Modell anhand der Konfiguration.
    """
    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", config.get("type", "ann"))).lower()

    in_channels = int(dataset_info.in_channels)
    in_height = int(dataset_info.in_height)
    in_width = int(dataset_info.in_width)
    input_size = int(dataset_info.input_size)
    output_size = int(dataset_info.num_classes)

    conv_layers = model_cfg.get("conv_layers", [])
    hidden_layers = model_cfg.get("hidden_layers", [64])

    if model_type == "snn":
        neuron_type = str(model_cfg.get("neuron_type", model_cfg.get("neuron", "lif"))).lower()
        encoding_type = str(model_cfg.get("encoding_type", model_cfg.get("encoding", "direct"))).lower()
        execution_mode = str(model_cfg.get("execution_mode", model_cfg.get("execution", "dense"))).lower()
        num_steps = int(model_cfg.get("num_steps", model_cfg.get("steps", 10)))
        beta = float(model_cfg.get("beta", 0.9)) if neuron_type == "lif" else 1.0
        threshold = float(model_cfg.get("threshold", 1.0))
        dropout_rate = float(model_cfg.get("dropout_rate", model_cfg.get("dropout", 0.0)))
        model = ConfigurableSpikingJellySNN(
            input_size=input_size,
            conv_layers=conv_layers,
            hidden_layers=hidden_layers,
            output_size=output_size,
            neuron_type=neuron_type,
            encoding_type=encoding_type,
            execution_mode=execution_mode,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            in_channels=in_channels,
            in_height=in_height,
            in_width=in_width,
            dropout_rate=dropout_rate,
        ).to(device)
    else:
        activation = model_cfg.get("activation", "relu")
        dropout_rate = float(model_cfg.get("dropout_rate", model_cfg.get("dropout", 0.0)))
        model = ConfigurableANN(
            input_size=input_size,
            conv_layers=conv_layers,
            hidden_layers=hidden_layers,
            activation=activation,
            output_size=output_size,
            in_channels=in_channels,
            in_height=in_height,
            in_width=in_width,
            dropout_rate=dropout_rate,
        ).to(device)

    return model


# -------------------------------------------------------------
# Post-Training Optimierung (Pruning, Quantisierung, Clustering)
# -------------------------------------------------------------
def compute_model_sparsity(model: nn.Module) -> Dict[str, Any]:
    """
    Berechnet detaillierte Sparsity-Metriken (Nullstellen in Gewichten) für alle Schichten.
    """
    total_weights = 0
    zero_weights = 0
    layer_stats = []

    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                cnt = w.numel()
                zeros = int((w == 0).sum().item())
                total_weights += cnt
                zero_weights += zeros
                layer_stats.append({
                    "name": name or module.__class__.__name__,
                    "shape": list(w.shape),
                    "total": cnt,
                    "zeros": zeros,
                    "sparsity_pct": (zeros / cnt * 100.0) if cnt > 0 else 0.0,
                })

    sparsity_pct = (zero_weights / total_weights * 100.0) if total_weights > 0 else 0.0
    return {
        "total_weights": total_weights,
        "zero_weights": zero_weights,
        "non_zero_weights": total_weights - zero_weights,
        "sparsity_pct": sparsity_pct,
        "layer_stats": layer_stats,
    }


def evaluate_model(
    model: nn.Module,
    test_dataset: Dataset,
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Evaluiert ein Modell auf dem gesamten Testdatensatz und gibt (test_acc, test_loss) zurück.
    """
    model.eval()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if HAS_SPIKINGJELLY and getattr(model, "neuron_type", None) is not None:
                functional.reset_net(model)
            total_loss += criterion(output, target).item() * len(target)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

    avg_loss = total_loss / max(1, total)
    acc = (100.0 * correct) / max(1, total)
    return acc, avg_loss


def apply_pruning(
    model: nn.Module,
    prune_cfg: Dict[str, Any],
    train_dataset: Optional[Dataset] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Wendet Post-Training Pruning (L1-/Magnitude-Unstructured, Structured, Random oder Threshold)
    auf Schichten (Conv und/oder FC) eines Modells an.
    """
    enabled = bool(prune_cfg.get("enabled", True))
    if not enabled:
        return {"applied": False}

    method = str(prune_cfg.get("method", "l1_unstructured")).lower().strip()
    amount = float(prune_cfg.get("amount", prune_cfg.get("sparsity", 0.3)))
    threshold = float(prune_cfg.get("threshold", 0.01))
    layer_targets = prune_cfg.get("layers", ["conv", "fc"])
    if isinstance(layer_targets, str):
        layer_targets = [layer_targets.lower()]
    else:
        layer_targets = [str(l).lower() for l in layer_targets]

    pruned_modules = []

    for name, module in model.named_modules():
        is_conv = isinstance(module, (nn.Conv2d, sj_layer.Conv2d if HAS_SPIKINGJELLY else nn.Conv2d))
        is_linear = isinstance(module, (nn.Linear, sj_layer.Linear if HAS_SPIKINGJELLY else nn.Linear))

        target_match = False
        if ("all" in layer_targets) or ("conv" in layer_targets and is_conv) or ("fc" in layer_targets and is_linear) or ("linear" in layer_targets and is_linear):
            target_match = True

        if target_match and hasattr(module, "weight") and module.weight is not None:
            mod_label = name or module.__class__.__name__
            if method in ("l1_unstructured", "magnitude", "l1"):
                torch_prune.l1_unstructured(module, name="weight", amount=amount)
                torch_prune.remove(module, "weight")
                pruned_modules.append(mod_label)
            elif method in ("random", "random_unstructured"):
                torch_prune.random_unstructured(module, name="weight", amount=amount)
                torch_prune.remove(module, "weight")
                pruned_modules.append(mod_label)
            elif method in ("structured", "ln_structured"):
                dim = 0 if is_linear or is_conv else 0
                torch_prune.ln_structured(module, name="weight", amount=amount, n=1, dim=dim)
                torch_prune.remove(module, "weight")
                pruned_modules.append(mod_label)
            elif method in ("threshold", "value_threshold"):
                with torch.no_grad():
                    mask = (module.weight.data.abs() >= threshold).float()
                    module.weight.data.mul_(mask)
                pruned_modules.append(mod_label)

    fine_tune_epochs = int(prune_cfg.get("fine_tune_epochs", 0))
    fine_tune_lr = float(prune_cfg.get("fine_tune_lr", 1e-4))

    if fine_tune_epochs > 0 and train_dataset is not None:
        zero_masks = {}
        for _, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                zero_masks[module] = (module.weight.data != 0).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=fine_tune_lr)
        loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        model.train()
        for ep in range(fine_tune_epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_x)
                if HAS_SPIKINGJELLY and getattr(model, "neuron_type", None) is not None:
                    functional.reset_net(model)
                loss = F.cross_entropy(out, batch_y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for mod, mask in zero_masks.items():
                        mod.weight.data.mul_(mask)

    return {
        "applied": True,
        "method": method,
        "amount": amount if method != "threshold" else threshold,
        "pruned_modules": pruned_modules,
        "fine_tune_epochs": fine_tune_epochs,
    }


def apply_quantization(
    model: nn.Module,
    quant_cfg: Dict[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Wendet Post-Training Quantisierung (INT8, INT4, FP16) auf die Gewichte an.
    """
    enabled = bool(quant_cfg.get("enabled", True))
    if not enabled:
        return {"applied": False}

    qtype = str(quant_cfg.get("type", "int8")).lower().strip()
    mode = str(quant_cfg.get("mode", "symmetric")).lower().strip()
    per_channel = bool(quant_cfg.get("per_channel", True))
    layer_targets = quant_cfg.get("layers", ["conv", "fc"])
    if isinstance(layer_targets, str):
        layer_targets = [layer_targets.lower()]
    else:
        layer_targets = [str(l).lower() for l in layer_targets]

    quantized_modules = []

    with torch.no_grad():
        for name, module in model.named_modules():
            is_conv = isinstance(module, (nn.Conv2d, sj_layer.Conv2d if HAS_SPIKINGJELLY else nn.Conv2d))
            is_linear = isinstance(module, (nn.Linear, sj_layer.Linear if HAS_SPIKINGJELLY else nn.Linear))

            target_match = False
            if ("all" in layer_targets) or ("conv" in layer_targets and is_conv) or ("fc" in layer_targets and is_linear) or ("linear" in layer_targets and is_linear):
                target_match = True

            if target_match and hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                mod_label = name or module.__class__.__name__

                if qtype in ("int8", "qint8", "8bit", "dynamic"):
                    q_min, q_max = -128, 127
                    if per_channel and w.dim() >= 2:
                        out_channels = w.shape[0]
                        w_reshaped = w.view(out_channels, -1)
                        if mode == "asymmetric":
                            w_min = w_reshaped.min(dim=1, keepdim=True)[0]
                            w_max = w_reshaped.max(dim=1, keepdim=True)[0]
                            scale = torch.clamp((w_max - w_min) / 255.0, min=1e-8)
                            zero_point = torch.round(-w_min / scale) - 128
                            w_q = torch.clamp(torch.round(w_reshaped / scale) + zero_point, q_min, q_max)
                            w_deq = (w_q - zero_point) * scale
                        else:
                            max_val = torch.clamp(w_reshaped.abs().max(dim=1, keepdim=True)[0], min=1e-8)
                            scale = max_val / 127.0
                            w_q = torch.clamp(torch.round(w_reshaped / scale), q_min, q_max)
                            w_deq = w_q * scale
                        module.weight.data.copy_(w_deq.view_as(w))
                    else:
                        if mode == "asymmetric":
                            w_min = w.min()
                            w_max = w.max()
                            scale = torch.clamp((w_max - w_min) / 255.0, min=1e-8)
                            zero_point = torch.round(-w_min / scale) - 128
                            w_q = torch.clamp(torch.round(w / scale) + zero_point, q_min, q_max)
                            w_deq = (w_q - zero_point) * scale
                        else:
                            max_val = torch.clamp(w.abs().max(), min=1e-8)
                            scale = max_val / 127.0
                            w_q = torch.clamp(torch.round(w / scale), q_min, q_max)
                            w_deq = w_q * scale
                        module.weight.data.copy_(w_deq)
                    quantized_modules.append(mod_label)

                elif qtype in ("int4", "qint4", "4bit"):
                    q_min, q_max = -8, 7
                    if per_channel and w.dim() >= 2:
                        out_channels = w.shape[0]
                        w_reshaped = w.view(out_channels, -1)
                        max_val = torch.clamp(w_reshaped.abs().max(dim=1, keepdim=True)[0], min=1e-8)
                        scale = max_val / 7.0
                        w_q = torch.clamp(torch.round(w_reshaped / scale), q_min, q_max)
                        w_deq = w_q * scale
                        module.weight.data.copy_(w_deq.view_as(w))
                    else:
                        max_val = torch.clamp(w.abs().max(), min=1e-8)
                        scale = max_val / 7.0
                        w_q = torch.clamp(torch.round(w / scale), q_min, q_max)
                        w_deq = w_q * scale
                        module.weight.data.copy_(w_deq)
                    quantized_modules.append(mod_label)

                elif qtype in ("fp16", "float16", "half"):
                    module.weight.data.copy_(w.half().float())
                    quantized_modules.append(mod_label)

    return {
        "applied": True,
        "type": qtype,
        "mode": mode,
        "per_channel": per_channel,
        "quantized_modules": quantized_modules,
    }


def apply_weight_clustering(
    model: nn.Module,
    cluster_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Clustert Modellgewichte in k diskrete Centroids (Weight-Sharing / Codebook).
    """
    enabled = bool(cluster_cfg.get("enabled", True))
    if not enabled:
        return {"applied": False}

    num_clusters = int(cluster_cfg.get("num_clusters", 16))
    clustered_modules = []

    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                w = module.weight.data
                w_np = w.cpu().numpy().flatten()
                non_zeros = w_np[w_np != 0.0]
                if len(non_zeros) > num_clusters:
                    quantiles = np.linspace(0, 100, num_clusters)
                    centroids = np.percentile(non_zeros, quantiles)
                    diffs = np.abs(w_np[:, None] - centroids[None, :])
                    nearest_idx = np.argmin(diffs, axis=1)
                    clustered_np = centroids[nearest_idx]
                    clustered_np[w_np == 0.0] = 0.0
                    module.weight.data.copy_(torch.from_numpy(clustered_np.reshape(w.shape)).to(w.device, w.dtype))
                    clustered_modules.append(name or module.__class__.__name__)

    return {
        "applied": True,
        "num_clusters": num_clusters,
        "clustered_modules": clustered_modules,
    }


def optimize_model(
    model: Union[ConfigurableANN, ConfigurableSpikingJellySNN],
    config: Dict[str, Any],
    test_dataset: Optional[Dataset] = None,
    train_dataset: Optional[Dataset] = None,
    device: str = "cpu",
) -> Tuple[Union[ConfigurableANN, ConfigurableSpikingJellySNN], Dict[str, Any]]:
    """
    Führt konfigurierte Post-Training-Optimierungen (Pruning, Quantisierung, Weight Clustering)
    auf einem trainierten ANN- oder SNN-Modell durch und misst die resultierende Sparsity & Genauigkeit.
    """
    opt_cfg = get_model_optimization_config(config)
    prune_cfg = opt_cfg.get("pruning", {})
    quant_cfg = opt_cfg.get("quantization", {})
    clust_cfg = opt_cfg.get("clustering", {})

    prune_enabled = bool(prune_cfg.get("enabled", False))
    quant_enabled = bool(quant_cfg.get("enabled", False))
    clust_enabled = bool(clust_cfg.get("enabled", False))

    if not (prune_enabled or quant_enabled or clust_enabled):
        return model, {"applied": False}

    model_name = str(config.get("name", config.get("model", {}).get("name", "Modell")))
    model_type = str(config.get("type", config.get("model", {}).get("type", "ann"))).upper()

    pre_acc, pre_loss = (0.0, 0.0)
    if test_dataset is not None:
        pre_acc, pre_loss = evaluate_model(model, test_dataset, device=device)

    pre_sparsity = compute_model_sparsity(model)

    prune_res = {}
    if prune_enabled:
        prune_res = apply_pruning(model, prune_cfg, train_dataset=train_dataset, device=device)

    clust_res = {}
    if clust_enabled:
        clust_res = apply_weight_clustering(model, clust_cfg)

    quant_res = {}
    if quant_enabled:
        quant_res = apply_quantization(model, quant_cfg, device=device)

    post_sparsity = compute_model_sparsity(model)

    post_acc, post_loss = (pre_acc, pre_loss)
    if test_dataset is not None:
        post_acc, post_loss = evaluate_model(model, test_dataset, device=device)

    acc_diff = post_acc - pre_acc

    # Schätzung des Kompressionsfaktors
    bits_per_param = 32.0
    if quant_enabled:
        qtype = str(quant_cfg.get("type", "int8")).lower()
        if "int4" in qtype or "4bit" in qtype:
            bits_per_param = 4.0
        elif "int8" in qtype or "8bit" in qtype:
            bits_per_param = 8.0
        elif "fp16" in qtype or "half" in qtype:
            bits_per_param = 16.0
    elif clust_enabled:
        bits_per_param = max(4.0, float(np.log2(max(2, int(clust_cfg.get("num_clusters", 16))))))

    compression_ratio = 32.0 / bits_per_param

    print("\n" + "=" * 65)
    print(f">>> [Post-Training Optimierung] Modell: '{model_name}' ({model_type}) <<<")
    print(f"  - Vor Optimierung:     Genauigkeit: {pre_acc:.2f}%, Verlust: {pre_loss:.4f}")
    if prune_enabled:
        print(f"  - Pruning:             {prune_res.get('method', 'l1')} (Ziel: {prune_res.get('amount')}) -> {len(prune_res.get('pruned_modules', []))} Schichten")
    if clust_enabled:
        print(f"  - Weight Clustering:   {clust_res.get('num_clusters')} Centroids -> {len(clust_res.get('clustered_modules', []))} Schichten")
    if quant_enabled:
        print(f"  - Quantisierung:       {quant_res.get('type', 'int8').upper()} ({quant_res.get('mode', 'symmetric')}) -> {len(quant_res.get('quantized_modules', []))} Schichten")
    print(f"  - Nach Optimierung:    Genauigkeit: {post_acc:.2f}% (Delta: {acc_diff:+.2f}%), Verlust: {post_loss:.4f}")
    print(f"  - Modell-Sparsity:     {post_sparsity['sparsity_pct']:.2f}% Nullen ({post_sparsity['zero_weights']:,} von {post_sparsity['total_weights']:,} Gewichten)")
    print(f"  - Geschätzte Kompr.:   ca. {compression_ratio:.1f}x Speicherreduktion")
    print("=" * 65 + "\n")

    summary = {
        "applied": True,
        "pre_acc": pre_acc,
        "pre_loss": pre_loss,
        "post_acc": post_acc,
        "post_loss": post_loss,
        "acc_diff": acc_diff,
        "pre_sparsity": pre_sparsity,
        "post_sparsity": post_sparsity,
        "compression_ratio": compression_ratio,
        "pruning": prune_res,
        "quantization": quant_res,
        "clustering": clust_res,
    }
    return model, summary


# -------------------------------------------------------------
# Training & Caching
# -------------------------------------------------------------
def train_model(
    config: Dict[str, Any],
    device: str = "cpu",
    force_retrain: bool = False,
    save_dir: Optional[str] = None,
) -> Tuple[Union[ConfigurableANN, ConfigurableSpikingJellySNN], Dataset, float, float]:
    """
    Trainiert das konfigurierte PyTorch-Netzwerk (ANN oder SNN mit SpikingJelly) auf dem gewählten Datensatz
    oder lädt ein bereits trainiertes Modell aus dem Cache, wenn die Konfiguration unverändert ist.
    Gibt (model, test_dataset, test_acc, avg_loss) zurück.
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    model_type = str(model_cfg.get("type", config.get("type", "ann"))).lower()
    model_name = str(config.get("name", model_cfg.get("name", "Modell")))

    resolved_save_dir = save_dir or train_cfg.get("save_dir", config.get("save_dir", "./saved_models"))
    auto_save = bool(train_cfg.get("auto_save", config.get("auto_save", True)))
    should_force = force_retrain or bool(train_cfg.get("force_retrain", False))

    epochs = train_cfg.get("epochs", 3)
    batch_size = train_cfg.get("batch_size", 64)
    lr = train_cfg.get("learning_rate", 0.001)
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    data_dir = train_cfg.get("data_dir", "./data")
    random_seed = train_cfg.get("random_seed", 42)

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # Datensatz laden
    dataset_name = model_cfg.get("dataset", config.get("dataset", train_cfg.get("dataset", "mnist")))
    if isinstance(dataset_name, dict):
        dataset_name = dataset_name.get("name", "mnist")
    dataset_name = normalize_dataset_name(dataset_name)

    train_dataset, test_dataset, dataset_info = get_dataset(
        name=dataset_name,
        data_dir=data_dir,
        seed=random_seed if random_seed is not None else 42,
    )

    signature, sig_hash = compute_model_signature(config, dataset_info)
    model = build_model_from_config(config, dataset_info, device=device)

    # Transfer Learning: ANN -> SNN (wenn SNN konfiguriert und ANN-Checkpoint existiert)
    if model_type == "snn":
        ann_model_name = model_cfg.get("ann_checkpoint", f"ANN_{dataset_name.capitalize()}_Conv32_64_128")
        ann_checkpoint_path = os.path.join(resolved_save_dir, f"{ann_model_name}.pt")
        if os.path.isfile(ann_checkpoint_path):
            try:
                ann_checkpoint = torch.load(ann_checkpoint_path, map_location=device, weights_only=False)
                conv_layers = model_cfg.get("conv_layers", [])
                hidden_layers = model_cfg.get("hidden_layers", [64])
                ann_model = ConfigurableANN(
                    input_size=input_size,
                    conv_layers=conv_layers,
                    hidden_layers=hidden_layers,
                    activation="relu",
                    output_size=output_size,
                    in_channels=in_channels,
                    in_height=in_height,
                    in_width=in_width,
                    dropout_rate=0.0,
                ).to(device)
                ann_model.load_state_dict(ann_checkpoint["state_dict"])
                ann_model.eval()
                if transfer_ann_to_snn(ann_model, model):
                    print(f"[Transfer] SNN '{model_name}' initialisiert mit ANN-Gewichten von '{ann_model_name}'\n")
                else:
                    print("[Transfer] Transfer fehlgeschlagen - SNN wird zufällig initialisiert.\n")
            except Exception as e:
                print(f"[Transfer] Fehler beim Laden des ANN-Checkpoints: {e}\n")
        else:
            print(f"[Transfer] Kein ANN-Checkpoint '{ann_checkpoint_path}' gefunden - SNN wird zufällig initialisiert.\n")

    in_channels = int(dataset_info.in_channels)
    in_height = int(dataset_info.in_height)
    in_width = int(dataset_info.in_width)
    input_size = int(dataset_info.input_size)
    output_size = int(dataset_info.num_classes)

    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name)
    checkpoint_path = os.path.join(resolved_save_dir, f"{safe_name}.pt")
    meta_path = os.path.join(resolved_save_dir, f"{safe_name}.json")

    # ---------------------------------------------------------
    # Cache-Prüfung: Bereits trainiertes Modell laden?
    # ---------------------------------------------------------
    if not should_force and auto_save and os.path.isfile(checkpoint_path):
        try:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=device)

            cached_hash = checkpoint.get("config_hash")
            if cached_hash == sig_hash and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                test_acc = float(checkpoint.get("test_acc", 0.0))
                test_loss = float(checkpoint.get("test_loss", 0.0))
                best_epoch = checkpoint.get("best_epoch", "N/A")

                print("\n" + "=" * 65)
                print(f">>> [Cache] Verwende gespeichertes Modell: '{model_name}' ({'SNN' if model_type == 'snn' else 'ANN'}) <<<")
                print(f"  - Checkpoint-Datei:    {checkpoint_path}")
                print(f"  - Datensatz:           {dataset_info.name} ({dataset_info.domain})")
                print(f"  - Netzwerk-Aufbau:     {model.get_structure_summary()}")
                print(f"  - Konfigurations-Hash: {sig_hash[:12]} (unverändert)")
                print(f"  - Gespeicherte Acc:    {test_acc:.2f}% (aus Epoche {best_epoch})")
                print(f"  - Gespeicherter Loss:  {test_loss:.4f}")
                print(f"  -> Kein erneutes Training nötig (Cache-Hit).\n")
                return model, test_dataset, test_acc, test_loss
            else:
                old_h = str(cached_hash)[:12] if cached_hash else "None"
                print(f"\n[Cache] Gespeichertes Modell '{model_name}' gefunden, aber Konfiguration/Hyperparameter geändert:")
                print(f"        Alter Hash: {old_h} -> Neuer Hash: {sig_hash[:12]}")
                print(f"        -> Modell wird neu trainiert.")
        except Exception as e:
            print(f"\n[Cache] Hinweis: Gespeichertes Modell '{checkpoint_path}' konnte nicht geladen werden ({e}).")
            print(f"        -> Modell wird neu trainiert.")

    # Early Stopping Konfiguration
    es_cfg = train_cfg.get("early_stopping", False)
    if isinstance(es_cfg, dict):
        early_stopping_enabled = bool(es_cfg.get("enabled", True))
        patience = int(es_cfg.get("patience", train_cfg.get("patience", 3)))
        min_delta = float(es_cfg.get("min_delta", train_cfg.get("min_delta", 0.0)))
        monitor = str(es_cfg.get("monitor", train_cfg.get("monitor", "test_acc"))).lower()
        restore_best_weights = bool(es_cfg.get("restore_best_weights", train_cfg.get("restore_best_weights", True)))
    else:
        early_stopping_enabled = bool(es_cfg)
        patience = int(train_cfg.get("patience", 3))
        min_delta = float(train_cfg.get("min_delta", 0.0))
        monitor = str(train_cfg.get("monitor", "test_acc")).lower()
        restore_best_weights = bool(train_cfg.get("restore_best_weights", True))

    print("\n" + "=" * 65)
    print(f">>> PyTorch Modell-Konfiguration: {model_name} ({'SNN' if model_type == 'snn' else 'ANN'}) <<<")
    print(f"  - Datensatz:           {dataset_info.name} ({dataset_info.domain})")
    print(f"  - Eingabedimensionen:  {in_channels}x{in_height}x{in_width} ({input_size} Werte) -> {output_size} Klassen")
    print(f"  - Netzwerk-Aufbau:     {model.get_structure_summary()}")
    if getattr(model, "has_conv", False):
        print(f"  - Conv-Schichten:      {len(model.resolved_conv)}")
    print(f"  - Verborgene Schichten:{model.hidden_layers}")
    if model_type == "snn":
        print(f"  - Neuronen-Typ:        {model.neuron_type.upper()}")
        print(f"  - Encoding-Typ:        {model.encoding_type.upper()}")
        print(f"  - Zeitschritte (Steps):{model.num_steps}")
        if model.neuron_type == "lif":
            print(f"  - Membran-Decay (Beta):{model.beta}")
        print(f"  - Schwellenwert (Thresh):{model.threshold}")
    else:
        print(f"  - Aktivierung:         {model.activation_name.upper()}")
    print(f"  - Epochen:             {epochs}")
    print(f"  - Batch-Größe:         {batch_size}")
    print(f"  - Lernrate (LR):       {lr}")
    print(f"  - Optimierer:          {optimizer_name.upper()}")
    if early_stopping_enabled:
        print(f"  - Early Stopping:      Aktiviert (Patience: {patience}, Min-Delta: {min_delta}, Metrik: {monitor}, Best-Weights: {restore_best_weights})")
    else:
        print(f"  - Early Stopping:      Deaktiviert (Best-Weights: {restore_best_weights})")
    print("=" * 65 + "\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer_name = optimizer_name.lower()
    print(f"  - Optimierer:          {optimizer_name.upper()}")
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # Learning rate scheduler: Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

    test_acc = 0.0
    avg_loss = 0.0
    best_state_dict = None
    best_epoch = 1
    patience_counter = 0

    if monitor == "test_loss":
        best_metric_val = float("inf")
    else:
        best_metric_val = -float("inf")
    best_test_acc = 0.0
    best_test_loss = float("inf")

    print(f"[PyTorch] Starte Training über {epochs} Epochen...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        # Validierung nach jeder Epoche
        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100.0 * correct / len(test_dataset)
        avg_loss = running_loss / max(1, total_batches)

        # Early Stopping Überprüfung
        is_best = False
        if monitor == "test_loss":
            current_metric = test_loss
            if current_metric < (best_metric_val - min_delta):
                is_best = True
                best_metric_val = current_metric
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            current_metric = test_acc
            if current_metric > (best_metric_val + min_delta):
                is_best = True
                best_metric_val = current_metric
                patience_counter = 0
            else:
                patience_counter += 1

        if is_best or best_state_dict is None:
            best_epoch = epoch
            best_test_acc = test_acc
            best_test_loss = test_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_tag = " [Bestes Modell]"
        else:
            best_tag = f" [Patience: {patience_counter}/{patience}]" if early_stopping_enabled else ""

        print(f"  Epoche {epoch:02d}/{epochs:02d} | Train-Verlust: {avg_loss:.4f} | Test-Verlust: {test_loss:.4f} | Test-Genauigkeit: {test_acc:.2f}%{best_tag}")

        scheduler.step()

        if early_stopping_enabled and patience_counter >= patience:
            print(f"[Early Stopping] Vorzeitiger Abbruch in Epoche {epoch}/{epochs}: Keine Verbesserung der Metrik '{monitor}' seit {patience} Epochen (Bester Stand in Epoche {best_epoch}: Test-Acc: {best_test_acc:.2f}%, Test-Verlust: {best_test_loss:.4f}).")
            break

    if best_state_dict is not None and restore_best_weights:
        model.load_state_dict(best_state_dict)
        test_acc = best_test_acc
        avg_loss = best_test_loss
        print(f"[PyTorch] Bestes Modell aus Epoche {best_epoch} für Export und Inferenz aktiviert (Test-Genauigkeit: {best_test_acc:.2f}%, Test-Verlust: {best_test_loss:.4f}).")

    print(f"[PyTorch] Training für '{model_name}' abgeschlossen (Beste Test-Genauigkeit: {test_acc:.2f}%).\n")

    # ---------------------------------------------------------
    # Post-Training Optimierung (Pruning, Quantisierung, Clustering)
    # ---------------------------------------------------------
    model, opt_info = optimize_model(
        model=model,
        config=config,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        device=device,
    )
    if opt_info.get("applied", False):
        test_acc = float(opt_info.get("post_acc", test_acc))
        avg_loss = float(opt_info.get("post_loss", avg_loss))

    # ---------------------------------------------------------
    # Automatisches Speichern des trainierten Modells (Checkpoint & Metadaten)
    # ---------------------------------------------------------
    if auto_save and resolved_save_dir:
        try:
            os.makedirs(resolved_save_dir, exist_ok=True)
            checkpoint_data = {
                "model_name": model_name,
                "model_type": model_type,
                "dataset": dataset_name,
                "config_signature": signature,
                "config_hash": sig_hash,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "test_acc": float(test_acc),
                "test_loss": float(avg_loss),
                "best_epoch": best_epoch,
                "optimization": opt_info if opt_info.get("applied", False) else None,
                "timestamp": time.time(),
            }
            torch.save(checkpoint_data, checkpoint_path)

            meta_data = {
                "model_name": model_name,
                "model_type": model_type,
                "dataset": dataset_name,
                "test_acc": float(test_acc),
                "test_loss": float(avg_loss),
                "best_epoch": best_epoch,
                "config_hash": sig_hash,
                "config_signature": signature,
                "optimization": opt_info if opt_info.get("applied", False) else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            print(f"[PyTorch] Modell '{model_name}' erfolgreich in '{checkpoint_path}' gespeichert.\n")
        except Exception as e:
            print(f"[PyTorch] Warnung: Modell konnte nicht gespeichert werden: {e}", file=sys.stderr)

    return model, test_dataset, test_acc, avg_loss


def evaluate_subset(
    model: Union[ConfigurableANN, ConfigurableSpikingJellySNN],
    test_dataset: Any,
    num_samples: int = 50,
    device: str = "cpu",
) -> Tuple[float, int, int]:
    """
    Evaluiert das Modell auf einer definierten Teilmenge des Testdatensatzes.
    """
    model.eval()
    total = min(num_samples, len(test_dataset))
    if total <= 0:
        return 0.0, 0, 0

    correct = 0
    with torch.no_grad():
        for i in range(total):
            img_tensor, label = test_dataset[i]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            out = model(img_tensor)
            pred = out.argmax(dim=1).item()
            if pred == label:
                correct += 1

    acc = (100.0 * correct) / total
    return acc, correct, total


# -------------------------------------------------------------
# Dynamischer TFLite FlatBuffer Generator
# -------------------------------------------------------------
def export_pytorch_to_tflite(model: Union[ConfigurableANN, ConfigurableSpikingJellySNN]) -> bytes:
    """
    Erstellt dynamisch einen standardkonformen TFLite FlatBuffer für ein
    beliebiges Convolutional- oder Fully-Connected-Netzwerk (ANN).
    """
    model.eval()
    bld = flatbuffers.Builder(2 * 1024 * 1024)

    tflite.BufferStart(bld)
    buf0 = tflite.BufferEnd(bld)
    buffers = [buf0]

    def make_buffer(arr: np.ndarray) -> int:
        raw_bytes = arr.tobytes(order="C")
        vec = bld.CreateByteVector(raw_bytes)
        tflite.BufferStart(bld)
        tflite.BufferAddData(bld, vec)
        return tflite.BufferEnd(bld)

    op_codes = []
    op_code_map = {}

    def get_opcode_idx(code):
        if code not in op_code_map:
            tflite.OperatorCodeStart(bld)
            tflite.OperatorCodeAddBuiltinCode(bld, code)
            op = tflite.OperatorCodeEnd(bld)
            op_codes.append(op)
            op_code_map[code] = len(op_codes) - 1
        return op_code_map[code]

    def make_tensor(name: str, shape: List[int], ttype: int, buf_idx: int) -> int:
        tflite.TensorStartShapeVector(bld, len(shape))
        for dim in reversed(shape):
            bld.PrependInt32(dim)
        shape_vec = bld.EndVector()
        name_str = bld.CreateString(name)
        tflite.TensorStart(bld)
        tflite.TensorAddShape(bld, shape_vec)
        tflite.TensorAddType(bld, ttype)
        tflite.TensorAddBuffer(bld, buf_idx)
        tflite.TensorAddName(bld, name_str)
        return tflite.TensorEnd(bld)

    tensors = []
    ops = []

    if getattr(model, "has_conv", False):
        t_input = make_tensor("input", [1, model.in_height, model.in_width, model.in_channels], tflite.TensorType.FLOAT32, 0)
        tensors.append(t_input)
        cur_tensor_idx = 0

        for c_idx, cl in enumerate(model.resolved_conv):
            seq = model.conv_modules[c_idx]
            conv_mod = seq[0]
            # PyTorch: (out_c, in_c, kh, kw) -> TFLite: (out_c, kh, kw, in_c)
            w = conv_mod.weight.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
            b = conv_mod.bias.detach().cpu().numpy().astype(np.float32)
            w_buf = make_buffer(w)
            b_buf = make_buffer(b)
            buffers.extend([w_buf, b_buf])
            w_buf_idx = len(buffers) - 2
            b_buf_idx = len(buffers) - 1

            t_w = make_tensor(f"conv{c_idx+1}_w", [cl["out_channels"], cl["kernel_size"], cl["kernel_size"], cl["in_channels"]], tflite.TensorType.FLOAT32, w_buf_idx)
            t_b = make_tensor(f"conv{c_idx+1}_b", [cl["out_channels"]], tflite.TensorType.FLOAT32, b_buf_idx)
            t_conv_out = make_tensor(f"conv{c_idx+1}_out", [1, cl["out_h"], cl["out_w"], cl["out_channels"]], tflite.TensorType.FLOAT32, 0)

            w_idx = len(tensors)
            b_idx = len(tensors) + 1
            conv_out_idx = len(tensors) + 2
            tensors.extend([t_w, t_b, t_conv_out])

            op_conv_idx = get_opcode_idx(tflite.BuiltinOperator.CONV_2D)
            tflite.Conv2DOptionsStart(bld)
            tflite.Conv2DOptionsAddPadding(bld, tflite.Padding.SAME if cl["padding"] > 0 else tflite.Padding.VALID)
            tflite.Conv2DOptionsAddStrideH(bld, cl["stride"])
            tflite.Conv2DOptionsAddStrideW(bld, cl["stride"])
            act_type = tflite.ActivationFunctionType.RELU if getattr(model, "activation_name", "relu") == "relu" else tflite.ActivationFunctionType.NONE
            tflite.Conv2DOptionsAddFusedActivationFunction(bld, act_type)
            opt_conv = tflite.Conv2DOptionsEnd(bld)

            tflite.OperatorStartInputsVector(bld, 3)
            bld.PrependInt32(b_idx)
            bld.PrependInt32(w_idx)
            bld.PrependInt32(cur_tensor_idx)
            in_conv = bld.EndVector()

            tflite.OperatorStartOutputsVector(bld, 1)
            bld.PrependInt32(conv_out_idx)
            out_conv = bld.EndVector()

            tflite.OperatorStart(bld)
            tflite.OperatorAddOpcodeIndex(bld, op_conv_idx)
            tflite.OperatorAddInputs(bld, in_conv)
            tflite.OperatorAddOutputs(bld, out_conv)
            tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.Conv2DOptions)
            tflite.OperatorAddBuiltinOptions(bld, opt_conv)
            ops.append(tflite.OperatorEnd(bld))
            cur_tensor_idx = conv_out_idx

            if cl["pool_size"] > 1:
                t_pool_out = make_tensor(f"pool{c_idx+1}_out", [1, cl["pool_out_h"], cl["pool_out_w"], cl["out_channels"]], tflite.TensorType.FLOAT32, 0)
                pool_out_idx = len(tensors)
                tensors.append(t_pool_out)

                pool_op_code = tflite.BuiltinOperator.MAX_POOL_2D if cl["pool_type"] == "max" else tflite.BuiltinOperator.AVERAGE_POOL_2D
                op_pool_idx = get_opcode_idx(pool_op_code)
                tflite.Pool2DOptionsStart(bld)
                tflite.Pool2DOptionsAddPadding(bld, tflite.Padding.VALID)
                tflite.Pool2DOptionsAddStrideH(bld, cl["pool_stride"])
                tflite.Pool2DOptionsAddStrideW(bld, cl["pool_stride"])
                tflite.Pool2DOptionsAddFilterHeight(bld, cl["pool_size"])
                tflite.Pool2DOptionsAddFilterWidth(bld, cl["pool_size"])
                tflite.Pool2DOptionsAddFusedActivationFunction(bld, tflite.ActivationFunctionType.NONE)
                opt_pool = tflite.Pool2DOptionsEnd(bld)

                tflite.OperatorStartInputsVector(bld, 1)
                bld.PrependInt32(cur_tensor_idx)
                in_pool = bld.EndVector()

                tflite.OperatorStartOutputsVector(bld, 1)
                bld.PrependInt32(pool_out_idx)
                out_pool = bld.EndVector()

                tflite.OperatorStart(bld)
                tflite.OperatorAddOpcodeIndex(bld, op_pool_idx)
                tflite.OperatorAddInputs(bld, in_pool)
                tflite.OperatorAddOutputs(bld, out_pool)
                tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.Pool2DOptions)
                tflite.OperatorAddBuiltinOptions(bld, opt_pool)
                ops.append(tflite.OperatorEnd(bld))
                cur_tensor_idx = pool_out_idx

        # Reshape / Flatten
        shape_arr = np.array([1, model.flat_size], dtype=np.int32)
        buf_shape = make_buffer(shape_arr)
        buffers.append(buf_shape)
        shape_buf_idx = len(buffers) - 1

        t_shape = make_tensor("flat_shape", [2], tflite.TensorType.INT32, shape_buf_idx)
        t_flat = make_tensor("flat_out", [1, model.flat_size], tflite.TensorType.FLOAT32, 0)
        shape_tensor_idx = len(tensors)
        flat_tensor_idx = len(tensors) + 1
        tensors.extend([t_shape, t_flat])

        op_reshape_idx = get_opcode_idx(tflite.BuiltinOperator.RESHAPE)
        tflite.ReshapeOptionsStartNewShapeVector(bld, 2)
        bld.PrependInt32(model.flat_size)
        bld.PrependInt32(1)
        new_shape_vec = bld.EndVector()
        tflite.ReshapeOptionsStart(bld)
        tflite.ReshapeOptionsAddNewShape(bld, new_shape_vec)
        opt_reshape = tflite.ReshapeOptionsEnd(bld)

        tflite.OperatorStartInputsVector(bld, 2)
        bld.PrependInt32(shape_tensor_idx)
        bld.PrependInt32(cur_tensor_idx)
        in_reshape = bld.EndVector()

        tflite.OperatorStartOutputsVector(bld, 1)
        bld.PrependInt32(flat_tensor_idx)
        out_reshape = bld.EndVector()

        tflite.OperatorStart(bld)
        tflite.OperatorAddOpcodeIndex(bld, op_reshape_idx)
        tflite.OperatorAddInputs(bld, in_reshape)
        tflite.OperatorAddOutputs(bld, out_reshape)
        tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.ReshapeOptions)
        tflite.OperatorAddBuiltinOptions(bld, opt_reshape)
        ops.append(tflite.OperatorEnd(bld))
        cur_tensor_idx = flat_tensor_idx

        # FC layers
        op_fc_idx = get_opcode_idx(tflite.BuiltinOperator.FULLY_CONNECTED)
        num_fc = len(model.fc_layers)
        for f_idx, fc in enumerate(model.fc_layers):
            is_last = (f_idx == num_fc - 1)
            use_relu = (not is_last) and (getattr(model, "activation_name", "relu") == "relu")

            w = fc.weight.detach().cpu().numpy().astype(np.float32)
            b = fc.bias.detach().cpu().numpy().astype(np.float32)
            w_buf = make_buffer(w)
            b_buf = make_buffer(b)
            buffers.extend([w_buf, b_buf])
            w_buf_idx = len(buffers) - 2
            b_buf_idx = len(buffers) - 1

            t_w = make_tensor(f"fc{f_idx+1}_w", [fc.out_features, fc.in_features], tflite.TensorType.FLOAT32, w_buf_idx)
            t_b = make_tensor(f"fc{f_idx+1}_b", [fc.out_features], tflite.TensorType.FLOAT32, b_buf_idx)
            out_name = "output" if is_last else f"fc{f_idx+1}_out"
            t_out = make_tensor(out_name, [1, fc.out_features], tflite.TensorType.FLOAT32, 0)

            w_idx = len(tensors)
            b_idx = len(tensors) + 1
            out_idx = len(tensors) + 2
            tensors.extend([t_w, t_b, t_out])

            tflite.FullyConnectedOptionsStart(bld)
            tflite.FullyConnectedOptionsAddFusedActivationFunction(bld, tflite.ActivationFunctionType.RELU if use_relu else tflite.ActivationFunctionType.NONE)
            opt_fc = tflite.FullyConnectedOptionsEnd(bld)

            tflite.OperatorStartInputsVector(bld, 3)
            bld.PrependInt32(b_idx)
            bld.PrependInt32(w_idx)
            bld.PrependInt32(cur_tensor_idx)
            in_fc = bld.EndVector()

            tflite.OperatorStartOutputsVector(bld, 1)
            bld.PrependInt32(out_idx)
            out_fc = bld.EndVector()

            tflite.OperatorStart(bld)
            tflite.OperatorAddOpcodeIndex(bld, op_fc_idx)
            tflite.OperatorAddInputs(bld, in_fc)
            tflite.OperatorAddOutputs(bld, out_fc)
            tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.FullyConnectedOptions)
            tflite.OperatorAddBuiltinOptions(bld, opt_fc)
            ops.append(tflite.OperatorEnd(bld))
            cur_tensor_idx = out_idx

    else:
        # Pure FC Model
        layers = list(model.fc_layers) if hasattr(model, "fc_layers") else list(model.layers)
        num_layers = len(layers)
        op_fc_idx = get_opcode_idx(tflite.BuiltinOperator.FULLY_CONNECTED)

        t_input = make_tensor("input", [1, model.input_size], tflite.TensorType.FLOAT32, 0)
        tensors.append(t_input)
        cur_tensor_idx = 0

        for i, layer in enumerate(layers):
            is_last = (i == num_layers - 1)
            use_relu = (not is_last) and (getattr(model, "activation_name", "relu") == "relu")

            w = layer.weight.detach().cpu().numpy().astype(np.float32)
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            w_buf = make_buffer(w)
            b_buf = make_buffer(b)
            buffers.extend([w_buf, b_buf])
            w_buf_idx = len(buffers) - 2
            b_buf_idx = len(buffers) - 1

            t_w = make_tensor(f"fc{i+1}_w", [layer.out_features, layer.in_features], tflite.TensorType.FLOAT32, w_buf_idx)
            t_b = make_tensor(f"fc{i+1}_b", [layer.out_features], tflite.TensorType.FLOAT32, b_buf_idx)
            out_name = "output" if is_last else f"fc{i+1}_out"
            t_out = make_tensor(out_name, [1, layer.out_features], tflite.TensorType.FLOAT32, 0)

            w_idx = len(tensors)
            b_idx = len(tensors) + 1
            out_idx = len(tensors) + 2
            tensors.extend([t_w, t_b, t_out])

            tflite.FullyConnectedOptionsStart(bld)
            tflite.FullyConnectedOptionsAddFusedActivationFunction(bld, tflite.ActivationFunctionType.RELU if use_relu else tflite.ActivationFunctionType.NONE)
            opt_fc = tflite.FullyConnectedOptionsEnd(bld)

            tflite.OperatorStartInputsVector(bld, 3)
            bld.PrependInt32(b_idx)
            bld.PrependInt32(w_idx)
            bld.PrependInt32(cur_tensor_idx)
            in_fc = bld.EndVector()

            tflite.OperatorStartOutputsVector(bld, 1)
            bld.PrependInt32(out_idx)
            out_fc = bld.EndVector()

            tflite.OperatorStart(bld)
            tflite.OperatorAddOpcodeIndex(bld, op_fc_idx)
            tflite.OperatorAddInputs(bld, in_fc)
            tflite.OperatorAddOutputs(bld, out_fc)
            tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.FullyConnectedOptions)
            tflite.OperatorAddBuiltinOptions(bld, opt_fc)
            ops.append(tflite.OperatorEnd(bld))
            cur_tensor_idx = out_idx

    final_output_idx = cur_tensor_idx

    # Subgraph
    tflite.SubGraphStartInputsVector(bld, 1)
    bld.PrependInt32(0)
    sub_in = bld.EndVector()

    tflite.SubGraphStartOutputsVector(bld, 1)
    bld.PrependInt32(final_output_idx)
    sub_out = bld.EndVector()

    tflite.SubGraphStartTensorsVector(bld, len(tensors))
    for t in reversed(tensors):
        bld.PrependUOffsetTRelative(t)
    sub_tensors = bld.EndVector()

    tflite.SubGraphStartOperatorsVector(bld, len(ops))
    for op in reversed(ops):
        bld.PrependUOffsetTRelative(op)
    sub_ops = bld.EndVector()

    tflite.SubGraphStart(bld)
    tflite.SubGraphAddTensors(bld, sub_tensors)
    tflite.SubGraphAddInputs(bld, sub_in)
    tflite.SubGraphAddOutputs(bld, sub_out)
    tflite.SubGraphAddOperators(bld, sub_ops)
    subgraph = tflite.SubGraphEnd(bld)

    # Model
    tflite.ModelStartOperatorCodesVector(bld, len(op_codes))
    for oc in reversed(op_codes):
        bld.PrependUOffsetTRelative(oc)
    opcodes = bld.EndVector()

    tflite.ModelStartSubgraphsVector(bld, 1)
    bld.PrependUOffsetTRelative(subgraph)
    subgraphs = bld.EndVector()

    tflite.ModelStartBuffersVector(bld, len(buffers))
    for b in reversed(buffers):
        bld.PrependUOffsetTRelative(b)
    bufs = bld.EndVector()

    desc = bld.CreateString(f"mnist_{model.get_structure_summary()}")

    tflite.ModelStart(bld)
    tflite.ModelAddVersion(bld, 3)
    tflite.ModelAddOperatorCodes(bld, opcodes)
    tflite.ModelAddSubgraphs(bld, subgraphs)
    tflite.ModelAddBuffers(bld, bufs)
    tflite.ModelAddDescription(bld, desc)
    model_obj = tflite.ModelEnd(bld)

    bld.Finish(model_obj, b"TFL3")
    return bytes(bld.Output())


# -------------------------------------------------------------
# C/C++ Code Emitter für ESP32 (ANN & SNN)
# -------------------------------------------------------------
def export_c_files(
    model_entry: Dict[str, Any],
    output_dir: str = "main",
):
    """
    Erzeugt model_data.h/cc und mnist_test_data.h im Zielverzeichnis für den ESP32.
    Unterstützt sowohl ANN (TFLM FlatBuffer) als auch SNN (LIF Parametrierung & Gewichte).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config = model_entry["config"]
    model = model_entry["model"]
    test_dataset = model_entry["test_dataset"]
    model_type = model_entry.get("type", "ann").lower()
    model_name = model_entry.get("name", "Modell")

    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})

    input_size = int(model_cfg.get("input_size", 784))
    output_size = int(model_cfg.get("output_size", 10))
    num_test_samples = int(eval_cfg.get("num_test_samples", 500))
    if test_dataset is not None and hasattr(test_dataset, "__len__"):
        num_test_samples = min(num_test_samples, len(test_dataset))

    # 1. model_data.h
    header_path = out_path / "model_data.h"
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stddef.h>\n\n")
        f.write("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n")
        f.write("#define MODEL_TYPE_ANN 0\n")
        f.write("#define MODEL_TYPE_SNN 1\n\n")
        f.write("#define NEURON_TYPE_LIF 0\n")
        f.write("#define NEURON_TYPE_IF  1\n\n")
        f.write("#define ENCODING_DIRECT    0\n")
        f.write("#define ENCODING_TTFS      1\n")
        f.write("#define ENCODING_POISSON   2\n")
        f.write("#define ENCODING_PERIODIC  3\n\n")
        f.write("#define EXECUTION_MODE_DENSE       0\n")
        f.write("#define EXECUTION_MODE_EVENT_BASED 1\n\n")
        f.write("#define SNN_MAX_FC_LAYERS   8\n")
        f.write("#define SNN_MAX_CONV_LAYERS 4\n\n")
        f.write("typedef struct {\n")
        f.write("    int in_channels;\n")
        f.write("    int out_channels;\n")
        f.write("    int kernel_size;\n")
        f.write("    int stride;\n")
        f.write("    int padding;\n")
        f.write("    int pool_size;\n")
        f.write("    int pool_stride;\n")
        f.write("    int in_h, in_w;\n")
        f.write("    int out_h, out_w;\n")
        f.write("    int pool_out_h, pool_out_w;\n")
        f.write("} snn_conv_layer_cfg_t;\n\n")
        f.write("typedef struct {\n")
        f.write("    int input_size;\n")
        f.write("    int num_conv_layers;\n")
        f.write("    snn_conv_layer_cfg_t conv_layers[SNN_MAX_CONV_LAYERS];\n")
        f.write("    int flat_size;\n")
        f.write("    int hidden_layers_count;\n")
        f.write("    int fc_layer_sizes[SNN_MAX_FC_LAYERS + 1];\n")
        f.write("    int num_steps;\n")
        f.write("    float beta;\n")
        f.write("    float threshold;\n")
        f.write("    int output_size;\n")
        f.write("    int neuron_type;\n")
        f.write("    int encoding_type;\n")
        f.write("    int execution_mode;\n")
        f.write("} snn_config_t;\n\n")
        f.write("extern const int g_model_type;\n")
        f.write("extern const char* const g_model_name;\n\n")
        f.write("// ANN-Spezifisch (TensorFlow Lite Micro FlatBuffer)\n")
        f.write("extern const unsigned char g_model[];\n")
        f.write("extern const int g_model_len;\n\n")
        f.write("// SNN-Spezifisch (LIF/IF Neuronen Parameter & Gewichte)\n")
        f.write("extern const snn_config_t g_snn_config;\n")
        f.write("extern const float* const g_snn_conv_weights[];\n")
        f.write("extern const float* const g_snn_conv_biases[];\n")
        f.write("extern const float* const g_snn_fc_weights[];\n")
        f.write("extern const float* const g_snn_fc_biases[];\n\n")
        f.write("#ifdef __cplusplus\n}\n#endif\n")

    # 2. model_data.cc
    cc_path = out_path / "model_data.cc"
    with open(cc_path, "w", encoding="utf-8") as f:
        f.write('#include "model_data.h"\n\n')
        f.write(f'const char* const g_model_name = "{model_name}";\n')

        tflite_bytes = model_entry.get("tflite_bytes", b"")
        if model_type == "ann":
            f.write("const int g_model_type = MODEL_TYPE_ANN;\n\n")
        else:
            f.write("const int g_model_type = MODEL_TYPE_SNN;\n\n")

        f.write('alignas(16) const unsigned char g_model[] = {\n')
        for i, b in enumerate(tflite_bytes):
            if i % 12 == 0:
                f.write('  ')
            f.write(f'0x{b:02x},')
            if i % 12 == 11:
                f.write('\n')
            else:
                f.write(' ')
        if len(tflite_bytes) % 12 != 0:
            f.write('\n')
        f.write('};\n\n')
        f.write(f'const int g_model_len = {len(tflite_bytes)};\n\n')

        if model_type == "ann":
            f.write("const snn_config_t g_snn_config = {};\n")
            f.write("const float* const g_snn_conv_weights[] = {0};\n")
            f.write("const float* const g_snn_conv_biases[] = {0};\n")
            f.write("const float* const g_snn_fc_weights[] = {0};\n")
            f.write("const float* const g_snn_fc_biases[] = {0};\n")
        elif model_type == "snn":
            num_conv = len(getattr(model, "conv_layers_list", []))
            if num_conv > 0:
                for c_idx, conv in enumerate(model.conv_layers_list):
                    w = conv.weight.detach().cpu().numpy().astype(np.float32)
                    b = conv.bias.detach().cpu().numpy().astype(np.float32)
                    f.write(f"// SNN Conv {c_idx}: shape {w.shape}\n")
                    f.write(f"static const float s_snn_conv_w{c_idx}[{w.size}] = {{\n")
                    w_flat = w.flatten()
                    for i, val in enumerate(w_flat):
                        if i % 8 == 0:
                            f.write('  ')
                        f.write(f'{val:.7f}f, ')
                        if i % 8 == 7:
                            f.write('\n')
                    if len(w_flat) % 8 != 0:
                        f.write('\n')
                    f.write("};\n\n")

                    f.write(f"static const float s_snn_conv_b{c_idx}[{b.size}] = {{\n  ")
                    f.write(", ".join(f"{val:.7f}f" for val in b))
                    f.write("\n};\n\n")

                f.write("const float* const g_snn_conv_weights[] = {\n")
                for c_idx in range(num_conv):
                    f.write(f"    s_snn_conv_w{c_idx},\n")
                f.write("};\n\n")

                f.write("const float* const g_snn_conv_biases[] = {\n")
                for c_idx in range(num_conv):
                    f.write(f"    s_snn_conv_b{c_idx},\n")
                f.write("};\n\n")
            else:
                f.write("const float* const g_snn_conv_weights[] = {0};\n")
                f.write("const float* const g_snn_conv_biases[] = {0};\n\n")

            # FC Schichten
            fcs = model.fcs
            num_fc = len(fcs)
            for l_idx, fc in enumerate(fcs):
                w = fc.weight.detach().cpu().numpy().astype(np.float32)
                b = fc.bias.detach().cpu().numpy().astype(np.float32)

                f.write(f"// SNN FC Schicht {l_idx}: {w.shape[1]} -> {w.shape[0]}\n")
                f.write(f"static const float s_snn_fc_w{l_idx}[{w.size}] = {{\n")
                w_flat = w.flatten()
                for i, val in enumerate(w_flat):
                    if i % 8 == 0:
                        f.write('  ')
                    f.write(f'{val:.7f}f, ')
                    if i % 8 == 7:
                        f.write('\n')
                if len(w_flat) % 8 != 0:
                    f.write('\n')
                f.write("};\n\n")

                f.write(f"static const float s_snn_fc_b{l_idx}[{b.size}] = {{\n  ")
                f.write(", ".join(f"{val:.7f}f" for val in b))
                f.write("\n};\n\n")

            f.write("const float* const g_snn_fc_weights[] = {\n")
            for l_idx in range(num_fc):
                f.write(f"    s_snn_fc_w{l_idx},\n")
            f.write("};\n\n")

            f.write("const float* const g_snn_fc_biases[] = {\n")
            for l_idx in range(num_fc):
                f.write(f"    s_snn_fc_b{l_idx},\n")
            f.write("};\n\n")

            n_type_code = 1 if getattr(model, "neuron_type", "lif") == "if" else 0
            enc_code = get_encoding_code(getattr(model, "encoding_type", "direct"))
            exec_mode_code = 1 if getattr(model, "execution_mode", "dense") in ("event_based", "event_driven", "event") else 0
            fc_layer_sizes = [model.flat_size] + model.hidden_layers + [output_size]
            fc_sizes_str = ", ".join(str(s) for s in fc_layer_sizes)

            f.write("const snn_config_t g_snn_config = {\n")
            f.write(f"    .input_size = {input_size},\n")
            f.write(f"    .num_conv_layers = {num_conv},\n")
            f.write("    .conv_layers = {\n")
            if num_conv > 0:
                for cl in model.resolved_conv:
                    f.write(f"        {{{cl['in_channels']}, {cl['out_channels']}, {cl['kernel_size']}, {cl['stride']}, {cl['padding']}, {cl['pool_size']}, {cl['pool_stride']}, {cl['in_h']}, {cl['in_w']}, {cl['out_h']}, {cl['out_w']}, {cl['pool_out_h']}, {cl['pool_out_w']}}},\n")
            f.write("    },\n")
            f.write(f"    .flat_size = {model.flat_size},\n")
            f.write(f"    .hidden_layers_count = {len(model.hidden_layers)},\n")
            f.write(f"    .fc_layer_sizes = {{{fc_sizes_str}}},\n")
            f.write(f"    .num_steps = {model.num_steps},\n")
            f.write(f"    .beta = {model.beta:.4f}f,\n")
            f.write(f"    .threshold = {model.threshold:.4f}f,\n")
            f.write(f"    .output_size = {output_size},\n")
            f.write(f"    .neuron_type = {n_type_code},\n")
            f.write(f"    .encoding_type = {enc_code},\n")
            f.write(f"    .execution_mode = {exec_mode_code},\n")
            f.write("};\n")

    dataset_name = config.get("dataset", config.get("training", {}).get("dataset", "mnist"))
    if isinstance(dataset_name, dict):
        dataset_name = dataset_name.get("name", "mnist")
    dataset_name = str(dataset_name).strip().lower()

    # 3. mnist_test_data.h
    test_header_path = out_path / "mnist_test_data.h"
    with open(test_header_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define DATASET_NAME \"{dataset_name}\"\n")
        f.write(f"#define DATASET_INPUT_SIZE {input_size}\n")
        f.write(f"#define DATASET_NUM_CLASSES {output_size}\n\n")
        f.write(f"#define MNIST_TEST_SUBSET {num_test_samples}\n")
        f.write(f"#define MNIST_IMAGE_SIZE {input_size}\n")
        f.write(f"#define MNIST_NUM_OUTPUTS {output_size}\n\n")

        labels = [int(test_dataset[i][1]) for i in range(num_test_samples)]
        f.write("static const uint8_t mnist_test_labels[MNIST_TEST_SUBSET] = {\n  ")
        f.write(", ".join(str(l) for l in labels))
        f.write("\n};\n\n")

        f.write("static const float mnist_test_images[MNIST_TEST_SUBSET][MNIST_IMAGE_SIZE] = {\n")
        for i in range(num_test_samples):
            img_tensor, _ = test_dataset[i]
            img_np = img_tensor.view(-1).numpy()
            f.write("  {\n")
            for row in range(0, input_size, 14):
                chunk = img_np[row:min(row + 14, input_size)]
                f.write("    " + ", ".join(f"{v:.4f}f" for v in chunk) + ",\n")
            f.write("  },\n")
        f.write("};\n")

    size_desc = f"{len(model_entry.get('tflite_bytes', b'')):,} Bytes" if model_type == "ann" else f"{model_entry.get('model_size_bytes', 0):,} Bytes"
    print(f"[Export] Modell '{model_name}' ({model_type.upper()}, {size_desc}) und {num_test_samples} Testsamples nach '{output_dir}/' exportiert:")
    print(f"  - {header_path}")
    print(f"  - {cc_path}")
    print(f"  - {test_header_path}\n")


def get_model_configs(config: Dict[str, Any], dataset_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extrahiert alle konfigurierten Modelle aus der Konfiguration.
    Optional kann nach einem spezifischen Datensatz gefiltert werden.
    """
    models_list = config.get("models")
    resolved_models = []

    global_training = dict(config.get("training", {}))
    global_evaluation = dict(config.get("evaluation", {}))
    global_optimization = dict(config.get("optimization", {})) if isinstance(config.get("optimization"), dict) else {}
    global_esp32 = dict(config.get("esp32", {}))
    global_joulescope = dict(config.get("joulescope", {}))

    target_ds_filter = normalize_dataset_name(dataset_filter) if dataset_filter else None

    global_dataset = target_ds_filter or config.get("dataset", global_training.get("dataset", "mnist"))
    if isinstance(global_dataset, dict):
        global_dataset = global_dataset.get("name", "mnist")
    global_dataset = normalize_dataset_name(global_dataset)

    if isinstance(models_list, list) and len(models_list) > 0:
        has_specific_matches = False
        if target_ds_filter:
            for m in models_list:
                if isinstance(m, dict):
                    m_d = m.get("dataset", "")
                    if isinstance(m_d, dict):
                        m_d = m_d.get("name", "")
                    if normalize_dataset_name(m_d) == target_ds_filter:
                        has_specific_matches = True
                        break

        for idx, m in enumerate(models_list):
            if not isinstance(m, dict):
                continue

            m_ds = m.get("dataset", global_dataset)
            if isinstance(m_ds, dict):
                m_ds = m_ds.get("name", global_dataset)
            m_ds = normalize_dataset_name(m_ds)

            # Wenn nach einem Datensatz gefiltert wird und passende Modelle existieren, nur diese wählen
            if target_ds_filter and has_specific_matches and m_ds != target_ds_filter:
                continue

            if target_ds_filter and not has_specific_matches:
                m_ds = target_ds_filter

            m_type = str(m.get("type", "ann")).lower()
            conv_layers = parse_conv_layers(m.get("conv_layers", m.get("conv", [])))
            hidden_layers = parse_hidden_layers(m.get("hidden_layers", [64]))
            name = m.get("name")
            if not name:
                layer_str = "_".join(str(h) for h in hidden_layers) if hidden_layers else "linear"
                conv_str = f"Conv{len(conv_layers)}_" if conv_layers else ""
                prefix = "SNN" if m_type == "snn" else "ANN"
                name = f"{prefix}_{conv_str}{layer_str}"

            ds_info = AVAILABLE_DATASETS.get(m_ds, AVAILABLE_DATASETS["mnist"])

            input_size = int(m.get("input_size", ds_info["input_size"]))
            output_size = int(m.get("output_size", ds_info["num_classes"]))
            in_channels = int(m.get("in_channels", ds_info["in_channels"]))
            in_height = int(m.get("in_height", ds_info["in_height"]))
            in_width = int(m.get("in_width", ds_info["in_width"]))

            model_dict = {
                "type": m_type,
                "dataset": m_ds,
                "input_size": input_size,
                "in_channels": in_channels,
                "in_height": in_height,
                "in_width": in_width,
                "conv_layers": conv_layers,
                "hidden_layers": hidden_layers,
                "output_size": output_size,
            }
            if m_type == "snn":
                model_dict["neuron_type"] = str(m.get("neuron_type", m.get("neuron", "lif"))).lower()
                model_dict["encoding_type"] = str(m.get("encoding_type", m.get("encoding", "direct"))).lower()
                model_dict["num_steps"] = int(m.get("num_steps", m.get("steps", 10)))
                model_dict["beta"] = float(m.get("beta", 0.9))
                model_dict["threshold"] = float(m.get("threshold", 1.0))
            else:
                model_dict["activation"] = str(m.get("activation", config.get("model", {}).get("activation", "relu")))

            # Globale Trainingsparameter als Basis
            model_training = dict(global_training)
            model_training["dataset"] = m_ds
            if isinstance(m.get("training"), dict):
                nested_tr = dict(m["training"])
                if "lr" in nested_tr and "learning_rate" not in nested_tr:
                    nested_tr["learning_rate"] = nested_tr.pop("lr")
                model_training.update(nested_tr)
            for k in ("epochs", "batch_size", "learning_rate", "lr", "optimizer", "data_dir", "random_seed", "early_stopping", "patience", "min_delta", "monitor", "restore_best_weights"):
                if k in m:
                    if k == "lr":
                        model_training["learning_rate"] = m[k]
                    else:
                        model_training[k] = m[k]

            if "epochs" in model_training and model_training["epochs"] is not None:
                model_training["epochs"] = int(model_training["epochs"])
            if "batch_size" in model_training and model_training["batch_size"] is not None:
                model_training["batch_size"] = int(model_training["batch_size"])
            if "learning_rate" in model_training and model_training["learning_rate"] is not None:
                model_training["learning_rate"] = float(model_training["learning_rate"])
            if "patience" in model_training and model_training["patience"] is not None:
                model_training["patience"] = int(model_training["patience"])
            if "min_delta" in model_training and model_training["min_delta"] is not None:
                model_training["min_delta"] = float(model_training["min_delta"])

            model_evaluation = dict(global_evaluation)
            if isinstance(m.get("evaluation"), dict):
                model_evaluation.update(m["evaluation"])
            for k in ("num_test_samples", "test_samples"):
                if k in m:
                    if k == "test_samples":
                        model_evaluation["num_test_samples"] = m[k]
                    else:
                        model_evaluation[k] = m[k]
            if "num_test_samples" in model_evaluation and model_evaluation["num_test_samples"] is not None:
                model_evaluation["num_test_samples"] = int(model_evaluation["num_test_samples"])

            model_optimization = copy.deepcopy(global_optimization)
            if isinstance(m.get("optimization"), dict):
                model_optimization.update(m["optimization"])
            for k in ("pruning", "quantization", "clustering"):
                if k in m:
                    model_optimization[k] = m[k]

            resolved_models.append({
                "name": name,
                "type": m_type,
                "dataset": m_ds,
                "model": model_dict,
                "training": model_training,
                "evaluation": model_evaluation,
                "optimization": model_optimization,
                "esp32": global_esp32,
                "joulescope": global_joulescope,
            })
        return resolved_models

    single_m = config.get("model", {})
    m_type = str(single_m.get("type", "ann")).lower()
    conv_layers = parse_conv_layers(single_m.get("conv_layers", single_m.get("conv", [])))
    hidden_layers = parse_hidden_layers(single_m.get("hidden_layers", [64]))
    name = single_m.get("name")
    if not name:
        layer_str = "_".join(str(h) for h in hidden_layers) if hidden_layers else "linear"
        conv_str = f"Conv{len(conv_layers)}_" if conv_layers else ""
        prefix = "SNN" if m_type == "snn" else "ANN"
        name = f"{prefix}_{conv_str}{layer_str}"

    single_ds = single_m.get("dataset", global_dataset)
    if isinstance(single_ds, dict):
        single_ds = single_ds.get("name", global_dataset)
    single_ds = normalize_dataset_name(single_ds)
    ds_info = AVAILABLE_DATASETS.get(single_ds, AVAILABLE_DATASETS["mnist"])

    input_size = int(single_m.get("input_size", ds_info["input_size"]))
    output_size = int(single_m.get("output_size", ds_info["num_classes"]))
    in_channels = int(single_m.get("in_channels", ds_info["in_channels"]))
    in_height = int(single_m.get("in_height", ds_info["in_height"]))
    in_width = int(single_m.get("in_width", ds_info["in_width"]))

    model_dict = {
        "type": m_type,
        "dataset": single_ds,
        "input_size": input_size,
        "in_channels": in_channels,
        "in_height": in_height,
        "in_width": in_width,
        "conv_layers": conv_layers,
        "hidden_layers": hidden_layers,
        "output_size": output_size,
    }
    if m_type == "snn":
        model_dict["neuron_type"] = str(single_m.get("neuron_type", single_m.get("neuron", "lif"))).lower()
        model_dict["encoding_type"] = str(single_m.get("encoding_type", single_m.get("encoding", "direct"))).lower()
        model_dict["num_steps"] = int(single_m.get("num_steps", single_m.get("steps", 10)))
        model_dict["beta"] = float(single_m.get("beta", 0.9))
        model_dict["threshold"] = float(single_m.get("threshold", 1.0))
    else:
        model_dict["activation"] = str(single_m.get("activation", "relu"))

    model_training = dict(global_training)
    model_training["dataset"] = single_ds
    if isinstance(single_m.get("training"), dict):
        nested_tr = dict(single_m["training"])
        if "lr" in nested_tr and "learning_rate" not in nested_tr:
            nested_tr["learning_rate"] = nested_tr.pop("lr")
        model_training.update(nested_tr)
    for k in ("epochs", "batch_size", "learning_rate", "lr", "optimizer", "data_dir", "random_seed", "early_stopping", "patience", "min_delta", "monitor", "restore_best_weights"):
        if k in single_m:
            if k == "lr":
                model_training["learning_rate"] = single_m[k]
            else:
                model_training[k] = single_m[k]

    if "epochs" in model_training and model_training["epochs"] is not None:
        model_training["epochs"] = int(model_training["epochs"])
    if "batch_size" in model_training and model_training["batch_size"] is not None:
        model_training["batch_size"] = int(model_training["batch_size"])
    if "learning_rate" in model_training and model_training["learning_rate"] is not None:
        model_training["learning_rate"] = float(model_training["learning_rate"])
    if "patience" in model_training and model_training["patience"] is not None:
        model_training["patience"] = int(model_training["patience"])
    if "min_delta" in model_training and model_training["min_delta"] is not None:
        model_training["min_delta"] = float(model_training["min_delta"])

    model_evaluation = dict(global_evaluation)
    if isinstance(single_m.get("evaluation"), dict):
        model_evaluation.update(single_m["evaluation"])
    for k in ("num_test_samples", "test_samples"):
        if k in single_m:
            if k == "test_samples":
                model_evaluation["num_test_samples"] = single_m[k]
            else:
                model_evaluation[k] = single_m[k]
    if "num_test_samples" in model_evaluation and model_evaluation["num_test_samples"] is not None:
        model_evaluation["num_test_samples"] = int(model_evaluation["num_test_samples"])

    model_optimization = copy.deepcopy(global_optimization)
    if isinstance(single_m.get("optimization"), dict):
        model_optimization.update(single_m["optimization"])
    for k in ("pruning", "quantization", "clustering"):
        if k in single_m:
            model_optimization[k] = single_m[k]

    num_steps = model_dict.get("num_steps") if m_type == "snn" else None
    resolved_models.append({
        "name": f"{name}{'_T' + str(num_steps) if num_steps is not None else ''}",
        "type": m_type,
        "dataset": single_ds,
        "model": model_dict,
        "training": model_training,
        "evaluation": model_evaluation,
        "optimization": model_optimization,
        "esp32": global_esp32,
        "joulescope": global_joulescope,
    })
    return resolved_models


def train_all_models(
    config_or_path: Union[str, Path, Dict[str, Any]] = None,
    dataset: str = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    early_stopping: bool = None,
    patience: int = None,
    min_delta: float = None,
    monitor: str = None,
    num_test_samples: int = None,
    force_retrain: bool = False,
    save_dir: str = None,
    prune_amount: float = None,
    prune_method: str = None,
    quantize_type: str = None,
    num_clusters: int = None,
    fine_tune_epochs: int = None,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Trainiert ALLE in der Konfiguration definierten Modelle nacheinander (Batch-Training Phase 1)
    oder lädt unveränderte Modelle aus dem Checkpoint-Cache und führt ggf. Post-Training Optimierungen durch.
    """
    if isinstance(config_or_path, dict):
        config = config_or_path
    else:
        config = load_config(config_or_path)

    target_dataset = normalize_dataset_name(dataset) if dataset is not None else None
    if target_dataset is not None:
        config["dataset"] = target_dataset

    model_configs = get_model_configs(config, dataset_filter=target_dataset)
    total_models = len(model_configs)
    print(f"\n=================================================================")
    print(f"=== BATCH-TRAINING: {total_models} Modell(e) konfiguriert ===")
    print(f"=================================================================")

    trained_models = []

    for idx, m_cfg in enumerate(model_configs, start=1):
        if target_dataset is not None:
            m_cfg["dataset"] = target_dataset
            m_cfg["model"]["dataset"] = target_dataset
            ds_info = AVAILABLE_DATASETS.get(target_dataset, AVAILABLE_DATASETS["mnist"])
            m_cfg["model"]["input_size"] = ds_info["input_size"]
            m_cfg["model"]["output_size"] = ds_info["num_classes"]
            m_cfg["model"]["in_channels"] = ds_info["in_channels"]
            m_cfg["model"]["in_height"] = ds_info["in_height"]
            m_cfg["model"]["in_width"] = ds_info["in_width"]
        if epochs is not None:
            m_cfg["training"]["epochs"] = epochs
        if batch_size is not None:
            m_cfg["training"]["batch_size"] = batch_size
        if learning_rate is not None:
            m_cfg["training"]["learning_rate"] = learning_rate
        if early_stopping is not None:
            m_cfg["training"]["early_stopping"] = early_stopping
        if patience is not None:
            m_cfg["training"]["patience"] = patience
        if min_delta is not None:
            m_cfg["training"]["min_delta"] = min_delta
        if monitor is not None:
            m_cfg["training"]["monitor"] = monitor
        if num_test_samples is not None:
            m_cfg["evaluation"]["num_test_samples"] = num_test_samples
        if force_retrain:
            m_cfg["training"]["force_retrain"] = True
        if save_dir is not None:
            m_cfg["training"]["save_dir"] = save_dir

        if "optimization" not in m_cfg:
            m_cfg["optimization"] = {}
        if prune_amount is not None:
            if "pruning" not in m_cfg["optimization"]:
                m_cfg["optimization"]["pruning"] = {}
            m_cfg["optimization"]["pruning"]["enabled"] = True
            m_cfg["optimization"]["pruning"]["amount"] = float(prune_amount)
        if prune_method is not None:
            if "pruning" not in m_cfg["optimization"]:
                m_cfg["optimization"]["pruning"] = {}
            m_cfg["optimization"]["pruning"]["method"] = str(prune_method)
        if quantize_type is not None:
            if "quantization" not in m_cfg["optimization"]:
                m_cfg["optimization"]["quantization"] = {}
            m_cfg["optimization"]["quantization"]["enabled"] = True
            m_cfg["optimization"]["quantization"]["type"] = str(quantize_type)
        if num_clusters is not None:
            if "clustering" not in m_cfg["optimization"]:
                m_cfg["optimization"]["clustering"] = {}
            m_cfg["optimization"]["clustering"]["enabled"] = True
            m_cfg["optimization"]["clustering"]["num_clusters"] = int(num_clusters)
        if fine_tune_epochs is not None:
            if "pruning" not in m_cfg["optimization"]:
                m_cfg["optimization"]["pruning"] = {}
            m_cfg["optimization"]["pruning"]["fine_tune_epochs"] = int(fine_tune_epochs)

        m_force = m_cfg["training"].get("force_retrain", force_retrain)
        m_save_dir = m_cfg["training"].get("save_dir", save_dir)

        m_type = m_cfg.get("type", "ann").lower()
        ep = m_cfg["training"].get("epochs", "?")
        bs = m_cfg["training"].get("batch_size", "?")
        lr = m_cfg["training"].get("learning_rate", "?")
        ds = m_cfg.get("dataset", m_cfg["model"].get("dataset", "mnist"))
        print(f"\n--- [{idx}/{total_models}] Modell ({m_type.upper()} mit {'SpikingJelly' if m_type == 'snn' else 'PyTorch'}): '{m_cfg['name']}' [Datensatz: {ds}, Epochen: {ep}, Batch: {bs}, LR: {lr}] ---")
        model, test_dataset, test_acc, test_loss = train_model(
            m_cfg,
            device=device,
            force_retrain=m_force,
            save_dir=m_save_dir,
        )

        samples_cnt = int(m_cfg.get("evaluation", {}).get("num_test_samples", 500))
        subset_acc, subset_correct, subset_total = evaluate_subset(
            model=model,
            test_dataset=test_dataset,
            num_samples=samples_cnt,
            device=device,
        )
        print(f"[Evaluation] Host-Genauigkeit auf ESP32-Testset ({subset_total} Samples): {subset_acc:.2f}% ({subset_correct}/{subset_total} korrekt).")

        tflite_bytes = export_pytorch_to_tflite(model) if m_type == "ann" else b""

        # Parameteranzahl, Sparsity & Modellgröße
        total_params = sum(p.numel() for p in model.parameters())
        sparsity_info = compute_model_sparsity(model)
        model_size_bytes = (total_params * 4) if m_type == "snn" else len(tflite_bytes)

        num_steps = m_cfg["model"].get("num_steps") if m_type == "snn" else None
        trained_entry = {
            "name": f"{m_cfg['name']}{'_T' + str(num_steps) if num_steps is not None else ''}",
            "type": m_type,
            "dataset": ds,
            "config": m_cfg,
            "model": model,
            "test_dataset": test_dataset,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "subset_acc": subset_acc,
            "subset_test_acc": subset_acc,
            "subset_correct": subset_correct,
            "subset_total": subset_total,
            "tflite_bytes": tflite_bytes,
            "model_size_bytes": model_size_bytes,
            "total_params": total_params,
            "sparsity_pct": sparsity_info["sparsity_pct"],
            "zero_weights": sparsity_info["zero_weights"],
            "structure": model.get_structure_summary(),
        }
        trained_models.append(trained_entry)

    print("\n=================================================================")
    print(f"=== BATCH-TRAINING ABGESCHLOSSEN: {len(trained_models)} Modell(e) bereit ===")
    print(f"=================================================================\n")
    return trained_models


def export_model_entry_to_main(model_entry: Dict[str, Any], output_dir: str = "main"):
    """
    Exportiert ein einzelnes bereits trainiertes Modell nach 'main/model_data.h/cc'.
    """
    export_c_files(model_entry, output_dir=output_dir)


def train_and_export(
    config_path: Union[str, Path] = None,
    dataset: str = None,
    output_dir: str = "main",
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    early_stopping: bool = None,
    patience: int = None,
    min_delta: float = None,
    monitor: str = None,
    num_test_samples: int = None,
    force_retrain: bool = False,
    save_dir: str = None,
    prune_amount: float = None,
    prune_method: str = None,
    quantize_type: str = None,
    num_clusters: int = None,
    fine_tune_epochs: int = None,
) -> Tuple[Union[ConfigurableANN, ConfigurableSpikingJellySNN], bytes]:
    """
    Trainiert alle Modelle und exportiert das erste Modell nach output_dir.
    """
    trained_models = train_all_models(
        config_or_path=config_path,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        monitor=monitor,
        num_test_samples=num_test_samples,
        force_retrain=force_retrain,
        save_dir=save_dir,
        prune_amount=prune_amount,
        prune_method=prune_method,
        quantize_type=quantize_type,
        num_clusters=num_clusters,
        fine_tune_epochs=fine_tune_epochs,
    )
    if trained_models:
        export_model_entry_to_main(trained_models[0], output_dir=output_dir)
        return trained_models[0]["model"], trained_models[0]["tflite_bytes"]
    return None, b""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuronale Netze (ANN & SNN) Training, Datensatzauswahl, Post-Training-Optimierung & TFLite/C-Export")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Pfad zur YAML- oder JSON-Konfigurationsdatei (Standard: 'config.yaml')",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default=None,
        help="Wählt den Datensatz (z.B. 'mnist', 'fashion_mnist', 'cifar10', 'speech_commands', 'har_time_series', 'eeg_data', 'ecg_data', 'synthetic_time_series')",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Zeigt alle verfügbaren Datensätze und Domänen in einer Tabelle an",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=None,
        help="Überschreibt die Anzahl der Trainingsepochen",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=None,
        help="Überschreibt die Batch-Größe für das Training",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Überschreibt die Lernrate für das Training",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=None,
        help="Aktiviert Early Stopping für das Training",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_false",
        dest="early_stopping",
        help="Deaktiviert Early Stopping für das Training",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Anzahl an Epochen ohne Verbesserung vor vorzeitigem Abbruch",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Mindestverbesserung der Metrik für Early Stopping",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        choices=["test_acc", "test_loss"],
        help="Zu überwachende Metrik: 'test_acc' (Standard) oder 'test_loss'",
    )
    parser.add_argument(
        "-n", "--test-samples",
        type=int,
        default=None,
        dest="test_samples",
        help="Überschreibt die Anzahl der zu exportierenden Test-Samples",
    )
    parser.add_argument(
        "--force-train", "--force-retrain",
        dest="force_retrain",
        action="store_true",
        default=False,
        help="Erzwingt ein erneutes Training aller Modelle, auch wenn gespeicherte Checkpoints unverändert vorliegen",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Verzeichnis zum automatischen Speichern und Laden trainierter Modell-Checkpoints (Standard: './saved_models')",
    )

    # Post-Training Optimierungs-Optionen
    parser.add_argument(
        "--prune", "--prune-amount",
        type=float,
        default=None,
        dest="prune_amount",
        help="Aktiviert Post-Training Pruning und setzt den Zielanteil entfernter Gewichte (z. B. 0.3 für 30%% Sparsity)",
    )
    parser.add_argument(
        "--prune-method",
        type=str,
        default=None,
        choices=["l1_unstructured", "random_unstructured", "structured", "threshold"],
        help="Pruning-Methode: 'l1_unstructured' (Standard), 'random_unstructured', 'structured' oder 'threshold'",
    )
    parser.add_argument(
        "--quantize", "--quantize-type",
        type=str,
        default=None,
        dest="quantize_type",
        choices=["int8", "int4", "fp16", "dynamic"],
        help="Aktiviert Post-Training Quantisierung der Gewichte ('int8', 'int4', 'fp16', 'dynamic')",
    )
    parser.add_argument(
        "--clustering", "--num-clusters",
        type=int,
        default=None,
        dest="num_clusters",
        help="Aktiviert Weight-Clustering mit k Centroids (z. B. 16 für 4-Bit Codebook)",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=None,
        help="Anzahl an Fine-Tuning Epochen nach dem Pruning zur Genauigkeitswiederherstellung",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print_available_datasets_table()
        sys.exit(0)

    train_and_export(
        config_path=args.config,
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        monitor=args.monitor,
        num_test_samples=args.test_samples,
        force_retrain=args.force_retrain,
        save_dir=args.save_dir,
        prune_amount=args.prune_amount,
        prune_method=args.prune_method,
        quantize_type=args.quantize_type,
        num_clusters=args.num_clusters,
        fine_tune_epochs=args.fine_tune_epochs,
    )
