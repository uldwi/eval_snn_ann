"""Configuration defaults, loading, and architecture parsing."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None
HAS_YAML = yaml is not None

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
        # Optimizers: "adam", "sgd", "adamw", "adagrad", "rmsprop" (configurable per model)
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
    "microcontroller": {
        "target": "esp32c6",
        "boards": {
            "esp32c6": {
                "name": "ESP32-C6",
                "board": "ESP32-C6-DevKitC-1",
                "architecture": "RISC-V",
                "cpu_freq_mhz": 160,
                "sram_kb": 512,
                "trigger_gpio": 4,
                "joulescope_gpi": "in1",
            },
            "esp32s3": {
                "name": "ESP32-S3",
                "board": "ESP32-S3-DevKitC-1 v1.1",
                "architecture": "Xtensa LX7",
                "cpu_freq_mhz": 240,
                "sram_kb": 512,
                "trigger_gpio": 4,
                "joulescope_gpi": "in1",
            },
        },
    },
    "joulescope": {
        "timeout": 60.0,
        "max_duration": None,
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
    Load a YAML or JSON configuration file.
    """
    if config_path is None:
        candidates = ["config.yaml", "config.yml", "config.json"]
        for c in candidates:
            if os.path.isfile(c):
                config_path = c
                break

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))

    if config_path and os.path.isfile(config_path):
        print(f"[Config] Loading configuration from '{config_path}'...")
        path_str = str(config_path)
        with open(path_str, "r", encoding="utf-8") as f:
            if path_str.endswith((".yaml", ".yml")):
                if not HAS_YAML:
                    raise ImportError("PyYAML is not installed. Install it with 'pip install pyyaml'.")
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
            print(f"[Config] Configuration file '{config_path}' not found; using defaults.")
        else:
            print("[Config] No configuration file specified; using defaults.")

    return cfg

def parse_hidden_layers(val: Any) -> List[int]:
    """
    Parse and normalize hidden-layer configuration in multiple formats.
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
    Parse and normalize convolutional-layer configuration in multiple formats.
    Supported formats include:
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
                kernel_value = item.get("kernel_size", item.get("kernel", 3))
                if isinstance(kernel_value, (list, tuple)):
                    kernel_h, kernel_w = int(kernel_value[0]), int(kernel_value[1])
                else:
                    kernel_h = kernel_w = int(kernel_value)
                stride = int(item.get("stride", 1))
                padding = item.get("padding", None)
                if padding is None:
                    ph, pw = ((kernel_h // 2, kernel_w // 2)
                              if (kernel_h, kernel_w) == (3, 3) else (0, 0))
                    padding = ph if ph == pw else (ph, pw)
                else:
                    if isinstance(padding, (list, tuple)):
                        ph, pw = int(padding[0]), int(padding[1])
                        padding = ph if ph == pw else (ph, pw)
                    else:
                        p = int(padding) if isinstance(padding, (int, str)) and str(padding).isdigit() else 0
                        padding = p

                pool_value = item.get("pool_size", item.get("pool", item.get("max_pool", 0)))
                if isinstance(pool_value, (list, tuple)):
                    pool_h, pool_w = int(pool_value[0]), int(pool_value[1])
                else:
                    pool_h = pool_w = int(pool_value) if pool_value else 0
                pool_stride_value = item.get("pool_stride", pool_value)
                if isinstance(pool_stride_value, (list, tuple)):
                    pool_stride_h, pool_stride_w = int(pool_stride_value[0]), int(pool_stride_value[1])
                else:
                    pool_stride_h = pool_stride_w = int(pool_stride_value) if pool_stride_value else 0
                pool_type = str(item.get("pool_type", "max")).lower()

                result.append({
                    "out_channels": out_channels,
                    "kernel_size": kernel_h if kernel_h == kernel_w else (kernel_h, kernel_w),
                    "kernel_h": kernel_h,
                    "kernel_w": kernel_w,
                    "stride": stride,
                    "padding": padding,
                    "pool_size": pool_h if pool_h == pool_w else (pool_h, pool_w),
                    "pool_h": pool_h,
                    "pool_w": pool_w,
                    "pool_stride": pool_stride_h if pool_stride_h == pool_stride_w else (pool_stride_h, pool_stride_w),
                    "pool_stride_h": pool_stride_h,
                    "pool_stride_w": pool_stride_w,
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
                    "kernel_h": k_size,
                    "kernel_w": k_size,
                    "stride": stride,
                    "padding": padding,
                    "pool_size": pool_size,
                    "pool_h": pool_size,
                    "pool_w": pool_size,
                    "pool_stride": pool_stride,
                    "pool_stride_h": pool_stride,
                    "pool_stride_w": pool_stride,
                    "pool_type": "max",
                })
        return result
    return []

def compute_conv_layer_dims(conv_layers: List[Dict[str, Any]], in_c: int = 1, in_h: int = 28, in_w: int = 28) -> Tuple[List[Dict[str, Any]], int]:
    """
    Compute input/output dimensions for each convolutional layer and the resulting flattened tensor.
    """
    cur_c, cur_h, cur_w = in_c, in_h, in_w
    resolved_layers = []
    for cl in conv_layers:
        out_c = cl["out_channels"]
        k_h = cl.get("kernel_h", cl["kernel_size"])
        k_w = cl.get("kernel_w", cl["kernel_size"])
        stride = cl["stride"]
        pad_h, pad_w = cl.get("padding", 0) if isinstance(cl.get("padding", 0), tuple) else (cl.get("padding", 0),) * 2
        pool_h = cl.get("pool_h", cl["pool_size"])
        pool_w = cl.get("pool_w", cl["pool_size"])
        pool_stride_h = cl.get("pool_stride_h", cl["pool_stride"])
        pool_stride_w = cl.get("pool_stride_w", cl["pool_stride"])

        out_h = (cur_h + 2 * pad_h - k_h) // stride + 1
        out_w = (cur_w + 2 * pad_w - k_w) // stride + 1

        if pool_h > 1 or pool_w > 1:
            pool_out_h = (out_h - pool_h) // pool_stride_h + 1
            pool_out_w = (out_w - pool_w) // pool_stride_w + 1
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
