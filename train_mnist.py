#!/usr/bin/env python3
"""
Training eines konfigurierbaren künstlichen neuronalen Netzes (ANN/MLP) für den MNIST-Datensatz
mit PyTorch und Export des Modells in das TensorFlow Lite FlatBuffer-Format (.tflite) sowie
als C/C++ Header- und Quelldateien für TensorFlow Lite Micro auf dem ESP32.

Die Architektur des Netzes, Trainings- und Evaluierungsparameter können vollständig über
eine Konfigurationsdatei (z. B. config.yaml oder config.json) gesteuert werden.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from spikingjelly.activation_based import neuron, functional, surrogate, layer as sj_layer
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import flatbuffers
    import tflite
except ImportError:
    print("Fehler: 'flatbuffers' und 'tflite' werden für den Modellexport benötigt.", file=sys.stderr)
    print("Bitte installieren mit: pip install flatbuffers tflite", file=sys.stderr)
    sys.exit(1)


# -------------------------------------------------------------
# Standardkonfiguration
# -------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "input_size": 784,
        "hidden_layers": [64],
        "activation": "relu",
        "output_size": 10,
    },
    "training": {
        "epochs": 3,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "data_dir": "./data",
        "random_seed": 42,
    },
    "evaluation": {
        "num_test_samples": 50,
    },
    "esp32": {
        "trigger_pin": 4,
    },
    "joulescope": {
        "trigger_pin": "in1",
        "timeout": 60.0,
    },
}


def load_config(config_path: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Lädt die Konfiguration aus einer YAML- oder JSON-Datei.
    Falls keine Datei angegeben ist, wird automatisch nach config.yaml oder config.json gesucht.
    """
    if config_path is None:
        candidates = ["config.yaml", "config.yml", "config.json"]
        for c in candidates:
            if os.path.isfile(c):
                config_path = c
                break

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # Deep copy der Defaults

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
    Unterstützt verschiedene Eingabeformate wie:
      - [128, 64]
      - [[128, 64]] (z. B. aus '- [128, 64]' in YAML)
      - [128, 64, 32]
      - 64 (einzelner Integer)
      - "128, 64" oder "128 64"
      - None / [] / ""
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


# -------------------------------------------------------------
# Konfigurierbare Modell-Architekturen (ANN & SNN)
# -------------------------------------------------------------
class ConfigurableANN(nn.Module):
    """
    Flexibles Fully-Connected Neural Network (MLP / ANN), dessen Schichtstruktur,
    Eingangs-/Ausgangsgrößen und Aktivierungsfunktionen konfigurierbar sind.
    """
    def __init__(
        self,
        input_size: int = 784,
        hidden_layers: Any = None,
        activation: str = "relu",
        output_size: int = 10,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_layers = parse_hidden_layers(hidden_layers)
        self.activation_name = (activation or "relu").lower()
        self.output_size = int(output_size)

        all_layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        self.layers = nn.ModuleList()
        for i in range(len(all_layer_sizes) - 1):
            self.layers.append(nn.Linear(all_layer_sizes[i], all_layer_sizes[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.view(x.size(0), -1)
        num_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < num_layers - 1:
                if self.activation_name == "relu":
                    out = F.relu(out)
                elif self.activation_name in ("none", "linear", "identity"):
                    pass
                else:
                    raise ValueError(f"Nicht unterstützte Aktivierungsfunktion: {self.activation_name}")
        return out

    def get_structure_summary(self) -> str:
        all_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        act_str = f" mit {self.activation_name.upper()}" if self.hidden_layers else ""
        return f"ANN [{' -> '.join(str(s) for s in all_sizes)}]{act_str}"


def encode_input(x: torch.Tensor, encoding_type: str, num_steps: int) -> torch.Tensor:
    """
    Kodiert Eingabedaten x (Batch, Features) über num_steps Zeitschritte.
    Unterstützte Encoding-Typen:
      - 'direct' (Standard): Konstanter Eingangswert über alle Zeitschritte (T, Batch, Features)
      - 'ttfs' / 'latency': Time-To-First-Spike / Latency Coding.
                            Helle Pixel feuern früher (t_f = round((T-1) * (1 - x))).
                            Hintergrundpixel (x <= 0) feuern keinen Spike.
      - 'poisson' / 'rate': Stochastisches Poisson-Rate-Coding mit Wahrscheinlichkeit proportional zu x.
      - 'periodic' / 'rate_periodic': Deterministisches Rate-Coding (k = round(x * T) Spikes).
    """
    enc = str(encoding_type).lower().strip()
    if enc in ("direct", "constant", "raw"):
        return x.unsqueeze(0).repeat(num_steps, 1, 1)

    elif enc in ("ttfs", "latency", "time_to_first_spike"):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        t_f = ((num_steps - 1) * (1.0 - x_clamped)).round().long()
        one_hot = F.one_hot(t_f, num_classes=num_steps).to(x.dtype).to(x.device)
        mask = (x_clamped > 0.0).unsqueeze(-1).to(x.dtype)
        one_hot = one_hot * mask
        return one_hot.permute(2, 0, 1)

    elif enc in ("poisson", "rate", "rate_poisson"):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        rand_tensor = torch.rand((num_steps, *x.shape), device=x.device, dtype=x.dtype)
        return (rand_tensor < x_clamped.unsqueeze(0)).to(x.dtype)

    elif enc in ("periodic", "rate_periodic", "phase"):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        k = (x_clamped * num_steps).round().long()
        t_indices = torch.arange(num_steps, device=x.device).unsqueeze(1).unsqueeze(2)
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
    Spiking Neural Network (SNN) trainiert mit SpikingJelly.
    Unterstützt LIFNode (Leaky-Integrate-and-Fire) und IFNode (Integrate-and-Fire) Neuronen
    mit konfigurierbaren Encodings (Direct, TTFS, Poisson, Periodic), Schichten,
    Zeitschritten (num_steps), Membran-Decay (beta / tau bei LIF) und Schwellenwert (threshold).
    """
    def __init__(
        self,
        input_size: int = 784,
        hidden_layers: Any = None,
        output_size: int = 10,
        neuron_type: str = "lif",
        encoding_type: str = "direct",
        num_steps: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_layers = parse_hidden_layers(hidden_layers)
        self.output_size = int(output_size)
        self.neuron_type = str(neuron_type).lower().strip()
        if self.neuron_type not in ("lif", "if"):
            self.neuron_type = "lif"
        self.encoding_type = str(encoding_type).lower().strip() if encoding_type else "direct"
        if self.encoding_type not in ("direct", "ttfs", "latency", "time_to_first_spike", "poisson", "rate", "rate_poisson", "periodic", "rate_periodic", "phase"):
            self.encoding_type = "direct"
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)

        if self.neuron_type == "if":
            self.beta = 1.0
            self.tau = 1000.0
        else:
            self.beta = float(beta)
            # tau in SpikingJelly entspricht: beta = 1 - 1/tau <=> tau = 1 / (1 - beta)
            if self.beta >= 1.0:
                self.tau = 1000.0
            elif self.beta <= 0.0:
                self.tau = 1.0
            else:
                self.tau = float(1.0 / (1.0 - self.beta))

        all_layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        self.fcs = nn.ModuleList([
            sj_layer.Linear(all_layer_sizes[i], all_layer_sizes[i + 1]) for i in range(len(all_layer_sizes) - 1)
        ])
        if self.neuron_type == "if":
            self.lif_layers = nn.ModuleList([
                neuron.IFNode(
                    v_threshold=self.threshold,
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(),
                    detach_reset=True,
                ) for _ in range(len(all_layer_sizes) - 1)
            ])
        else:
            self.lif_layers = nn.ModuleList([
                neuron.LIFNode(
                    tau=self.tau,
                    decay_input=False,
                    v_threshold=self.threshold,
                    v_reset=0.0,
                    surrogate_function=surrogate.ATan(),
                    detach_reset=True,
                ) for _ in range(len(all_layer_sizes) - 1)
            ])
        if HAS_SPIKINGJELLY:
            functional.set_step_mode(self, 'm')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        # Sequence input over num_steps (T, Batch, Features)
        x_seq = encode_input(x_flat, self.encoding_type, self.num_steps)

        cur = x_seq
        for fc, lif in zip(self.fcs, self.lif_layers):
            cur = fc(cur)
            cur = lif(cur)

        out_spikes = cur.sum(dim=0)
        if HAS_SPIKINGJELLY:
            functional.reset_net(self)
        return out_spikes

    def get_structure_summary(self) -> str:
        all_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        n_label = "SNN-IF" if self.neuron_type == "if" else "SNN-LIF"
        enc_label = self.encoding_type.upper()
        if self.neuron_type == "if":
            param_str = f"T={self.num_steps}, Enc={enc_label}, θ={self.threshold:.1f}"
        else:
            param_str = f"T={self.num_steps}, Enc={enc_label}, β={self.beta:.2f}, θ={self.threshold:.1f}"
        return f"SpikingJelly {n_label} [{' -> '.join(str(s) for s in all_sizes)}] ({param_str})"


# -------------------------------------------------------------
# Training
# -------------------------------------------------------------
def train_model(config: Dict[str, Any], device: str = "cpu") -> Tuple[Union[ConfigurableANN, ConfigurableSpikingJellySNN], datasets.MNIST, float, float]:
    """
    Trainiert das konfigurierte PyTorch-Netzwerk (ANN oder SNN mit SpikingJelly) auf dem MNIST-Datensatz.
    Gibt (model, test_dataset, test_acc, avg_loss) zurück.
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    model_type = str(model_cfg.get("type", config.get("type", "ann"))).lower()

    input_size = model_cfg.get("input_size", 784)
    hidden_layers = model_cfg.get("hidden_layers", [64])
    output_size = model_cfg.get("output_size", 10)

    epochs = train_cfg.get("epochs", 3)
    batch_size = train_cfg.get("batch_size", 64)
    lr = train_cfg.get("learning_rate", 0.001)
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    data_dir = train_cfg.get("data_dir", "./data")
    random_seed = train_cfg.get("random_seed", 42)

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    if model_type == "snn":
        neuron_type = str(model_cfg.get("neuron_type", model_cfg.get("neuron", "lif"))).lower()
        encoding_type = str(model_cfg.get("encoding_type", model_cfg.get("encoding", "direct"))).lower()
        num_steps = int(model_cfg.get("num_steps", model_cfg.get("steps", 10)))
        beta = float(model_cfg.get("beta", 0.9)) if neuron_type == "lif" else 1.0
        threshold = float(model_cfg.get("threshold", 1.0))
        model = ConfigurableSpikingJellySNN(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            neuron_type=neuron_type,
            encoding_type=encoding_type,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
        ).to(device)
    else:
        activation = model_cfg.get("activation", "relu")
        model = ConfigurableANN(
            input_size=input_size,
            hidden_layers=hidden_layers,
            activation=activation,
            output_size=output_size,
        ).to(device)

    print("\n" + "=" * 65)
    print(f">>> PyTorch Modell-Konfiguration: {config.get('name', 'Modell')} ({'SNN' if model_type == 'snn' else 'ANN'}) <<<")
    print(f"  - Netzwerk-Aufbau:     {model.get_structure_summary()}")
    print(f"  - Verborgene Schichten:{hidden_layers}")
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
    print("=" * 65 + "\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),  # Wertebereich [0.0, 1.0]
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    test_acc = 0.0
    avg_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)

        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total

        # Test-Evaluierung
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += data.size(0)

        test_acc = 100.0 * test_correct / test_total
        print(f"[PyTorch] Epoche {epoch}/{epochs} - Verlust: {avg_loss:.4f} | Train-Genauigkeit: {train_acc:.2f}% | Test-Genauigkeit: {test_acc:.2f}%")

    print(f"[PyTorch] Training für '{config.get('name', 'Modell')}' abgeschlossen (Test-Genauigkeit: {test_acc:.2f}%).\n")
    return model, test_dataset, test_acc, avg_loss


def evaluate_subset(
    model: Union[ConfigurableANN, ConfigurableSpikingJellySNN],
    test_dataset: datasets.MNIST,
    num_samples: int = 50,
    device: str = "cpu",
) -> Tuple[float, int, int]:
    """
    Evaluiert das Modell auf einer definierten Teilmenge des Testdatensatzes (z. B. den auf den ESP32 exportierten Samples).
    Gibt (genauigkeit_in_prozent, anzahl_korrekt, anzahl_gesamt) zurück.
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
    beliebiges mehrschichtiges Fully-Connected-Netzwerk (ANN oder SNN).
    """
    model.eval()
    if isinstance(model, ConfigurableSpikingJellySNN):
        layers = list(model.fcs)
        act_name = "none"
    else:
        layers = list(model.layers)
        act_name = getattr(model, "activation_name", "relu")

    num_layers = len(layers)
    if num_layers == 0:
        raise ValueError("Modell enthält keine Schichten!")

    bld = flatbuffers.Builder(2 * 1024 * 1024)

    # 1. Buffers anlegen (Buffer 0 ist per TFLite-Spezifikation leer)
    tflite.BufferStart(bld)
    buf0 = tflite.BufferEnd(bld)
    buffers = [buf0]

    def make_buffer(arr: np.ndarray) -> int:
        raw_bytes = arr.tobytes(order="C")
        vec = bld.CreateByteVector(raw_bytes)
        tflite.BufferStart(bld)
        tflite.BufferAddData(bld, vec)
        return tflite.BufferEnd(bld)

    weights_buffers = []
    biases_buffers = []

    for layer in layers:
        w = layer.weight.detach().cpu().numpy().astype(np.float32)  # Shape (out_dim, in_dim)
        b = layer.bias.detach().cpu().numpy().astype(np.float32)    # Shape (out_dim,)
        buf_w = make_buffer(w)
        buf_b = make_buffer(b)
        buffers.append(buf_w)
        buffers.append(buf_b)
        weights_buffers.append(len(buffers) - 2)
        biases_buffers.append(len(buffers) - 1)

    # 2. Operator Code (FULLY_CONNECTED)
    tflite.OperatorCodeStart(bld)
    tflite.OperatorCodeAddBuiltinCode(bld, tflite.BuiltinOperator.FULLY_CONNECTED)
    op_code_fc = tflite.OperatorCodeEnd(bld)

    # 3. Tensors erzeugen
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
    # Tensor 0: input
    t_input = make_tensor("input", [1, model.input_size], tflite.TensorType.FLOAT32, 0)
    tensors.append(t_input)

    # Für jeden Layer: W, B, Output
    for i, layer in enumerate(layers):
        in_dim = layer.in_features
        out_dim = layer.out_features
        w_idx = weights_buffers[i]
        b_idx = biases_buffers[i]

        t_w = make_tensor(f"fc{i+1}_w", [out_dim, in_dim], tflite.TensorType.FLOAT32, w_idx)
        t_b = make_tensor(f"fc{i+1}_b", [out_dim], tflite.TensorType.FLOAT32, b_idx)
        out_name = "output" if i == num_layers - 1 else f"fc{i+1}_out"
        t_out = make_tensor(out_name, [1, out_dim], tflite.TensorType.FLOAT32, 0)

        tensors.extend([t_w, t_b, t_out])

    # 4. Operatoren (FullyConnected für jeden Layer)
    ops = []
    for i in range(num_layers):
        is_last = (i == num_layers - 1)
        use_relu = (not is_last) and (act_name == "relu")

        tflite.FullyConnectedOptionsStart(bld)
        act_type = tflite.ActivationFunctionType.RELU if use_relu else tflite.ActivationFunctionType.NONE
        tflite.FullyConnectedOptionsAddFusedActivationFunction(bld, act_type)
        opt_fc = tflite.FullyConnectedOptionsEnd(bld)

        input_tensor_idx = 0 if i == 0 else (3 * i)
        weight_tensor_idx = 3 * i + 1
        bias_tensor_idx = 3 * i + 2
        out_tensor_idx = 3 * i + 3

        tflite.OperatorStartInputsVector(bld, 3)
        bld.PrependInt32(bias_tensor_idx)
        bld.PrependInt32(weight_tensor_idx)
        bld.PrependInt32(input_tensor_idx)
        in_vec = bld.EndVector()

        tflite.OperatorStartOutputsVector(bld, 1)
        bld.PrependInt32(out_tensor_idx)
        out_vec = bld.EndVector()

        tflite.OperatorStart(bld)
        tflite.OperatorAddOpcodeIndex(bld, 0)
        tflite.OperatorAddInputs(bld, in_vec)
        tflite.OperatorAddOutputs(bld, out_vec)
        tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.FullyConnectedOptions)
        tflite.OperatorAddBuiltinOptions(bld, opt_fc)
        op = tflite.OperatorEnd(bld)
        ops.append(op)

    # 5. Subgraph
    tflite.SubGraphStartInputsVector(bld, 1)
    bld.PrependInt32(0)  # Eingabetensor 0
    sub_in = bld.EndVector()

    final_output_idx = 3 * num_layers
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

    # 6. Model
    tflite.ModelStartOperatorCodesVector(bld, 1)
    bld.PrependUOffsetTRelative(op_code_fc)
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
    num_test_samples = int(eval_cfg.get("num_test_samples", 50))

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
        f.write("#define SNN_MAX_LAYERS 8\n\n")
        f.write("typedef struct {\n")
        f.write("    int input_size;\n")
        f.write("    int hidden_layers_count;\n")
        f.write("    int layer_sizes[SNN_MAX_LAYERS + 1];\n")
        f.write("    int num_steps;\n")
        f.write("    float beta;\n")
        f.write("    float threshold;\n")
        f.write("    int output_size;\n")
        f.write("    int neuron_type;\n")
        f.write("    int encoding_type;\n")
        f.write("} snn_config_t;\n\n")
        f.write("extern const int g_model_type;\n")
        f.write("extern const char* const g_model_name;\n\n")
        f.write("// ANN-Spezifisch (TensorFlow Lite Micro FlatBuffer)\n")
        f.write("extern const unsigned char g_model[];\n")
        f.write("extern const int g_model_len;\n\n")
        f.write("// SNN-Spezifisch (LIF Neuronen Parameter & Gewichte)\n")
        f.write("extern const snn_config_t g_snn_config;\n")
        f.write("extern const float* const g_snn_weights[];\n")
        f.write("extern const float* const g_snn_biases[];\n\n")
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
            # Dummy SNN Symbole
            f.write("const snn_config_t g_snn_config = {0, 0, {0}, 0, 0.0f, 0.0f, 0, 0, 0};\n")
            f.write("const float* const g_snn_weights[] = {0};\n")
            f.write("const float* const g_snn_biases[] = {0};\n")
        elif model_type == "snn":
            # SNN Gewichte und Biases
            fcs = model.fcs
            num_layers = len(fcs)
            layer_sizes = [input_size] + model.hidden_layers + [output_size]

            for l_idx, fc in enumerate(fcs):
                w = fc.weight.detach().cpu().numpy().astype(np.float32)  # shape: (out_features, in_features)
                b = fc.bias.detach().cpu().numpy().astype(np.float32)    # shape: (out_features,)

                f.write(f"// SNN Schicht {l_idx}: {w.shape[1]} -> {w.shape[0]}\n")
                f.write(f"static const float s_snn_w{l_idx}[{w.size}] = {{\n")
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

                f.write(f"static const float s_snn_b{l_idx}[{b.size}] = {{\n  ")
                f.write(", ".join(f"{val:.7f}f" for val in b))
                f.write("\n};\n\n")

            f.write("const float* const g_snn_weights[] = {\n")
            for l_idx in range(num_layers):
                f.write(f"    s_snn_w{l_idx},\n")
            f.write("};\n\n")

            f.write("const float* const g_snn_biases[] = {\n")
            for l_idx in range(num_layers):
                f.write(f"    s_snn_b{l_idx},\n")
            f.write("};\n\n")

            n_type_code = 1 if getattr(model, "neuron_type", "lif") == "if" else 0
            enc_code = get_encoding_code(getattr(model, "encoding_type", "direct"))
            layer_sizes_str = ", ".join(str(s) for s in layer_sizes)
            f.write("const snn_config_t g_snn_config = {\n")
            f.write(f"    .input_size = {input_size},\n")
            f.write(f"    .hidden_layers_count = {len(model.hidden_layers)},\n")
            f.write(f"    .layer_sizes = {{{layer_sizes_str}}},\n")
            f.write(f"    .num_steps = {model.num_steps},\n")
            f.write(f"    .beta = {model.beta:.4f}f,\n")
            f.write(f"    .threshold = {model.threshold:.4f}f,\n")
            f.write(f"    .output_size = {output_size},\n")
            f.write(f"    .neuron_type = {n_type_code},\n")
            f.write(f"    .encoding_type = {enc_code},\n")
            f.write("};\n")

    # 3. mnist_test_data.h (Testbilder und Labels)
    test_header_path = out_path / "mnist_test_data.h"
    with open(test_header_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define MNIST_TEST_SUBSET {num_test_samples}\n")
        f.write(f"#define MNIST_IMAGE_SIZE {input_size}\n")
        f.write(f"#define MNIST_NUM_OUTPUTS {output_size}\n\n")

        # Labels
        labels = [test_dataset[i][1] for i in range(num_test_samples)]
        f.write("static const uint8_t mnist_test_labels[MNIST_TEST_SUBSET] = {\n  ")
        f.write(", ".join(str(l) for l in labels))
        f.write("\n};\n\n")

        # Images
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


def get_model_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extrahiert alle konfigurierten Modelle aus der Konfiguration.
    Unterstützt sowohl 'models:' (Liste mehrerer Modelle) als auch 'model:' (einzelnes Modell).
    """
    models_list = config.get("models")
    resolved_models = []

    global_training = dict(config.get("training", {}))
    global_evaluation = dict(config.get("evaluation", {}))
    global_esp32 = dict(config.get("esp32", {}))
    global_joulescope = dict(config.get("joulescope", {}))

    if isinstance(models_list, list) and len(models_list) > 0:
        for idx, m in enumerate(models_list):
            if not isinstance(m, dict):
                continue
            m_type = str(m.get("type", "ann")).lower()
            hidden_layers = parse_hidden_layers(m.get("hidden_layers", [64]))
            name = m.get("name")
            if not name:
                layer_str = "_".join(str(h) for h in hidden_layers) if hidden_layers else "linear"
                prefix = "SNN" if m_type == "snn" else "ANN"
                name = f"{prefix}_{layer_str}"

            input_size = int(m.get("input_size", config.get("model", {}).get("input_size", 784)))
            output_size = int(m.get("output_size", config.get("model", {}).get("output_size", 10)))

            model_dict = {
                "type": m_type,
                "input_size": input_size,
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

            model_training = dict(global_training)
            for k in ("epochs", "batch_size", "learning_rate", "optimizer", "data_dir", "random_seed"):
                if k in m:
                    model_training[k] = m[k]

            model_evaluation = dict(global_evaluation)
            for k in ("num_test_samples",):
                if k in m:
                    model_evaluation[k] = m[k]

            resolved_models.append({
                "name": name,
                "type": m_type,
                "model": model_dict,
                "training": model_training,
                "evaluation": model_evaluation,
                "esp32": global_esp32,
                "joulescope": global_joulescope,
            })
    else:
        single_m = config.get("model", {})
        m_type = str(single_m.get("type", "ann")).lower()
        hidden_layers = parse_hidden_layers(single_m.get("hidden_layers", [64]))
        name = single_m.get("name")
        if not name:
            layer_str = "_".join(str(h) for h in hidden_layers) if hidden_layers else "linear"
            prefix = "SNN" if m_type == "snn" else "ANN"
            name = f"{prefix}_{layer_str}"

        model_dict = {
            "type": m_type,
            "input_size": int(single_m.get("input_size", 784)),
            "hidden_layers": hidden_layers,
            "output_size": int(single_m.get("output_size", 10)),
        }
        if m_type == "snn":
            model_dict["neuron_type"] = str(single_m.get("neuron_type", single_m.get("neuron", "lif"))).lower()
            model_dict["encoding_type"] = str(single_m.get("encoding_type", single_m.get("encoding", "direct"))).lower()
            model_dict["num_steps"] = int(single_m.get("num_steps", single_m.get("steps", 10)))
            model_dict["beta"] = float(single_m.get("beta", 0.9))
            model_dict["threshold"] = float(single_m.get("threshold", 1.0))
        else:
            model_dict["activation"] = str(single_m.get("activation", "relu"))

        resolved_models.append({
            "name": name,
            "type": m_type,
            "model": model_dict,
            "training": global_training,
            "evaluation": global_evaluation,
            "esp32": global_esp32,
            "joulescope": global_joulescope,
        })

    return resolved_models


def export_model_entry_to_main(model_entry: Dict[str, Any], output_dir: str = "main"):
    """
    Schreibt die C/C++ Header- und Quelldateien für ein einzelnes trainiertes Modell nach output_dir.
    """
    export_c_files(
        model_entry=model_entry,
        output_dir=output_dir,
    )


def train_all_models(
    config_or_path: Union[str, Path, Dict[str, Any]] = None,
    epochs: int = None,
    num_test_samples: int = None,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Trainiert alle in der Konfiguration definierten Modelle nacheinander
    und erzeugt für jedes Modell die entsprechenden Parameter / FlatBuffer.
    """
    if isinstance(config_or_path, dict):
        config = config_or_path
    else:
        config = load_config(config_or_path)

    model_configs = get_model_configs(config)
    total_models = len(model_configs)

    print("\n" + "=" * 70)
    print(f">>> PHASE 1: TRAINING ALLER MODELLE ({total_models} Modell{'e' if total_models > 1 else ''}) <<<")
    print("=" * 70)

    trained_models = []
    for idx, m_cfg in enumerate(model_configs, start=1):
        if epochs is not None:
            m_cfg["training"]["epochs"] = epochs
        if num_test_samples is not None:
            m_cfg["evaluation"]["num_test_samples"] = num_test_samples

        m_type = m_cfg.get("type", "ann").lower()
        print(f"\n--- [{idx}/{total_models}] Trainiere Modell ({m_type.upper()} mit {'SpikingJelly' if m_type == 'snn' else 'PyTorch'}): '{m_cfg['name']}' ---")
        model, test_dataset, test_acc, test_loss = train_model(m_cfg, device=device)

        samples_cnt = int(m_cfg.get("evaluation", {}).get("num_test_samples", 50))
        subset_acc, subset_corr, subset_tot = evaluate_subset(model, test_dataset, num_samples=samples_cnt, device=device)

        tflite_bytes = export_pytorch_to_tflite(model)
        model_size = len(tflite_bytes)

        trained_models.append({
            "name": m_cfg["name"],
            "type": m_type,
            "model": model,
            "tflite_bytes": tflite_bytes,
            "test_dataset": test_dataset,
            "test_acc": test_acc,
            "subset_test_acc": subset_acc,
            "subset_correct": subset_corr,
            "subset_total": subset_tot,
            "test_loss": test_loss,
            "config": m_cfg,
            "structure": model.get_structure_summary(),
            "model_size_bytes": model_size,
            "num_test_samples": samples_cnt,
        })

    print("\n" + "=" * 70)
    print(f">>> PHASE 1 ABGESCHLOSSEN: {len(trained_models)} Modell{'e' if len(trained_models) > 1 else ''} erfolgreich trainiert! <<<")
    print("=" * 70)
    for m in trained_models:
        print(f"  * {m['name']:<20} | Typ: {m['type'].upper():<4} | Test-Acc (10k): {m['test_acc']:.2f}% | Test-Acc (ESP32-Subset {m['num_test_samples']}): {m['subset_test_acc']:.2f}% ({m['subset_correct']}/{m['subset_total']}) | Größe: {m['model_size_bytes']:,} B")
    print("=" * 70 + "\n")

    return trained_models


def train_and_export(
    config_path: Union[str, Path] = None,
    output_dir: str = "main",
    epochs: int = None,
    num_test_samples: int = None,
) -> Tuple[Union[ConfigurableANN, ConfigurableSpikingJellySNN], bytes]:
    """
    Hauptfunktion für Einzelausführung: Lädt Konfiguration, trainiert alle Modelle und exportiert das erste Modell.
    """
    trained_models = train_all_models(
        config_or_path=config_path,
        epochs=epochs,
        num_test_samples=num_test_samples,
    )
    if trained_models:
        export_model_entry_to_main(trained_models[0], output_dir=output_dir)
        return trained_models[0]["model"], trained_models[0]["tflite_bytes"]
    raise RuntimeError("Keine Modelle zum Trainieren gefunden.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST ANN Training & TFLite Micro Export")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Pfad zur Konfigurationsdatei (YAML oder JSON, Standard: config.yaml / config.json)",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=None,
        help="Überschreibt die Anzahl der Trainingsepochen",
    )
    parser.add_argument(
        "-n", "--test-samples",
        type=int,
        default=None,
        help="Überschreibt die Anzahl der Test-Samples für den ESP32",
    )
    args = parser.parse_args()

    train_and_export(
        config_path=args.config,
        epochs=args.epochs,
        num_test_samples=args.test_samples,
    )
