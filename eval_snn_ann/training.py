"""Training loops, checkpoint caching, and validation helpers."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import DEFAULT_CONFIG
from .data.hub import DatasetInfo, get_dataset, normalize_dataset_name
from .models import (
    ConfigurableANN, ConfigurableSpikingJellySNN, HAS_SPIKINGJELLY,
    build_model_from_config, transfer_ann_to_snn,
)
from .models.snn import functional
from .optimization import evaluate_model, get_model_optimization_config, optimize_model

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
        # Bump when cached dataset preprocessing changes without changing its
        # public dataset name or tensor dimensions.
        "dataset_pipeline_version": (
            1 if str(dataset_name).lower() == "physionet_eeg_mi_csp_temporal"
            else (4 if str(dataset_name).lower() == "physionet_eeg_mi_temporal"
                  else (2 if str(dataset_name).lower() == "speech_commands_spikcommander" else 1))
        ),
        "conv_layers": canonical_conv,
        "hidden_layers": hidden_layers,
        "architecture": str(model_cfg.get("architecture", "standard")).lower(),
        "transformer": model_cfg.get("transformer", {}),
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

    architecture = str(model_cfg.get("architecture", "standard")).lower()
    if architecture in ("spikcommander", "spik_commander"):
        signature["spikcommander"] = {
            "transformer": copy.deepcopy(model_cfg.get("transformer", {})),
            "beta": float(model_cfg.get("beta", 0.9)),
            "threshold": float(model_cfg.get("threshold", 1.0)),
            "dropout_rate": float(model_cfg.get("dropout_rate", 0.1)),
        }
    elif architecture in ("spiking_transformer", "spikingtransformer"):
        signature["spiking_transformer"] = {
            "transformer": copy.deepcopy(model_cfg.get("transformer", {})),
            "num_steps": int(model_cfg.get("num_steps", 8)),
            "beta": float(model_cfg.get("beta", 0.5)),
            "threshold": float(model_cfg.get("threshold", 1.0)),
            "input_scale": float(model_cfg.get("input_scale", 1.0)),
            "distillation": copy.deepcopy(model_cfg.get("distillation", {})),
            "ann_checkpoint": model_cfg.get("ann_checkpoint"),
        }
    elif model_type == "snn":
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
            "temporal_input": bool(model_cfg.get("temporal_input", False)),
            "input_scale": float(model_cfg.get("input_scale", 1.0)),
            "output_readout": str(model_cfg.get("output_readout", "spike")).lower(),
            "beta": float(model_cfg.get("beta", 0.9)) if neuron_type == "lif" else 1.0,
            "threshold": float(model_cfg.get("threshold", 1.0)),
            "ann_checkpoint": model_cfg.get("ann_checkpoint"),
        }
    else:
        signature["ann_params"] = {
            "activation": str(model_cfg.get("activation", "relu")).lower(),
        }

    signature["dropout_rate"] = float(model_cfg.get("dropout_rate", model_cfg.get("dropout", 0.0)))

    sig_json = json.dumps(signature, sort_keys=True)
    sig_hash = hashlib.sha256(sig_json.encode("utf-8")).hexdigest()
    return signature, sig_hash

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
    selection_dataset = dataset_info.validation_dataset or test_dataset
    selection_label = "Validierung" if dataset_info.validation_dataset is not None else "Test"
    in_channels = int(dataset_info.in_channels)
    in_height = int(dataset_info.in_height)
    in_width = int(dataset_info.in_width)
    input_size = int(dataset_info.input_size)
    output_size = int(dataset_info.num_classes)

    signature, sig_hash = compute_model_signature(config, dataset_info)
    # Ein SNN-Transformer hängt von den ANN-Startgewichten ab. Der Name des
    # Checkpoints allein reicht für den Cache nicht: derselbe ANN-Checkpoint
    # kann mit neuen Gewichten überschrieben worden sein.
    if model_type == "snn" and str(model_cfg.get("architecture", "")).lower() in (
        "spiking_transformer", "spikingtransformer"
    ):
        ann_model_name = model_cfg.get(
            "ann_checkpoint", "ANN_KWS_SpeechCommandsV02_35Words_CompactTransformer"
        )
        ann_checkpoint_path = os.path.join(resolved_save_dir, f"{ann_model_name}.pt")
        if os.path.isfile(ann_checkpoint_path):
            digest = hashlib.sha256()
            with open(ann_checkpoint_path, "rb") as checkpoint_file:
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            signature["ann_checkpoint_sha256"] = digest.hexdigest()
        else:
            signature["ann_checkpoint_sha256"] = "missing"
        sig_hash = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode("utf-8")
        ).hexdigest()

    model = build_model_from_config(config, dataset_info, device=device)

    # Transfer Learning: denselben ANN-Transformer als Initialisierung für den
    # Spiking-Transformer verwenden; LIF-Zustände selbst haben keine Gewichte.
    if model_type == "snn" and getattr(model, "is_spiking_transformer", False) and not getattr(model, "is_spikcommander", False):
        ann_model_name = model_cfg.get(
            "ann_checkpoint", "ANN_KWS_SpeechCommandsV02_35Words_CompactTransformer"
        )
        ann_checkpoint_path = os.path.join(resolved_save_dir, f"{ann_model_name}.pt")
        if not os.path.isfile(ann_checkpoint_path):
            raise FileNotFoundError(
                f"ANN-Transformer-Checkpoint '{ann_checkpoint_path}' fehlt; "
                "der Spiking Transformer wird nicht zufällig initialisiert."
            )
        ann_checkpoint = torch.load(ann_checkpoint_path, map_location=device, weights_only=False)
        checkpoint_dataset = normalize_dataset_name(ann_checkpoint.get("dataset", ""))
        if checkpoint_dataset != dataset_name:
            raise ValueError(
                f"ANN-Checkpoint-Datensatz '{checkpoint_dataset}' passt nicht zum SNN-Datensatz "
                f"'{dataset_name}'."
            )
        checkpoint_model_type = str(ann_checkpoint.get("model_type", "ann")).lower()
        if checkpoint_model_type != "ann":
            raise ValueError(
                f"Erwartet wird ein ANN-Checkpoint, gefunden wurde model_type='{checkpoint_model_type}'."
            )
        ann_state = ann_checkpoint.get("state_dict", {})
        target_state = model.state_dict()
        compatible = {
            key: value for key, value in ann_state.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        parameter_keys = dict(model.named_parameters())
        copied_parameters = sum(
            parameter_keys[key].numel() for key in compatible if key in parameter_keys
        )
        total_parameters = sum(parameter.numel() for parameter in parameter_keys.values())
        if copied_parameters != total_parameters:
            missing_parameters = sorted(set(parameter_keys) - set(compatible))
            raise ValueError(
                "ANN-Transformer-Gewichtstransfer unvollständig: "
                f"{copied_parameters:,}/{total_parameters:,} Parameter; "
                f"fehlend oder inkompatibel: {missing_parameters}"
            )
        incompatible = model.load_state_dict(compatible, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unerwartete ANN-Checkpoint-Gewichte: {incompatible.unexpected_keys}")
        print(
            f"[Transfer] Vollständig: {copied_parameters:,}/{total_parameters:,} Parameter "
            f"({len(compatible)} Tensoren) vom ANN-Transformer '{ann_model_name}' übertragen.\n"
        )

    # Transfer Learning: ANN -> konventionelles SNN.
    elif model_type == "snn":
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

    teacher_model = None
    distill_cfg = model_cfg.get("distillation", {})
    distill_enabled = (
        model_type == "snn"
        and getattr(model, "is_spiking_transformer", False)
        and not getattr(model, "is_spikcommander", False)
        and bool(distill_cfg.get("enabled", False))
    )
    distill_alpha = float(distill_cfg.get("alpha", 0.5)) if distill_enabled else 0.0
    distill_temperature = float(distill_cfg.get("temperature", 2.0)) if distill_enabled else 1.0
    if distill_enabled:
        if not 0.0 <= distill_alpha <= 1.0:
            raise ValueError("model.distillation.alpha muss zwischen 0 und 1 liegen.")
        if distill_temperature <= 0.0:
            raise ValueError("model.distillation.temperature muss größer als 0 sein.")
        transformer_cfg = model_cfg.get("transformer", {})
        teacher_model = CompactTransformerANN(
            input_size=input_size,
            output_size=output_size,
            in_channels=in_channels,
            in_height=in_height,
            in_width=in_width,
            embed_dim=int(transformer_cfg.get("embed_dim", 32)),
            num_heads=int(transformer_cfg.get("num_heads", 4)),
            num_layers=int(transformer_cfg.get("num_layers", 2)),
            feedforward_dim=int(transformer_cfg.get("feedforward_dim", 64)),
            dropout_rate=float(model_cfg.get("dropout_rate", model_cfg.get("dropout", 0.1))),
        ).to(device)
        teacher_checkpoint = torch.load(ann_checkpoint_path, map_location=device, weights_only=False)
        teacher_model.load_state_dict(teacher_checkpoint["state_dict"], strict=True)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)

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
        if getattr(model, "temporal_input", False):
            print(f"  - Sequenz-Eingabe:     {model.flat_size} Merkmale pro Zeitschritt, State-Reset nach dem Sample")
            print(f"  - Eingangsverstärkung: {getattr(model, 'input_scale', 1.0):g}x")
        if model.neuron_type == "lif":
            print(f"  - Membran-Decay (Beta):{model.beta}")
        print(f"  - Schwellenwert (Thresh):{model.threshold}")
    else:
        print(f"  - Aktivierung:         {model.activation_name.upper()}")
    print(f"  - Epochen:             {epochs}")
    print(f"  - Batch-Größe:         {batch_size}")
    print(f"  - Lernrate (LR):       {lr}")
    print(f"  - Optimierer:          {optimizer_name.upper()}")
    if distill_enabled:
        print(
            f"  - ANN-Lehrermodell:    aktiv (α={distill_alpha:.2f}, "
            f"Temperatur={distill_temperature:g})"
        )
    if early_stopping_enabled:
        print(f"  - Early Stopping:      Aktiviert (Patience: {patience}, Min-Delta: {min_delta}, Metrik: {monitor}, Best-Weights: {restore_best_weights})")
    else:
        print(f"  - Early Stopping:      Deaktiviert (Best-Weights: {restore_best_weights})")
    print("=" * 65 + "\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(selection_dataset, batch_size=128, shuffle=False)

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
            hard_loss = criterion(output, target)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_output = teacher_model(data)
                soft_loss = F.kl_div(
                    F.log_softmax(output / distill_temperature, dim=1),
                    F.softmax(teacher_output / distill_temperature, dim=1),
                    reduction="batchmean",
                ) * (distill_temperature ** 2)
                loss = (1.0 - distill_alpha) * hard_loss + distill_alpha * soft_loss
            else:
                loss = hard_loss
            loss.backward()
            optimizer.step()

            # Train-Verlust als überwachten Cross-Entropy-Wert ausgeben; die
            # optionale KD-Komponente wird separat zum Optimieren addiert.
            running_loss += hard_loss.item()
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
        test_acc = 100.0 * correct / len(selection_dataset)
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

        print(f"  Epoche {epoch:02d}/{epochs:02d} | Train-Verlust: {avg_loss:.4f} | {selection_label}-Verlust: {test_loss:.4f} | {selection_label}-Genauigkeit: {test_acc:.2f}%{best_tag}")

        scheduler.step()

        if early_stopping_enabled and patience_counter >= patience:
            print(f"[Early Stopping] Vorzeitiger Abbruch in Epoche {epoch}/{epochs}: Keine Verbesserung der Metrik '{monitor}' seit {patience} Epochen (Bester Stand in Epoche {best_epoch}: {selection_label}-Acc: {best_test_acc:.2f}%, {selection_label}-Verlust: {best_test_loss:.4f}).")
            break

    if best_state_dict is not None and restore_best_weights:
        model.load_state_dict(best_state_dict)
        test_acc = best_test_acc
        avg_loss = best_test_loss
        print(f"[PyTorch] Bestes Modell aus Epoche {best_epoch} für Export und Inferenz aktiviert ({selection_label}-Genauigkeit: {best_test_acc:.2f}%, {selection_label}-Verlust: {best_test_loss:.4f}).")

    print(f"[PyTorch] Training für '{model_name}' abgeschlossen (beste {selection_label}-Genauigkeit: {test_acc:.2f}%).\n")

    # ---------------------------------------------------------
    # Post-Training Optimierung (Pruning, Quantisierung, Clustering)
    # ---------------------------------------------------------
    model, opt_info = optimize_model(
        model=model,
        config=config,
        test_dataset=selection_dataset,
        train_dataset=train_dataset,
        device=device,
    )
    if opt_info.get("applied", False):
        test_acc = float(opt_info.get("post_acc", test_acc))
        avg_loss = float(opt_info.get("post_loss", avg_loss))

    if dataset_info.validation_dataset is not None:
        test_acc, avg_loss = evaluate_model(model, test_dataset, device=device)
        print(f"[PyTorch] Genauigkeit auf dem unangetasteten {dataset_info.name}-Testsplit: {test_acc:.2f}%.")

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
