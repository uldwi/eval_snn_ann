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

from .config import DEFAULT_CONFIG, parse_conv_layers
from .data.hub import DatasetInfo, get_dataset, normalize_dataset_name
from .models import (
    CompactTransformerANN, ConfigurableANN, ConfigurableSpikingJellySNN, HAS_SPIKINGJELLY,
    build_model_from_config, transfer_ann_to_snn,
)
from .models.snn import functional
from .optimization import evaluate_model, get_model_optimization_config, optimize_model

def compute_model_signature(config: Dict[str, Any], dataset_info: Any = None) -> Tuple[Dict[str, Any], str]:
    """
    Create a canonical signature and SHA256 hash of model architecture,
    dataset, training, and optimization hyperparameters for change detection.
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

    # Preserve existing cache signatures for models without SpecAugment.
    if bool(model_cfg.get("spec_augment", False)):
        signature["spec_augment"] = True

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
    elif architecture in ("spiking_transformer", "spikingtransformer",
                          "spike_driven_transformer", "spikedriven_transformer"):
        signature["spike_driven_transformer" if "spike_driven" in architecture or architecture == "spikedriven_transformer" else "spiking_transformer"] = {
            "transformer": copy.deepcopy(model_cfg.get("transformer", {})),
            "num_steps": int(model_cfg.get("num_steps", 8)),
            "inference_steps": int(model_cfg.get("inference_steps", model_cfg.get("num_steps", 8))),
            "beta": float(model_cfg.get("beta", 0.5)),
            "threshold": float(model_cfg.get("threshold", 1.0)),
            "input_scale": float(model_cfg.get("input_scale", 1.0)),
            "distillation": copy.deepcopy(model_cfg.get("distillation", {})),
            "ann_checkpoint": model_cfg.get("ann_checkpoint"),
        }
        if architecture in ("spike_driven_transformer", "spikedriven_transformer"):
            signature["spike_driven_transformer"]["attention_threshold"] = float(
                model_cfg.get("attention_threshold", 4.0)
            )
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


def _signature_differences(previous: Any, current: Any, prefix: str = "") -> List[str]:
    """Return readable paths and values for fields that differ between signatures."""
    if isinstance(previous, dict) and isinstance(current, dict):
        differences = []
        for key in sorted(set(previous) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in previous:
                differences.append(f"{path}: <missing> -> {current[key]!r}")
            elif key not in current:
                differences.append(f"{path}: {previous[key]!r} -> <missing>")
            else:
                differences.extend(_signature_differences(previous[key], current[key], path))
        return differences
    if previous != current:
        return [f"{prefix}: {previous!r} -> {current!r}"]
    return []


def _normalize_signature_for_comparison(signature: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize harmless schema differences in checkpoints from older releases."""
    normalized = copy.deepcopy(signature)
    dataset_name = normalize_dataset_name(normalized.get("dataset", "mnist"))

    # These fields were added after early checkpoints had already been saved.
    normalized.setdefault("architecture", "standard")
    normalized.setdefault("transformer", {})
    if "dataset_pipeline_version" not in normalized:
        pipeline_version = (
            2 if dataset_name == "speech_commands_spikcommander"
            else (1 if dataset_name == "physionet_eeg_mi_csp_temporal" else 1)
        )
        # Do not treat a missing version as current if the preprocessing has
        # since had an explicit version bump.
        if pipeline_version == 1:
            normalized["dataset_pipeline_version"] = 1

    # Older signatures stored only user-specified convolution values, while
    # newer ones include the parser's expanded defaults.
    if isinstance(normalized.get("conv_layers"), list):
        normalized["conv_layers"] = parse_conv_layers(normalized["conv_layers"])

    # These SNN controls were added to checkpoint signatures after their
    # behavior had already been established as the constructor defaults.
    if normalized.get("type") == "snn" and isinstance(normalized.get("snn_params"), dict):
        snn_params = normalized["snn_params"]
        snn_params.setdefault("temporal_input", False)
        snn_params.setdefault("input_scale", 1.0)
        snn_params.setdefault("output_readout", "spike")
    return normalized


def _signature_matches_except_inference_steps(previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """Allow trained weights to be reused when only the deployed step count changed."""
    old = _normalize_signature_for_comparison(previous)
    new = _normalize_signature_for_comparison(current)
    old_spiking = old.get("spiking_transformer")
    new_spiking = new.get("spiking_transformer")
    if not isinstance(old_spiking, dict) or not isinstance(new_spiking, dict):
        return False

    # Before inference_steps was configurable, inference used num_steps.
    old_steps = old_spiking.pop("inference_steps", old_spiking.get("num_steps"))
    new_steps = new_spiking.pop("inference_steps", new_spiking.get("num_steps"))
    return old_steps != new_steps and old == new


def _calibrate_transformer_early_exit(model, validation_dataset, early_exit_cfg, device: str) -> None:
    """Choose an exit margin on validation data under a bounded accuracy loss."""
    if not (getattr(model, "is_spiking_transformer", False)
            and not getattr(model, "is_spikcommander", False)
            and isinstance(early_exit_cfg, dict)
            and bool(early_exit_cfg.get("enabled", False))):
        return
    if validation_dataset is None:
        print("[Early exit] Disabled: this dataset has no validation split for calibration.")
        model.early_exit_enabled = False
        return

    steps = int(model.inference_steps)
    min_steps = max(1, min(steps, int(early_exit_cfg.get("min_steps", 2))))
    stable_steps = max(1, min(steps, int(early_exit_cfg.get("stable_steps", 2))))
    max_accuracy_drop = max(0.0, float(early_exit_cfg.get("max_accuracy_drop", 0.5)))
    loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
    all_logits, all_targets = [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            logits = model.forward_step_logits(data.to(device)).detach().cpu()
            all_logits.append(logits)
            all_targets.append(target.detach().cpu())
    model.train(was_training)

    if not all_logits:
        print("[Early exit] Disabled: validation split is empty.")
        model.early_exit_enabled = False
        return

    # [steps, samples, classes]
    logits = torch.cat(all_logits, dim=1)
    targets = torch.cat(all_targets, dim=0)
    predictions = logits.argmax(dim=-1)
    top_values = logits.topk(k=2, dim=-1).values
    margins = top_values[..., 0] - top_values[..., 1]
    sample_count = targets.numel()
    final_correct = int((predictions[-1] == targets).sum().item())
    baseline_accuracy = 100.0 * final_correct / max(1, sample_count)

    stable = torch.zeros_like(predictions, dtype=torch.bool)
    for step in range(stable_steps - 1, steps):
        start = step - stable_steps + 1
        stable[step] = (predictions[start:step + 1] == predictions[step:step + 1]).all(dim=0)
    eligible = stable.clone()
    eligible[:min_steps - 1] = False
    candidate_margins = margins[eligible]
    if candidate_margins.numel() == 0:
        model.early_exit_enabled = False
        print("[Early exit] Disabled: no validation observations meet the stability rule.")
        return

    # Quantile thresholds give a compact, deterministic search over exit rates.
    quantiles = torch.linspace(0.0, 1.0, 101)
    thresholds = torch.quantile(candidate_margins, quantiles).unique().tolist()
    thresholds.append(float(candidate_margins.max().item()) + 1e-6)  # no-exit baseline
    best = None
    min_accuracy = baseline_accuracy - max_accuracy_drop
    for threshold in thresholds:
        exit_at = torch.full((sample_count,), steps, dtype=torch.long)
        for step in range(min_steps - 1, steps):
            take = (exit_at == steps) & eligible[step] & (margins[step] >= threshold)
            exit_at[take] = step + 1
        chosen = predictions[exit_at - 1, torch.arange(sample_count)]
        accuracy = 100.0 * float((chosen == targets).sum().item()) / max(1, sample_count)
        mean_steps = float(exit_at.float().mean().item())
        if accuracy + 1e-9 >= min_accuracy and (best is None or mean_steps < best[0]):
            best = (mean_steps, float(threshold), accuracy)

    if best is None:
        model.early_exit_enabled = False
        print("[Early exit] Disabled: no calibrated threshold meets the accuracy constraint.")
        return
    model.early_exit_enabled = True
    model.early_exit_min_steps = min_steps
    model.early_exit_stable_steps = stable_steps
    model.early_exit_margin = best[1]
    model.early_exit_validation_accuracy = best[2]
    model.early_exit_mean_steps = best[0]
    print(
        f"[Early exit] Calibrated on {sample_count} validation samples: margin >= "
        f"{best[1]:.6g}, stable for {stable_steps} steps, min {min_steps}; "
        f"accuracy {best[2]:.2f}% (full-step baseline {baseline_accuracy:.2f}%, "
        f"allowed drop {max_accuracy_drop:.2f} pp), mean steps {best[0]:.2f}/{steps}."
    )

def train_model(
    config: Dict[str, Any],
    device: str = "cpu",
    force_retrain: bool = False,
    save_dir: Optional[str] = None,
) -> Tuple[Union[ConfigurableANN, ConfigurableSpikingJellySNN], Dataset, float, float]:
    """
    Train the configured PyTorch model (ANN or SpikingJelly SNN) on the selected dataset
    or load a trained model from cache if its configuration has not changed.
    Return (model, test_dataset, test_acc, avg_loss).
    """
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    model_type = str(model_cfg.get("type", config.get("type", "ann"))).lower()
    model_name = str(config.get("name", model_cfg.get("name", "model")))

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

    # Load dataset
    dataset_name = model_cfg.get("dataset", config.get("dataset", train_cfg.get("dataset", "mnist")))
    if isinstance(dataset_name, dict):
        dataset_name = dataset_name.get("name", "mnist")
    dataset_name = normalize_dataset_name(dataset_name)

    train_dataset, test_dataset, dataset_info = get_dataset(
        name=dataset_name,
        data_dir=data_dir,
        seed=random_seed if random_seed is not None else 42,
        augment_train=bool(model_cfg.get("spec_augment", False)),
    )
    selection_dataset = dataset_info.validation_dataset or test_dataset
    selection_label = "Validation" if dataset_info.validation_dataset is not None else "Test"
    in_channels = int(dataset_info.in_channels)
    in_height = int(dataset_info.in_height)
    in_width = int(dataset_info.in_width)
    input_size = int(dataset_info.input_size)
    output_size = int(dataset_info.num_classes)

    signature, sig_hash = compute_model_signature(config, dataset_info)
    # An SNN Transformer depends on its ANN initialization weights. The checkpoint
    # path alone is not enough for cache validation: the same ANN checkpoint
    # may have been overwritten with different weights.
    if model_type == "snn" and str(model_cfg.get("architecture", "")).lower() in (
        "spiking_transformer", "spikingtransformer", "spike_driven_transformer",
        "spikedriven_transformer",
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

    # Transfer learning: initialize the spiking Transformer from the same ANN Transformer;
    # LIF states themselves do not have weights.
    if model_type == "snn" and getattr(model, "is_spiking_transformer", False) and not getattr(model, "is_spikcommander", False):
        ann_model_name = model_cfg.get(
            "ann_checkpoint", "ANN_KWS_SpeechCommandsV02_35Words_CompactTransformer"
        )
        ann_checkpoint_path = os.path.join(resolved_save_dir, f"{ann_model_name}.pt")
        if not os.path.isfile(ann_checkpoint_path):
            raise FileNotFoundError(
                f"ANN Transformer checkpoint '{ann_checkpoint_path}' is missing; "
                "the spiking Transformer will not be randomly initialized."
            )
        ann_checkpoint = torch.load(ann_checkpoint_path, map_location=device, weights_only=False)
        checkpoint_dataset = normalize_dataset_name(ann_checkpoint.get("dataset", ""))
        if checkpoint_dataset != dataset_name:
            raise ValueError(
                f"ANN checkpoint dataset '{checkpoint_dataset}' does not match SNN dataset "
                f"'{dataset_name}'."
            )
        checkpoint_model_type = str(ann_checkpoint.get("model_type", "ann")).lower()
        if checkpoint_model_type != "ann":
            raise ValueError(
                f"Expected an ANN checkpoint, found model_type='{checkpoint_model_type}'."
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
                "Incomplete ANN Transformer weight transfer: "
                f"{copied_parameters:,}/{total_parameters:,} parameters; "
                f"missing or incompatible: {missing_parameters}"
            )
        incompatible = model.load_state_dict(compatible, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected ANN checkpoint weights: {incompatible.unexpected_keys}")
        print(
            f"[Transfer] Complete: {copied_parameters:,}/{total_parameters:,} parameters "
            f"({len(compatible)} tensors) from ANN Transformer '{ann_model_name}'.\n"
        )

    # Transfer learning: ANN -> conventional SNN.
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
                    print(f"[Transfer] Initialized SNN '{model_name}' with ANN weights from '{ann_model_name}'\n")
                else:
                    print("[Transfer] Weight transfer failed; initializing SNN randomly.\n")
            except Exception as e:
                print(f"[Transfer] Failed to load ANN checkpoint: {e}\n")
        else:
            print(f"[Transfer] ANN checkpoint '{ann_checkpoint_path}' not found; initializing SNN randomly.\n")

    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model_name)
    checkpoint_path = os.path.join(resolved_save_dir, f"{safe_name}.pt")
    meta_path = os.path.join(resolved_save_dir, f"{safe_name}.json")

    # During the package-layout migration, existing checkpoints may still live
    # beside this module in eval_snn_ann/saved_models while the configured
    # default points to the repository-level saved_models directory.
    cache_checkpoint_path = checkpoint_path
    project_saved_dir = Path(__file__).resolve().parents[1] / "saved_models"
    package_saved_dir = Path(__file__).resolve().parent / "saved_models"
    if (
        not os.path.isfile(cache_checkpoint_path)
        and Path(resolved_save_dir).resolve() == project_saved_dir.resolve()
    ):
        legacy_checkpoint_path = package_saved_dir / f"{safe_name}.pt"
        if legacy_checkpoint_path.is_file():
            cache_checkpoint_path = str(legacy_checkpoint_path)
            print(
                f"[Cache] Found legacy checkpoint for '{model_name}' at "
                f"'{cache_checkpoint_path}'. Checking its configuration hash."
            )

    # ---------------------------------------------------------
    # Check whether a trained model can be loaded from cache.
    # ---------------------------------------------------------
    if not should_force and auto_save and os.path.isfile(cache_checkpoint_path):
        try:
            try:
                checkpoint = torch.load(cache_checkpoint_path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(cache_checkpoint_path, map_location=device)

            cached_hash = checkpoint.get("config_hash")
            cached_signature = checkpoint.get("config_signature")
            cache_signature_matches = (
                isinstance(cached_signature, dict)
                and _normalize_signature_for_comparison(cached_signature)
                == _normalize_signature_for_comparison(signature)
            )
            inference_only_cache_match = (
                isinstance(cached_signature, dict)
                and _signature_matches_except_inference_steps(cached_signature, signature)
            )
            if (
                cached_hash == sig_hash or cache_signature_matches or inference_only_cache_match
            ) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                test_acc = float(checkpoint.get("test_acc", 0.0))
                test_loss = float(checkpoint.get("test_loss", 0.0))
                best_epoch = checkpoint.get("best_epoch", "N/A")

                if inference_only_cache_match:
                    # The training trajectory and weights are unchanged. Refresh
                    # metrics using the new deployed step count without retraining.
                    test_acc, test_loss = evaluate_model(model, test_dataset, device=device)
                    checkpoint.update({
                        "config_signature": signature,
                        "config_hash": sig_hash,
                        "test_acc": float(test_acc),
                        "test_loss": float(test_loss),
                        "timestamp": time.time(),
                    })
                    torch.save(checkpoint, cache_checkpoint_path)
                    cache_meta_path = Path(cache_checkpoint_path).with_suffix(".json")
                    if cache_meta_path.exists():
                        try:
                            meta_data = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            meta_data = {}
                        meta_data.update({
                            "config_signature": signature,
                            "config_hash": sig_hash,
                            "test_acc": float(test_acc),
                            "test_loss": float(test_loss),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        })
                        cache_meta_path.write_text(
                            json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8"
                        )

                print("\n" + "=" * 65)
                print(f">>> [Cache] Using saved model: '{model_name}' ({'SNN' if model_type == 'snn' else 'ANN'}) <<<")
                print(f"  - Checkpoint:           {cache_checkpoint_path}")
                print(f"  - Dataset:              {dataset_info.name} ({dataset_info.domain})")
                print(f"  - Architecture:         {model.get_structure_summary()}")
                if inference_only_cache_match:
                    hash_status = "inference steps changed; weights reused and accuracy refreshed"
                else:
                    hash_status = "unchanged" if cached_hash == sig_hash else "compatible legacy signature"
                print(f"  - Configuration hash:   {sig_hash[:12]} ({hash_status})")
                print(f"  - Accuracy:             {test_acc:.2f}% (epoch {best_epoch})")
                print(f"  - Loss:                 {test_loss:.4f}")
                _calibrate_transformer_early_exit(
                    model, dataset_info.validation_dataset, model_cfg.get("early_exit", {}), device
                )
                print("  -> No retraining needed (cache hit).\n")
                return model, test_dataset, test_acc, test_loss
            else:
                old_h = str(cached_hash)[:12] if cached_hash else "None"
                print(f"\n[Cache] Found saved model '{model_name}', but configuration/hyperparameters changed:")
                print(f"        Previous hash: {old_h} -> New hash: {sig_hash[:12]}")
                if cached_signature:
                    differences = _signature_differences(cached_signature, signature)
                    for difference in differences[:8]:
                        print(f"        Changed setting: {difference}")
                    if len(differences) > 8:
                        print(f"        ... and {len(differences) - 8} more changed setting(s)")
                print(f"        -> Retraining model.")
        except Exception as e:
            print(f"\n[Cache] Could not load saved model '{cache_checkpoint_path}' ({e}).")
            print(f"        -> Retraining model.")

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
            raise ValueError("model.distillation.alpha must be between 0 and 1.")
        if distill_temperature <= 0.0:
            raise ValueError("model.distillation.temperature must be greater than 0.")
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

    # Early stopping configuration
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
    print(f">>> PyTorch Model Configuration: {model_name} ({'SNN' if model_type == 'snn' else 'ANN'}) <<<")
    print(f"  - Dataset:             {dataset_info.name} ({dataset_info.domain})")
    print(f"  - Input dimensions:    {in_channels}x{in_height}x{in_width} ({input_size} values) -> {output_size} classes")
    print(f"  - Architecture:        {model.get_structure_summary()}")
    if getattr(model, "has_conv", False):
        print(f"  - Conv layers:         {len(model.resolved_conv)}")
    print(f"  - Hidden layers:       {model.hidden_layers}")
    if model_type == "snn":
        print(f"  - Neuron type:         {model.neuron_type.upper()}")
        print(f"  - Encoding type:       {model.encoding_type.upper()}")
        if getattr(model, "is_spiking_transformer", False):
            print(f"  - Training time steps: {model.num_steps}")
            print(f"  - Inference time steps: {model.inference_steps}")
        else:
            print(f"  - Time steps:          {model.num_steps}")
        if getattr(model, "temporal_input", False):
            print(f"  - Sequence input:      {model.flat_size} features per time step; state reset after each sample")
            print(f"  - Input scale:         {getattr(model, 'input_scale', 1.0):g}x")
        if model.neuron_type == "lif":
            print(f"  - Membrane decay (beta): {model.beta}")
        print(f"  - Threshold:           {model.threshold}")
    else:
        print(f"  - Activation:          {model.activation_name.upper()}")
    print(f"  - Epochs:              {epochs}")
    print(f"  - Batch size:          {batch_size}")
    print(f"  - Learning rate:       {lr}")
    print(f"  - Optimizer:           {optimizer_name.upper()}")
    if distill_enabled:
        print(
            f"  - ANN teacher model:  enabled (alpha={distill_alpha:.2f}, "
            f"temperature={distill_temperature:g})"
        )
    if bool(model_cfg.get("spec_augment", False)):
        print("  - Training augmentation: SpecAugment (training split only)")
    if early_stopping_enabled:
        print(f"  - Early stopping:      Enabled (patience: {patience}, min delta: {min_delta}, metric: {monitor}, restore best weights: {restore_best_weights})")
    else:
        print(f"  - Early stopping:      Disabled (restore best weights: {restore_best_weights})")
    print("=" * 65 + "\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(selection_dataset, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer_name = optimizer_name.lower()
    print(f"  - Optimizer:           {optimizer_name.upper()}")
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

    print(f"[PyTorch] Starting training for {epochs} epochs...")
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

            # Report supervised cross-entropy as the training loss;
            # add the optional knowledge-distillation component separately during optimization.
            running_loss += hard_loss.item()
            total_batches += 1

        # Validate after every epoch
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

        # Check early stopping conditions
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
            best_tag = " [Best model]"
        else:
            best_tag = f" [Patience: {patience_counter}/{patience}]" if early_stopping_enabled else ""

        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train loss: {avg_loss:.4f} | {selection_label} loss: {test_loss:.4f} | {selection_label} accuracy: {test_acc:.2f}%{best_tag}")

        scheduler.step()

        if early_stopping_enabled and patience_counter >= patience:
            print(f"[Early stopping] Stopped at epoch {epoch}/{epochs}: metric '{monitor}' did not improve for {patience} epochs (best at epoch {best_epoch}: {selection_label} accuracy {best_test_acc:.2f}%, {selection_label} loss {best_test_loss:.4f}).")
            break

    if best_state_dict is not None and restore_best_weights:
        model.load_state_dict(best_state_dict)
        test_acc = best_test_acc
        avg_loss = best_test_loss
        print(f"[PyTorch] Restored best model from epoch {best_epoch} for export and inference ({selection_label} accuracy: {best_test_acc:.2f}%, {selection_label} loss: {best_test_loss:.4f}).")

    print(f"[PyTorch] Training for '{model_name}' complete (best {selection_label} accuracy: {test_acc:.2f}%).\n")

    # ---------------------------------------------------------
    # Post-training optimization (pruning, quantization, clustering)
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

    _calibrate_transformer_early_exit(
        model, dataset_info.validation_dataset, model_cfg.get("early_exit", {}), device
    )

    if dataset_info.validation_dataset is not None:
        test_acc, avg_loss = evaluate_model(model, test_dataset, device=device)
        print(f"[PyTorch] Accuracy on the untouched {dataset_info.name} test split: {test_acc:.2f}%.")

    # ---------------------------------------------------------
    # Automatically save the trained model (checkpoint and metadata)
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
            print(f"[PyTorch] Saved model '{model_name}' to '{checkpoint_path}'.\n")
        except Exception as e:
            print(f"[PyTorch] Warning: Could not save model: {e}", file=sys.stderr)

    return model, test_dataset, test_acc, avg_loss

def evaluate_subset(
    model: Union[ConfigurableANN, ConfigurableSpikingJellySNN],
    test_dataset: Any,
    num_samples: int = 50,
    device: str = "cpu",
) -> Tuple[float, int, int]:
    """
    Evaluate the model on a specified subset of the test dataset.
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
