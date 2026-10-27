"""Model evaluation, pruning, quantization, and clustering."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as torch_prune
from torch.utils.data import DataLoader, Dataset

from .models import ConfigurableANN, ConfigurableSpikingJellySNN, HAS_SPIKINGJELLY
from .models.snn import functional, sj_layer

def get_model_optimization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalize a model optimization configuration.
    Combine global optimization settings with model-specific settings.
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

    # Model-specific overrides
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

def compute_model_sparsity(model: nn.Module) -> Dict[str, Any]:
    """
    Compute detailed sparsity metrics (zero weights) for all layers.
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
    Evaluate a model on the full test dataset and return (test_acc, test_loss).
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
    Apply post-training pruning (L1/magnitude unstructured, structured, random, or threshold)
    to model layers (convolutional and/or fully connected).
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
    Apply post-training quantization (INT8, INT4, FP16) to weights.
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
    Cluster model weights into k discrete centroids (weight sharing/codebook).
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
    Apply configured post-training optimizations (pruning, quantization, and weight clustering)
    to a trained ANN or SNN and measure the resulting sparsity and accuracy.
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

    model_name = str(config.get("name", config.get("model", {}).get("name", "model")))
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

    # Estimate the compression ratio
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
    print(f">>> [Post-training optimization] Model: '{model_name}' ({model_type}) <<<")
    print(f"  - Before optimization: Accuracy: {pre_acc:.2f}%, loss: {pre_loss:.4f}")
    if prune_enabled:
        print(f"  - Pruning:             {prune_res.get('method', 'l1')} (target: {prune_res.get('amount')}) -> {len(prune_res.get('pruned_modules', []))} layers")
    if clust_enabled:
        print(f"  - Weight clustering:   {clust_res.get('num_clusters')} centroids -> {len(clust_res.get('clustered_modules', []))} layers")
    if quant_enabled:
        print(f"  - Quantization:        {quant_res.get('type', 'int8').upper()} ({quant_res.get('mode', 'symmetric')}) -> {len(quant_res.get('quantized_modules', []))} layers")
    print(f"  - After optimization:  Accuracy: {post_acc:.2f}% (delta: {acc_diff:+.2f}%), loss: {post_loss:.4f}")
    print(f"  - Model sparsity:      {post_sparsity['sparsity_pct']:.2f}% zeros ({post_sparsity['zero_weights']:,} of {post_sparsity['total_weights']:,} weights)")
    print(f"  - Estimated compression: about {compression_ratio:.1f}x memory reduction")
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
