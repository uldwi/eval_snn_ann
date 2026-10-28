"""Configuration expansion and end-to-end training workflow."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import DEFAULT_CONFIG, load_config, parse_conv_layers, parse_hidden_layers
from .data.hub import AVAILABLE_DATASETS, get_dataset, normalize_dataset_name
from .exporting import export_c_files, export_model_entry_to_main, export_pytorch_to_tflite
from .models import ConfigurableANN, ConfigurableSpikingJellySNN
from .optimization import compute_model_sparsity
from .training import evaluate_subset, train_model

def get_model_configs(config: Dict[str, Any], dataset_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract all configured models.
    Optionally filter the models by dataset.
    """
    models_list = config.get("models")
    resolved_models = []

    global_training = dict(config.get("training", {}))
    global_evaluation = dict(config.get("evaluation", {}))
    global_optimization = dict(config.get("optimization", {})) if isinstance(config.get("optimization"), dict) else {}
    global_microcontroller = dict(config.get("microcontroller", {}))
    global_joulescope = dict(config.get("joulescope", {}))
    
    # Get board-specific settings for the selected target
    mcu_target = global_microcontroller.get("target", "esp32c6")
    mcu_boards = global_microcontroller.get("boards", {})
    mcu_board = mcu_boards.get(mcu_target, {})
    if not mcu_board:
        # Fall back to the legacy esp32 block for backward compatibility
        mcu_board = dict(config.get("esp32", {}))

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

            # When filtering by dataset, use matching models if any exist
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
            if bool(m.get("spec_augment", False)):
                model_dict["spec_augment"] = True
            architecture = str(m.get("architecture", "standard"))
            if m_type != "snn" or architecture in ("spiking_transformer", "spikingtransformer", "spikcommander", "spik_commander"):
                model_dict["architecture"] = architecture
                model_dict["dropout_rate"] = float(m.get("dropout_rate", m.get("dropout", 0.0)))
                if isinstance(m.get("transformer"), dict):
                    model_dict["transformer"] = copy.deepcopy(m["transformer"])
                if isinstance(m.get("distillation"), dict):
                    model_dict["distillation"] = copy.deepcopy(m["distillation"])
            if m_type == "snn":
                model_dict["neuron_type"] = str(m.get("neuron_type", m.get("neuron", "lif"))).lower()
                model_dict["encoding_type"] = str(m.get("encoding_type", m.get("encoding", "direct"))).lower()
                model_dict["num_steps"] = int(m.get("num_steps", m.get("steps", 10)))
                model_dict["beta"] = float(m.get("beta", 0.9))
                model_dict["threshold"] = float(m.get("threshold", 1.0))
                model_dict["temporal_input"] = bool(m.get("temporal_input", False))
                model_dict["input_scale"] = float(m.get("input_scale", 1.0))
                model_dict["output_readout"] = str(m.get("output_readout", "spike")).lower()
                model_dict["execution_mode"] = str(m.get("execution_mode", m.get("execution", "dense")))
                if m.get("ann_checkpoint"):
                    model_dict["ann_checkpoint"] = str(m["ann_checkpoint"])
            else:
                model_dict["activation"] = str(m.get("activation", config.get("model", {}).get("activation", "relu")))

            # Start with global training parameters
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
            if m_ds in ("speech_commands", "speech_commands_spikcommander"):
                model_evaluation["num_test_samples"] = min(model_evaluation.get("num_test_samples", 500), 100)

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
                "microcontroller": global_microcontroller,
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
    architecture = str(single_m.get("architecture", "standard"))
    if m_type != "snn" or architecture in ("spiking_transformer", "spikingtransformer", "spikcommander", "spik_commander"):
        model_dict["architecture"] = architecture
        model_dict["dropout_rate"] = float(single_m.get("dropout_rate", single_m.get("dropout", 0.0)))
        if isinstance(single_m.get("transformer"), dict):
            model_dict["transformer"] = copy.deepcopy(single_m["transformer"])
        if isinstance(single_m.get("distillation"), dict):
            model_dict["distillation"] = copy.deepcopy(single_m["distillation"])
    if m_type == "snn":
        model_dict["neuron_type"] = str(single_m.get("neuron_type", single_m.get("neuron", "lif"))).lower()
        model_dict["encoding_type"] = str(single_m.get("encoding_type", single_m.get("encoding", "direct"))).lower()
        model_dict["num_steps"] = int(single_m.get("num_steps", single_m.get("steps", 10)))
        model_dict["beta"] = float(single_m.get("beta", 0.9))
        model_dict["threshold"] = float(single_m.get("threshold", 1.0))
        model_dict["temporal_input"] = bool(single_m.get("temporal_input", False))
        model_dict["input_scale"] = float(single_m.get("input_scale", 1.0))
        model_dict["output_readout"] = str(single_m.get("output_readout", "spike")).lower()
        model_dict["execution_mode"] = str(single_m.get("execution_mode", single_m.get("execution", "dense")))
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
    if single_ds in ("speech_commands", "speech_commands_spikcommander"):
        model_evaluation["num_test_samples"] = min(model_evaluation.get("num_test_samples", 500), 100)

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
        "microcontroller": global_microcontroller,
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
    skip_evaluation: bool = False,
) -> List[Dict[str, Any]]:
    """
    Train all models defined in the configuration sequentially (batch training, phase 1)
    or load unchanged models from the checkpoint cache and apply any configured post-training optimizations.
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
    print(f"=== BATCH TRAINING: {total_models} model(s) configured ===")
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
        print(f"\n--- [{idx}/{total_models}] Model ({m_type.upper()} with {'SpikingJelly' if m_type == 'snn' else 'PyTorch'}): '{m_cfg['name']}' [Dataset: {ds}, Epochs: {ep}, Batch: {bs}, LR: {lr}] ---")
        model, test_dataset, test_acc, test_loss = train_model(
            m_cfg,
            device=device,
            force_retrain=m_force,
            save_dir=m_save_dir,
        )

        if not skip_evaluation:
            samples_cnt = int(m_cfg.get("evaluation", {}).get("num_test_samples", 500))
            subset_acc, subset_correct, subset_total = evaluate_subset(
                model=model,
                test_dataset=test_dataset,
                num_samples=samples_cnt,
                device=device,
            )
            print(f"[Evaluation] Host accuracy on ESP32 test set ({subset_total} samples): {subset_acc:.2f}% ({subset_correct}/{subset_total} correct).")

            deployable = bool(getattr(model, "deployable", True))
            is_transformer = bool(getattr(model, "is_compact_transformer", False) or getattr(model, "is_spiking_transformer", False))
            tflite_bytes = export_pytorch_to_tflite(model) if m_type == "ann" and deployable and not is_transformer else b""

            # Parameter count, sparsity, and model size
            total_params = sum(p.numel() for p in model.parameters())
            sparsity_info = compute_model_sparsity(model)
            model_size_bytes = (total_params * 4) if m_type == "snn" or not tflite_bytes else len(tflite_bytes)

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
                "deployable": deployable,
            }
            trained_models.append(trained_entry)
        else:
            print("[Evaluation] Skipped (--skip-evaluation).\n")

    print("\n=================================================================")
    print(f"=== BATCH TRAINING COMPLETE: {len(trained_models)} model(s) ready ===")
    print(f"=================================================================\n")
    return trained_models

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
    device: str = "cpu",
    skip_evaluation: bool = False,
) -> Tuple[Union[ConfigurableANN, ConfigurableSpikingJellySNN], bytes]:
    """
    Train all models and export the first model to output_dir.
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
        device=device,
        skip_evaluation=skip_evaluation,
    )
    if trained_models:
        export_model_entry_to_main(trained_models[0], output_dir=output_dir)
        return trained_models[0]["model"], trained_models[0]["tflite_bytes"]
    return None, b""
