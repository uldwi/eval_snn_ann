"""Model construction and ANN-to-SNN parameter transfer."""
from __future__ import annotations
import copy
from typing import Any, Dict, Union
import torch
from torch import nn
from ..config import parse_conv_layers, parse_hidden_layers
from ..data.hub import DatasetInfo
from .ann import ConfigurableANN, CompactTransformerANN
from .snn import ConfigurableSpikingJellySNN, CompactSpikingTransformer, SpikCommander

def transfer_ann_to_snn(
    ann_model: ConfigurableANN,
    snn_model: ConfigurableSpikingJellySNN,
) -> bool:
    """
    Transfer weights from an ANN to an SNN with the same architecture.
    Copy convolutional and fully connected weights and biases.
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
                    print(f"[Transfer] Warning: incompatible Conv layer {i} shapes: ANN {ann_conv.weight.shape} vs SNN {snn_conv.weight.shape}")
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
                print(f"[Transfer] Warning: incompatible FC layer {i} shapes: ANN {ann_fc.weight.shape} vs SNN {snn_fc.weight.shape}")
                return False

        print("[Transfer] Successfully transferred all weights from ANN to SNN.")
        return True
    except Exception as e:
        print(f"[Transfer] Error: {e}")
        return False

def build_model_from_config(
    config: Dict[str, Any],
    dataset_info: DatasetInfo,
    device: str = "cpu",
) -> Union[ConfigurableANN, ConfigurableSpikingJellySNN]:
    """
    Build a ConfigurableANN or ConfigurableSpikingJellySNN from its configuration.
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

    architecture = str(model_cfg.get("architecture", "standard")).lower()
    if architecture in ("spikcommander", "spik_commander"):
        transformer_cfg = model_cfg.get("transformer", {})
        model = SpikCommander(
            input_size=input_size, output_size=output_size,
            in_channels=in_channels, in_height=in_height, in_width=in_width,
            embed_dim=int(transformer_cfg.get("embed_dim", 64)),
            num_heads=int(transformer_cfg.get("num_heads", 4)),
            beta=float(model_cfg.get("beta", 0.9)),
            threshold=float(model_cfg.get("threshold", 1.0)),
            dropout_rate=float(model_cfg.get("dropout_rate", 0.1)),
        ).to(device)
    elif architecture in ("spiking_transformer", "spikingtransformer"):
        transformer_cfg = model_cfg.get("transformer", {})
        model = CompactSpikingTransformer(
            input_size=input_size, output_size=output_size,
            in_channels=in_channels, in_height=in_height, in_width=in_width,
            embed_dim=int(transformer_cfg.get("embed_dim", 32)),
            num_heads=int(transformer_cfg.get("num_heads", 4)),
            num_layers=int(transformer_cfg.get("num_layers", 2)),
            feedforward_dim=int(transformer_cfg.get("feedforward_dim", 64)),
            num_steps=int(model_cfg.get("num_steps", 8)),
            beta=float(model_cfg.get("beta", 0.5)),
            threshold=float(model_cfg.get("threshold", 1.0)),
            input_scale=float(model_cfg.get("input_scale", 1.0)),
        ).to(device)
    elif model_type == "snn":
        neuron_type = str(model_cfg.get("neuron_type", model_cfg.get("neuron", "lif"))).lower()
        encoding_type = str(model_cfg.get("encoding_type", model_cfg.get("encoding", "direct"))).lower()
        execution_mode = str(model_cfg.get("execution_mode", model_cfg.get("execution", "dense"))).lower()
        num_steps = int(model_cfg.get("num_steps", model_cfg.get("steps", 10)))
        beta = float(model_cfg.get("beta", 0.9)) if neuron_type == "lif" else 1.0
        threshold = float(model_cfg.get("threshold", 1.0))
        dropout_rate = float(model_cfg.get("dropout_rate", model_cfg.get("dropout", 0.0)))
        temporal_input = bool(model_cfg.get("temporal_input", False))
        input_scale = float(model_cfg.get("input_scale", 1.0))
        output_readout = str(model_cfg.get("output_readout", "spike"))
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
            temporal_input=temporal_input,
            input_scale=input_scale,
            output_readout=output_readout,
        ).to(device)
    elif architecture in ("transformer", "compact_transformer"):
        transformer_cfg = model_cfg.get("transformer", {})
        model = CompactTransformerANN(
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
