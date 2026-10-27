"""Feed-forward and compact Transformer ANN definitions."""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import compute_conv_layer_dims, parse_conv_layers, parse_hidden_layers

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
                    kernel_size=(cl["kernel_h"], cl["kernel_w"]),
                    stride=cl["stride"],
                    padding=cl["padding"],
                    bias=True,
                ))
                if self.activation_name == "relu":
                    mods.append(nn.ReLU())
                if cl["pool_h"] > 1 or cl["pool_w"] > 1:
                    if cl["pool_type"] == "avg":
                        mods.append(nn.AvgPool2d((cl["pool_h"], cl["pool_w"]), stride=(cl["pool_stride_h"], cl["pool_stride_w"])))
                    else:
                        mods.append(nn.MaxPool2d((cl["pool_h"], cl["pool_w"]), stride=(cl["pool_stride_h"], cl["pool_stride_w"])))
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
                p_str = f",p{cl['pool_h']}x{cl['pool_w']}" if cl['pool_h'] > 1 or cl['pool_w'] > 1 else ""
                parts.append(f"Conv({cl['in_channels']}->{cl['out_channels']},{cl['kernel_h']}x{cl['kernel_w']}{p_str})")
            parts.append(f"Flat({self.flat_size})")
        else:
            parts.append(str(self.input_size))

        for h in self.hidden_layers:
            parts.append(str(h))
        parts.append(str(self.output_size))

        act_str = f" mit {self.activation_name.upper()}" if (self.hidden_layers or self.has_conv) else ""
        return f"ANN [{' -> '.join(parts)}]{act_str}"

class CompactTransformerANN(nn.Module):
    """Kleiner Transformer für 16x16-Mel-Spektrogramme (Zeitfenster als Tokens)."""
    def __init__(self, input_size: int, output_size: int, in_channels: int,
                 in_height: int, in_width: int, embed_dim: int = 32,
                 num_heads: int = 4, num_layers: int = 2,
                 feedforward_dim: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("transformer.embed_dim muss durch num_heads teilbar sein")
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        self.in_height = int(in_height)
        self.in_width = int(in_width)
        self.output_size = int(output_size)
        self.hidden_layers = [int(embed_dim)]
        self.activation_name = "gelu"
        self.has_conv = False
        self.is_compact_transformer = True
        self.deployable = True
        self.input_projection = nn.Linear(self.in_height, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.in_width, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, output_size)
        self.fc_layers = nn.ModuleList([self.classifier])
        self.dropout_rate = float(dropout_rate)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.in_channels, self.in_height, self.in_width)
        if x.dim() == 4:
            x = x.mean(dim=1)
        if x.dim() != 3 or x.size(1) != self.in_height or x.size(2) != self.in_width:
            raise ValueError(f"Transformer erwartet (Batch, {self.in_height}, {self.in_width}), erhalten: {tuple(x.shape)}")
        # Mel-Spektrogramm: Achse 0 = Frequenzbänder, Achse 1 = Zeitfenster.
        tokens = self.input_projection(x.transpose(1, 2)) + self.position_embedding
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        return self.classifier(pooled)

    def get_structure_summary(self) -> str:
        return (f"ANN CompactTransformer [{self.in_width} Zeit-Tokens -> "
                f"{self.input_projection.out_features} Embedding -> "
                f"{len(self.encoder.layers)} Encoder-Blöcke -> {self.output_size}]")
