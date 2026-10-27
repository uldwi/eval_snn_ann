"""Spiking network, temporal SNN and SpikCommander model definitions."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import compute_conv_layer_dims, parse_conv_layers, parse_hidden_layers

try:
    from spikingjelly.activation_based import neuron, layer as sj_layer, functional, surrogate
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False
    neuron = sj_layer = functional = surrogate = None

class CompactSpikingTransformer(nn.Module):
    """Kleiner spiking Transformer für 16x16-Mel-Spektrogramme.

    Die Zeitachse bildet Tokens. Projektion, Q/K/V, Attention-Residual und
    MLP-Residual verwenden LIF-Neuronen; die Vorhersage mittelt Spike-Raten.
    """
    def __init__(self, input_size: int, output_size: int, in_channels: int,
                 in_height: int, in_width: int, embed_dim: int = 32,
                 num_heads: int = 4, num_layers: int = 2,
                 feedforward_dim: int = 64, num_steps: int = 8,
                 beta: float = 0.5, threshold: float = 1.0,
                 input_scale: float = 1.0):
        super().__init__()
        if not HAS_SPIKINGJELLY:
            raise ImportError("SpikingTransformer benötigt spikingjelly")
        if embed_dim % num_heads:
            raise ValueError("transformer.embed_dim muss durch num_heads teilbar sein")
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        self.in_height = int(in_height)
        self.in_width = int(in_width)
        self.output_size = int(output_size)
        self.hidden_layers = [int(embed_dim)]
        self.has_conv = False
        self.is_compact_transformer = True
        self.is_spiking_transformer = True
        self.deployable = True
        self.neuron_type = "lif"
        self.encoding_type = "direct"
        self.num_steps = int(num_steps)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.input_scale = float(input_scale)
        self.input_projection = nn.Linear(self.in_height, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.in_width, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=feedforward_dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, output_size)
        self.fc_layers = nn.ModuleList([self.classifier])
        self.input_node = self._lif()
        self.q_nodes = nn.ModuleList([self._lif() for _ in range(num_layers)])
        self.k_nodes = nn.ModuleList([self._lif() for _ in range(num_layers)])
        self.v_nodes = nn.ModuleList([self._lif() for _ in range(num_layers)])
        self.attn_nodes = nn.ModuleList([self._lif() for _ in range(num_layers)])
        self.ff_nodes = nn.ModuleList([self._lif() for _ in range(num_layers)])
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def _lif(self):
        return neuron.LIFNode(
            tau=1.0 / max(1e-4, 1.0 - self.beta), decay_input=False,
            v_threshold=self.threshold, v_reset=0.0,
            surrogate_function=surrogate.ATan(), detach_reset=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.in_channels, self.in_height, self.in_width)
        if x.dim() == 4:
            x = x.mean(dim=1)
        if x.dim() != 3 or x.size(1) != self.in_height or x.size(2) != self.in_width:
            raise ValueError(f"SpikingTransformer erwartet (Batch, {self.in_height}, {self.in_width})")

        functional.reset_net(self)
        tokens = (self.input_projection(x.transpose(1, 2)) + self.position_embedding) * self.input_scale
        spike_sum = torch.zeros_like(tokens)
        try:
            for _ in range(self.num_steps):
                cur = self.input_node(tokens)
                for i, layer in enumerate(self.encoder.layers):
                    b, n, d = cur.shape
                    normed = layer.norm1(cur)
                    qkv = F.linear(normed, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
                    q, k, v = qkv.chunk(3, dim=-1)
                    q, k, v = self.q_nodes[i](q), self.k_nodes[i](k), self.v_nodes[i](v)
                    heads = layer.self_attn.num_heads
                    head_dim = d // heads
                    q = q.view(b, n, heads, head_dim).transpose(1, 2)
                    k = k.view(b, n, heads, head_dim).transpose(1, 2)
                    v = v.view(b, n, heads, head_dim).transpose(1, 2)
                    attn = torch.softmax((q @ k.transpose(-2, -1)) * (head_dim ** -0.5), dim=-1) @ v
                    attn = attn.transpose(1, 2).contiguous().view(b, n, d)
                    attn = F.linear(attn, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)
                    cur = self.attn_nodes[i](cur + attn)
                    hidden = F.gelu(F.linear(layer.norm2(cur), layer.linear1.weight, layer.linear1.bias))
                    ff = F.linear(hidden, layer.linear2.weight, layer.linear2.bias)
                    cur = self.ff_nodes[i](cur + ff)
                spike_sum = spike_sum + cur
            pooled = spike_sum.mean(dim=1) / float(self.num_steps)
            return self.classifier(self.norm(pooled))
        finally:
            functional.reset_net(self)

    def get_structure_summary(self) -> str:
        return (f"SNN-SpikingTransformer [{self.in_width} Zeit-Tokens -> "
                f"{self.input_projection.out_features} LIF-Embedding -> "
                f"{len(self.encoder.layers)} spikende Attention/MLP-Blöcke] "
                f"(T={self.num_steps}, β={self.beta:.2f}, θ={self.threshold:.2f}, "
                f"Input×{self.input_scale:g})")

class SpikCommander(nn.Module):
    """SpikCommander-inspired temporal SNN for Mel sequences.

    This follows the paper's main recipe: temporal sequence input, spiking
    depthwise/pointwise embedding, local and global temporal paths, and a
    spike-aware contextual MLP. The compact defaults are sized for this repo.
    """
    def __init__(self, input_size: int, output_size: int, in_channels: int,
                 in_height: int, in_width: int, embed_dim: int = 64,
                 num_heads: int = 4, beta: float = 0.9,
                 threshold: float = 1.0, dropout_rate: float = 0.1):
        super().__init__()
        if not HAS_SPIKINGJELLY:
            raise ImportError("SpikCommander benötigt spikingjelly für Surrogate Gradients")
        if embed_dim % num_heads:
            raise ValueError("transformer.embed_dim muss durch num_heads teilbar sein")
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        self.in_height = int(in_height)
        self.in_width = int(in_width)
        self.output_size = int(output_size)
        self.hidden_layers = [int(embed_dim)]
        self.has_conv = False
        self.is_spiking_transformer = True
        self.is_spikcommander = True
        self.is_compact_transformer = True
        self.deployable = False  # Firmware needs the dedicated temporal architecture.
        self.neuron_type = "lif"
        self.encoding_type = "temporal"
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.num_steps = int(in_width)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.input_projection = nn.Linear(in_height, embed_dim)
        self.embed_pw = nn.Conv1d(embed_dim, embed_dim, 1, bias=False)
        self.embed_dw = nn.Conv1d(embed_dim, embed_dim, 7, padding=3, groups=embed_dim, bias=False)
        self.embed_bn1 = nn.BatchNorm1d(embed_dim)
        self.embed_bn2 = nn.BatchNorm1d(embed_dim)
        self.embed_res = nn.Linear(embed_dim, embed_dim, bias=False)
        self.layers = nn.ModuleList([_SpikCommanderBlock(embed_dim, num_heads, beta, threshold) for _ in range(1)])
        self.dropout = nn.Dropout(float(dropout_rate))
        self.classifier = nn.Linear(embed_dim, output_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_layers = nn.ModuleList([self.classifier])
        self._spike = surrogate.ATan()

    def _lif_sequence(self, current: torch.Tensor) -> torch.Tensor:
        # current is [time, batch, channels]; reset membrane for every sample.
        v = torch.zeros_like(current[0])
        spikes = []
        for x_t in current.unbind(0):
            v = self.beta * v + x_t
            s = self._spike(v - self.threshold)
            v = v - s.detach() * self.threshold
            spikes.append(s)
        return torch.stack(spikes, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.in_channels, self.in_height, self.in_width)
        if x.dim() == 4:
            x = x.mean(dim=1)
        if x.dim() != 3 or x.size(1) != self.in_height or x.size(2) != self.in_width:
            raise ValueError(f"SpikCommander erwartet (Batch, {self.in_height}, {self.in_width}), erhalten: {tuple(x.shape)}")
        seq = self.input_projection(x.transpose(1, 2))  # B,T,D
        z = seq.transpose(1, 2)
        z = self._lif_sequence(self.embed_bn1(self.embed_pw(z)).permute(2, 0, 1)).permute(1, 2, 0)
        local = self._lif_sequence(self.embed_bn2(self.embed_dw(z)).permute(2, 0, 1)).permute(1, 2, 0)
        z = self._lif_sequence((local.permute(0, 2, 1) + self.embed_res(seq)).permute(1, 0, 2))
        for block in self.layers:
            z = block(z, self._spike)
        # Temporal class evidence is accumulated across the full Mel sequence.
        return self.classifier(self.norm(self.dropout(z).mean(dim=0)))

    def get_structure_summary(self) -> str:
        return (f"SNN-SpikCommander [Mel-Sequenz {self.in_width}×{self.in_height} -> "
                f"spikendes Embedding {self.embed_dim} -> 1 Multi-view-Block -> {self.output_size}] "
                f"(β={self.beta:.2f}, θ={self.threshold:.2f})")

class _SpikCommanderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, beta: float, threshold: float):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.global_proj = nn.Linear(dim, dim)
        self.local_proj = nn.Linear(dim, dim)
        self.value_conv = nn.Conv1d(dim, dim, 9, padding=4, groups=dim, bias=False)
        self.attn_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, 4 * dim)
        self.ff_context = nn.Conv1d(4 * dim, 4 * dim, 31, padding=15, groups=4 * dim, bias=False)
        self.ff_out = nn.Linear(4 * dim, dim)
        self.beta, self.threshold = float(beta), float(threshold)

    def _lif(self, x: torch.Tensor, spike_fn) -> torch.Tensor:
        v = torch.zeros_like(x[0])
        out = []
        for x_t in x.unbind(0):
            v = self.beta * v + x_t
            s = spike_fn(v - self.threshold)
            v = v - s.detach() * self.threshold
            out.append(s)
        return torch.stack(out, 0)

    def forward(self, x: torch.Tensor, spike_fn) -> torch.Tensor:
        # x: T,B,D. Global STASA gate + local convolutional/value view.
        t, b, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = self._lif(q, spike_fn), self._lif(k, spike_fn), self._lif(v, spike_fn)
        global_gate = self._lif(self.global_proj((q + k).sum(dim=0, keepdim=True) / max(1, t)), spike_fn)
        global_path = v * global_gate.expand_as(v)
        value_conv = self.value_conv(v.permute(1, 2, 0)).permute(2, 0, 1)
        local_gate = self._lif(self.local_proj(value_conv + q), spike_fn)
        # Per-head local correlation is restricted to a 41-frame neighborhood.
        h, hd = self.heads, d // self.heads
        qh = q.view(t, b, h, hd)
        kh = k.view(t, b, h, hd)
        vh = v.view(t, b, h, hd)
        positions = torch.arange(t, device=x.device)
        local_scores, local_values = [], []
        for offset in range(-20, 21):
            source = positions + offset
            valid = (source >= 0) & (source < t)
            source = source.clamp(0, t - 1)
            key_at_offset = kh.index_select(0, source)
            value_at_offset = vh.index_select(0, source)
            score = (qh * key_at_offset).sum(dim=-1) / (hd ** 0.5)
            local_scores.append(score.masked_fill(~valid[:, None, None], -1e4))
            local_values.append(value_at_offset)
        local_weights = torch.softmax(torch.stack(local_scores, dim=-1), dim=-1)
        local_attn = (torch.stack(local_values, dim=-2) * local_weights[..., None]).sum(dim=-2).reshape(t, b, d)
        attended = self._lif(self.attn_proj(global_path + local_gate * local_attn), spike_fn)
        x = self._lif(x + attended, spike_fn)
        expanded = self._lif(self.ff_in(self.norm2(x)), spike_fn)
        context = self.ff_context(expanded.permute(1, 2, 0)).permute(2, 0, 1)
        refined = self._lif(self.ff_out(self._lif(expanded + context, spike_fn)), spike_fn)
        return self._lif(x + refined, spike_fn)

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
        temporal_input: bool = False,
        input_scale: float = 1.0,
        output_readout: str = "spike",
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
        self.temporal_input = bool(temporal_input)
        self.input_scale = float(input_scale)
        self.output_readout = str(output_readout).lower().strip()
        if self.output_readout not in ("spike", "analog"):
            raise ValueError("output_readout muss 'spike' oder 'analog' sein.")
        if self.input_scale <= 0.0:
            raise ValueError("input_scale muss groesser als 0 sein.")
        if self.temporal_input and conv_layers:
            raise ValueError("temporal_input benötigt derzeit ein SNN ohne Conv-Layer.")
        if self.temporal_input:
            if self.encoding_type != "direct":
                raise ValueError("temporal_input benötigt encoding_type: direct.")
            # Dataset-Tensor liegt als [Kanal, Merkmal, Zeit] vor. Jeder
            # Zeitschritt wird als eigener Vektor in die FC-Schichten gegeben.
            self.num_steps = int(in_width)
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
                    kernel_size=(cl["kernel_h"], cl["kernel_w"]),
                    stride=cl["stride"],
                    padding=cl["padding"],
                    bias=True,
                ))
                self.conv_lif_list.append(create_neuron())
                if cl["pool_h"] > 1 or cl["pool_w"] > 1:
                    if cl["pool_type"] == "avg":
                        self.conv_pool_list.append(sj_layer.AvgPool2d(
                            (cl["pool_h"], cl["pool_w"]),
                            stride=(cl["pool_stride_h"], cl["pool_stride_w"]),
                        ))
                    else:
                        self.conv_pool_list.append(sj_layer.MaxPool2d(
                            (cl["pool_h"], cl["pool_w"]),
                            stride=(cl["pool_stride_h"], cl["pool_stride_w"]),
                        ))
                else:
                    self.conv_pool_list.append(nn.Identity())
            self.flatten = sj_layer.Flatten()
            fc_in = self.flat_size
        else:
            self.resolved_conv = []
            self.flat_size = (self.in_channels * self.in_height) if self.temporal_input else self.input_size
            self.flatten = sj_layer.Flatten()
            fc_in = self.flat_size

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
            if self.temporal_input:
                x_seq = x.view(batch_size, self.in_channels, self.in_height, self.in_width)
                x_seq = x_seq.reshape(batch_size, self.in_channels * self.in_height, self.in_width)
                x_seq = x_seq.permute(2, 0, 1).contiguous()
                x_seq = x_seq * self.input_scale
            else:
                x_flat = x.view(batch_size, -1)
                x_seq = encode_input(x_flat, self.encoding_type, self.num_steps)
            cur = x_seq

        for i, (fc, lif) in enumerate(zip(self.fcs, self.lif_layers)):
            cur = fc(cur)
            if not (self.output_readout == "analog" and i == len(self.fcs) - 1):
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
                if self.temporal_input:
                    x_seq = x.view(batch_size, self.in_channels, self.in_height, self.in_width)
                    x_seq = x_seq.reshape(batch_size, self.in_channels * self.in_height, self.in_width)
                    x_seq = x_seq.permute(2, 0, 1).contiguous()
                    x_seq = x_seq * self.input_scale
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

            for layer_idx, (fc, lif) in enumerate(zip(self.fcs, self.lif_layers)):
                in_spikes_count = (cur != 0).sum().item()
                out_dim = fc.out_features
                total_synops += int(in_spikes_count * out_dim)
                total_possible_ops += int(cur.numel() * out_dim)
                cur = fc(cur)
                if not (self.output_readout == "analog" and layer_idx == len(self.fcs) - 1):
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
                p_str = f",p{cl['pool_h']}x{cl['pool_w']}" if cl['pool_h'] > 1 or cl['pool_w'] > 1 else ""
                parts.append(f"Conv({cl['in_channels']}->{cl['out_channels']},{cl['kernel_h']}x{cl['kernel_w']}{p_str})")
            parts.append(f"Flat({self.flat_size})")
        else:
            if self.temporal_input:
                parts.append(f"{self.flat_size}×{self.num_steps} temporal")
            else:
                parts.append(str(self.input_size))

        for h in self.hidden_layers:
            parts.append(str(h))
        parts.append(str(self.output_size))

        n_label = "SNN-IF" if self.neuron_type == "if" else "SNN-LIF"
        enc_label = self.encoding_type.upper()
        if self.temporal_input:
            enc_label = f"TEMPORAL/{enc_label}"
        readout_label = f", Readout={self.output_readout.upper()}"
        if self.neuron_type == "if":
            param_str = f"T={self.num_steps}, Enc={enc_label}, θ={self.threshold:.1f}"
        else:
            param_str = f"T={self.num_steps}, Enc={enc_label}, β={self.beta:.2f}, θ={self.threshold:.1f}"
        return f"SpikingJelly {n_label} [{' -> '.join(parts)}] ({param_str}{readout_label})"
