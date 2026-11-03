"""TFLite Micro export for the compact speech Transformer family.

The supported models share the repository's fixed 16-token/16-feature input
layout. Spiking time is statically unrolled, with LIF membrane updates expressed
using TFLite built-in arithmetic, comparison, and SELECT_V2 operators.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import tflite

from .tflm_snn import _Graph, _empty_add_options, _empty_mul_options


def export_transformer_to_tflite(model: Any) -> bytes:
    """Export a compact ANN, compact spiking Transformer, or SDSA model."""
    if not (getattr(model, "is_compact_transformer", False)
            or getattr(model, "is_spiking_transformer", False)):
        raise ValueError("TFLM Transformer export requires a compact Transformer model.")
    if int(model.in_channels) != 1:
        raise ValueError("TFLM Transformer export currently expects single-channel Mel features.")
    tokens, features = int(model.in_width), int(model.in_height)
    embed = int(model.input_projection.out_features)
    if (tokens, features, embed, int(model.output_size)) != (16, 16, 32, 35):
        raise ValueError("TFLM Transformer export currently supports 16x16 input, D=32, and 35 classes.")

    g = _Graph()
    b = g.builder
    spiking = bool(getattr(model, "is_spiking_transformer", False))
    spike_driven = bool(getattr(model, "is_spike_driven_transformer", False))
    steps = int(model.inference_steps) if spiking else 1
    if steps < 1 or steps > 32:
        raise ValueError("TFLM Transformer export supports 1–32 inference steps.")

    # For SNNs, firmware supplies one encoded Mel tensor per simulated step.
    # ANN has a single input tensor.
    raw_inputs = [g.tensor(f"mel_t{t}", [1, features, tokens]) for t in range(steps)]
    g.inputs.extend(raw_inputs)

    def const(name: str, values: Any, shape: list[int] | None = None,
              dtype: int = tflite.TensorType.FLOAT32) -> int:
        arr = np.asarray(values, dtype=np.float32 if dtype == tflite.TensorType.FLOAT32 else np.int32)
        if shape is not None:
            arr = arr.reshape(shape)
        return g.tensor(name, list(arr.shape), dtype, arr)

    scalar = {
        "zero": const("scalar_zero", [0.0], [1]),
        "one": const("scalar_one", [1.0], [1]),
        "half": const("scalar_half", [0.5], [1]),
        "gelu_a": const("gelu_a", [0.7978845608028654], [1]),
        "gelu_b": const("gelu_b", [0.044715], [1]),
        "gelu_three": const("gelu_three", [3.0], [1]),
        "epsilon": const("layer_norm_epsilon", [1e-5], [1]),
    }
    if spiking:
        scalar["beta"] = const("lif_beta", [float(model.beta)], [1])
        scalar["threshold"] = const("lif_threshold", [float(model.threshold)], [1])
        scalar["attn_threshold"] = const(
            "attention_threshold", [float(getattr(model, "attention_threshold", model.threshold))], [1]
        )

    def op(code: int, inputs: list[int], shape: list[int], name: str,
           opt_type: int = tflite.BuiltinOptions.NONE, options: int = 0,
           dtype: int = tflite.TensorType.FLOAT32) -> int:
        out = g.tensor(name, shape, dtype)
        g.op(code, inputs, out, opt_type, options)
        return out

    def add(x: int, y: int, name: str) -> int:
        return op(tflite.BuiltinOperator.ADD, [x, y], g.shapes[x], name,
                  tflite.BuiltinOptions.AddOptions, _empty_add_options(b))

    def sub(x: int, y: int, name: str) -> int:
        return op(tflite.BuiltinOperator.SUB, [x, y], g.shapes[x], name,
                  tflite.BuiltinOptions.SubOptions, _empty_sub_options(b))

    def mul(x: int, y: int, name: str) -> int:
        return op(tflite.BuiltinOperator.MUL, [x, y], g.shapes[x], name,
                  tflite.BuiltinOptions.MulOptions, _empty_mul_options(b))

    def reduce(x: int, axis: list[int], name: str, code: int,
               keep_dims: bool = True) -> int:
        axes = const(f"{name}_axes", axis, [len(axis)], tflite.TensorType.INT32)
        shape = list(g.shapes[x])
        for a in axis:
            a = a if a >= 0 else len(shape) + a
            if keep_dims:
                shape[a] = 1
            else:
                shape[a] = -1
        if not keep_dims:
            shape = [d for d in shape if d != -1]
        opts = _reducer_options(b, keep_dims)
        return op(code, [x, axes], shape, name, tflite.BuiltinOptions.ReducerOptions, opts)

    def reshape(x: int, shape: list[int], name: str) -> int:
        shape_tensor = const(f"{name}_shape", shape, [len(shape)], tflite.TensorType.INT32)
        opts = _reshape_options(b, shape)
        return op(tflite.BuiltinOperator.RESHAPE, [x, shape_tensor], shape, name,
                  tflite.BuiltinOptions.ReshapeOptions, opts)

    weight_tensors: dict[Any, int] = {}
    bias_tensors: dict[Any, int] = {}

    def linear(x: int, layer: Any, name: str, weight: np.ndarray | None = None,
               bias: np.ndarray | None = None, weight_key: Any = None,
               no_bias: bool = False) -> int:
        w = (layer.weight.detach().cpu().numpy() if weight is None else weight).astype(np.float32)
        bias_values = (None if no_bias else
                       layer.bias.detach().cpu().numpy() if bias is None and layer.bias is not None
                       else bias)
        out_features = int(w.shape[0])
        if weight_key is None:
            weight_key = ("parameter", id(layer.weight))
        if weight_key not in weight_tensors:
            weight_tensors[weight_key] = const(f"weights_{len(weight_tensors)}", w, list(w.shape))
        w_idx = weight_tensors[weight_key]
        if bias_values is None:
            bias_values = np.zeros((out_features,), dtype=np.float32)
            bias_key = (weight_key, "zero_bias")
        elif bias is None:
            bias_key = ("parameter", id(layer.bias))
        else:
            bias_key = (weight_key, "provided_bias")
        if bias_key not in bias_tensors:
            bias_tensors[bias_key] = const(f"bias_{len(bias_tensors)}", bias_values, [out_features])
        bias_idx = bias_tensors[bias_key]
        shape = list(g.shapes[x])
        shape[-1] = out_features
        opts = _fc_options_keep_dims(b)
        return op(tflite.BuiltinOperator.FULLY_CONNECTED, [x, w_idx, bias_idx], shape, name,
                  tflite.BuiltinOptions.FullyConnectedOptions, opts)

    def layer_norm(x: int, layer: Any, name: str) -> int:
        width = g.shapes[x][-1]
        mean = reduce(x, [len(g.shapes[x]) - 1], f"{name}_mean", tflite.BuiltinOperator.MEAN)
        centered = sub(x, mean, f"{name}_centered")
        variance = reduce(mul(centered, centered, f"{name}_square"), [len(g.shapes[x]) - 1],
                          f"{name}_variance", tflite.BuiltinOperator.MEAN)
        variance_eps = add(variance, scalar["epsilon"], f"{name}_variance_eps")
        inv_std = op(tflite.BuiltinOperator.RSQRT, [variance_eps], g.shapes[variance_eps],
                     f"{name}_inv_std")
        normed = mul(centered, inv_std, f"{name}_normalized")
        affine_shape = [1, width] if len(g.shapes[x]) == 2 else [1, 1, width]
        gamma = const(f"{name}_gamma", layer.weight.detach().cpu().numpy(), affine_shape)
        beta = const(f"{name}_beta", layer.bias.detach().cpu().numpy(), affine_shape)
        normed = mul(normed, gamma, f"{name}_scaled")
        return add(normed, beta, f"{name}_shifted")

    def gelu(x: int, name: str) -> int:
        # tanh GELU approximation, equivalent to the common Transformer form.
        x2 = mul(x, x, f"{name}_x2")
        x3 = mul(x2, x, f"{name}_x3")
        cubic = mul(x3, scalar["gelu_b"], f"{name}_cubic")
        inner = add(x, cubic, f"{name}_inner")
        scaled = mul(inner, scalar["gelu_a"], f"{name}_scaled")
        tanh = op(tflite.BuiltinOperator.TANH, [scaled], g.shapes[scaled], f"{name}_tanh")
        one_plus = add(tanh, scalar["one"], f"{name}_one_plus")
        half_x = mul(x, scalar["half"], f"{name}_half_x")
        return mul(half_x, one_plus, name)

    def lif(current: int, previous: int | None, name: str,
            threshold: int | None = None) -> tuple[int, int, int]:
        if previous is None:
            previous = const(f"{name}_mem0", np.zeros(g.shapes[current], dtype=np.float32))
        decayed = mul(previous, scalar["beta"], f"{name}_decayed")
        potential = add(decayed, current, f"{name}_potential")
        condition = op(tflite.BuiltinOperator.GREATER_EQUAL,
                       [potential, threshold or scalar["threshold"]],
                       g.shapes[potential], f"{name}_condition",
                       tflite.BuiltinOptions.GreaterEqualOptions,
                       _empty_greater_equal_options(b), tflite.TensorType.BOOL)
        spikes = op(tflite.BuiltinOperator.SELECT_V2,
                    [condition, scalar["one"], scalar["zero"]], g.shapes[potential],
                    f"{name}_spikes", tflite.BuiltinOptions.SelectV2Options,
                    _empty_select_options(b))
        membrane = op(tflite.BuiltinOperator.SELECT_V2,
                      [condition, scalar["zero"], potential], g.shapes[potential],
                      f"{name}_membrane", tflite.BuiltinOptions.SelectV2Options,
                      _empty_select_options(b))
        return spikes, membrane, potential

    def split_linear_params(layer: Any, part: int) -> tuple[np.ndarray, np.ndarray]:
        weight = layer.self_attn.in_proj_weight.detach().cpu().numpy().astype(np.float32)
        bias = layer.self_attn.in_proj_bias.detach().cpu().numpy().astype(np.float32)
        width = int(weight.shape[0] // 3)
        return weight[part * width:(part + 1) * width], bias[part * width:(part + 1) * width]

    def attention(q: int, k: int, v: int, name: str) -> int:
        head_count = int(model.encoder.layers[0].self_attn.num_heads)
        head_dim = embed // head_count
        q4 = reshape(q, [1, tokens, head_count, head_dim], f"{name}_q4")
        k4 = reshape(k, [1, tokens, head_count, head_dim], f"{name}_k4")
        v4 = reshape(v, [1, tokens, head_count, head_dim], f"{name}_v4")
        permute = const(f"{name}_head_perm", [0, 2, 1, 3], [4], tflite.TensorType.INT32)
        qh = op(tflite.BuiltinOperator.TRANSPOSE, [q4, permute], [1, head_count, tokens, head_dim], f"{name}_qh")
        kh = op(tflite.BuiltinOperator.TRANSPOSE, [k4, permute], [1, head_count, tokens, head_dim], f"{name}_kh")
        vh = op(tflite.BuiltinOperator.TRANSPOSE, [v4, permute], [1, head_count, tokens, head_dim], f"{name}_vh")
        bmm_opts = _batch_matmul_options(b, adj_y=True)
        scores = op(tflite.BuiltinOperator.BATCH_MATMUL, [qh, kh],
                    [1, head_count, tokens, tokens], f"{name}_scores",
                    tflite.BuiltinOptions.BatchMatMulOptions, bmm_opts)
        scale = const(f"{name}_scale", [1.0 / np.sqrt(float(head_dim))], [1])
        scores = mul(scores, scale, f"{name}_scaled_scores")
        softmax_opts = _softmax_options(b)
        probs = op(tflite.BuiltinOperator.SOFTMAX, [scores], g.shapes[scores],
                   f"{name}_softmax", tflite.BuiltinOptions.SoftmaxOptions, softmax_opts)
        values = op(tflite.BuiltinOperator.BATCH_MATMUL, [probs, vh],
                    [1, head_count, tokens, head_dim], f"{name}_values",
                    tflite.BuiltinOptions.BatchMatMulOptions, _batch_matmul_options(b))
        merged = op(tflite.BuiltinOperator.TRANSPOSE, [values, permute],
                    [1, tokens, head_count, head_dim], f"{name}_merged4")
        return reshape(merged, [1, tokens, embed], f"{name}_merged")

    def dense_qkv(x: int, layer: Any, name: str) -> tuple[int, int, int]:
        result = []
        for part, label in enumerate(("q", "k", "v")):
            w, bias = split_linear_params(layer, part)
            stable_prefix = name.rsplit("_t", 1)[0]
            result.append(linear(x, None, f"{name}_{label}", w, bias,
                                 weight_key=(stable_prefix, label)))
        return tuple(result)  # type: ignore[return-value]

    # The input arrives as [frequency,time], matching the exported Mel array.
    pos = const("position_embedding", model.position_embedding.detach().cpu().numpy(),
                [1, tokens, embed])
    proj_w = model.input_projection.weight.detach().cpu().numpy().astype(np.float32)
    proj_b = model.input_projection.bias.detach().cpu().numpy().astype(np.float32)

    if not spiking:
        trans_perm = const("mel_token_transpose", [0, 2, 1], [3], tflite.TensorType.INT32)
        token_input = op(tflite.BuiltinOperator.TRANSPOSE, [raw_inputs[0], trans_perm],
                         [1, tokens, features], "mel_tokens")
        projected = linear(token_input, model.input_projection, "input_projection")
        current = add(projected, pos, "position_added")
        for idx, layer in enumerate(model.encoder.layers):
            n1 = layer_norm(current, layer.norm1, f"block{idx}_norm1")
            q, k, v = dense_qkv(n1, layer, f"block{idx}_qkv")
            attn = attention(q, k, v, f"block{idx}_attention")
            attn = linear(attn, layer.self_attn.out_proj, f"block{idx}_attention_output")
            current = add(current, attn, f"block{idx}_attention_residual")
            n2 = layer_norm(current, layer.norm2, f"block{idx}_norm2")
            hidden = linear(n2, layer.linear1, f"block{idx}_ff1")
            hidden = gelu(hidden, f"block{idx}_gelu")
            ff = linear(hidden, layer.linear2, f"block{idx}_ff2")
            current = add(current, ff, f"block{idx}_ff_residual")
        pooled = reduce(current, [1], "token_mean", tflite.BuiltinOperator.MEAN, keep_dims=False)
        pooled = layer_norm(pooled, model.norm, "final_norm")
        logits = linear(pooled, model.classifier, "classifier")
        g.outputs.append(logits)
        return g.finish("TFLM compact Transformer ANN")

    input_scale = const("input_scale", [float(model.input_scale)], [1])
    beta = scalar["beta"]
    input_mem = None
    spike_accum = None
    qkv_mem: list[list[int | None]] = [[None, None, None] for _ in model.encoder.layers]
    if spike_driven:
        context_mem: list[int | None] = [None] * len(model.encoder.layers)
        attention_mem: list[int | None] = [None] * len(model.encoder.layers)
        hidden_mem: list[int | None] = [None] * len(model.encoder.layers)
        ff_mem: list[int | None] = [None] * len(model.encoder.layers)
    else:
        attention_mem = [None] * len(model.encoder.layers)
        ff_mem = [None] * len(model.encoder.layers)

    for step, raw in enumerate(raw_inputs):
        transpose = const(f"mel_transpose_t{step}", [0, 2, 1], [3], tflite.TensorType.INT32)
        mel = op(tflite.BuiltinOperator.TRANSPOSE, [raw, transpose],
                 [1, tokens, features], f"mel_tokens_t{step}")
        if spike_driven and bool(getattr(model, "ttfs_input", False)):
            current = linear(mel, model.input_projection, f"input_projection_t{step}", proj_w,
                             proj_b if step == 0 else None, weight_key="input_projection",
                             no_bias=step != 0)
            if step == 0:
                current = add(current, pos, "position_added_t0")
        else:
            current = linear(mel, model.input_projection, f"input_projection_t{step}")
            current = add(current, pos, f"position_added_t{step}")
        current = mul(current, input_scale, f"input_scaled_t{step}")
        current, input_mem, residual_potential = lif(current, input_mem, f"input_lif_t{step}")

        for idx, layer in enumerate(model.encoder.layers):
            if not spike_driven:
                normed = layer_norm(current, layer.norm1, f"block{idx}_norm1_t{step}")
                q, k, v = dense_qkv(normed, layer, f"block{idx}_qkv_t{step}")
                q, qkv_mem[idx][0], _ = lif(q, qkv_mem[idx][0], f"block{idx}_q_lif_t{step}")
                k, qkv_mem[idx][1], _ = lif(k, qkv_mem[idx][1], f"block{idx}_k_lif_t{step}")
                v, qkv_mem[idx][2], _ = lif(v, qkv_mem[idx][2], f"block{idx}_v_lif_t{step}")
                attn = attention(q, k, v, f"block{idx}_attention_t{step}")
                attn = linear(attn, layer.self_attn.out_proj, f"block{idx}_attention_output_t{step}")
                current, attention_mem[idx], _ = lif(
                    add(current, attn, f"block{idx}_attention_residual_t{step}"),
                    attention_mem[idx], f"block{idx}_attention_lif_t{step}")
                normed = layer_norm(current, layer.norm2, f"block{idx}_norm2_t{step}")
                hidden = gelu(linear(normed, layer.linear1, f"block{idx}_ff1_t{step}"),
                              f"block{idx}_gelu_t{step}")
                ff = linear(hidden, layer.linear2, f"block{idx}_ff2_t{step}")
                current, ff_mem[idx], _ = lif(
                    add(current, ff, f"block{idx}_ff_residual_t{step}"),
                    ff_mem[idx], f"block{idx}_ff_lif_t{step}")
            else:
                q, k, v = dense_qkv(current, layer, f"block{idx}_qkv_t{step}")
                q, qkv_mem[idx][0], _ = lif(q, qkv_mem[idx][0], f"block{idx}_q_lif_t{step}")
                k, qkv_mem[idx][1], _ = lif(k, qkv_mem[idx][1], f"block{idx}_k_lif_t{step}")
                v, qkv_mem[idx][2], _ = lif(v, qkv_mem[idx][2], f"block{idx}_v_lif_t{step}")
                kv = mul(k, v, f"block{idx}_kv_t{step}")
                context_current = reduce(kv, [1], f"block{idx}_context_sum_t{step}",
                                         tflite.BuiltinOperator.SUM, keep_dims=False)
                context, context_mem[idx], _ = lif(
                    context_current, context_mem[idx], f"block{idx}_context_lif_t{step}",
                    scalar["attn_threshold"])
                context = reshape(context, [1, 1, embed], f"block{idx}_context_expand_t{step}")
                masked_q = mul(q, context, f"block{idx}_query_mask_t{step}")
                attn = linear(masked_q, layer.self_attn.out_proj,
                              f"block{idx}_attention_output_t{step}")
                attn_input = add(attn, residual_potential,
                                 f"block{idx}_attention_residual_potential_t{step}")
                attention_spikes, attention_mem[idx], attention_potential = lif(
                    attn_input, attention_mem[idx], f"block{idx}_attention_lif_t{step}")
                hidden_current = linear(attention_spikes, layer.linear1, f"block{idx}_ff1_t{step}")
                hidden_spikes, hidden_mem[idx], _ = lif(
                    hidden_current, hidden_mem[idx], f"block{idx}_hidden_lif_t{step}")
                ff = linear(hidden_spikes, layer.linear2, f"block{idx}_ff2_t{step}")
                current, ff_mem[idx], residual_potential = lif(
                    add(ff, attention_potential, f"block{idx}_ff_residual_potential_t{step}"),
                    ff_mem[idx], f"block{idx}_ff_lif_t{step}")
        spike_accum = current if spike_accum is None else add(spike_accum, current, f"spike_accum_t{step}")

    assert spike_accum is not None
    denom = const("readout_denominator", [float(steps * tokens)], [1])
    pooled = reduce(spike_accum, [1], "spike_token_sum", tflite.BuiltinOperator.SUM,
                    keep_dims=False)
    pooled = op(tflite.BuiltinOperator.DIV, [pooled, denom], g.shapes[pooled],
                 "spike_rate_mean", tflite.BuiltinOptions.DivOptions, _empty_div_options(b))
    pooled = layer_norm(pooled, model.norm, "final_norm")
    logits = linear(pooled, model.classifier, "classifier")
    g.outputs.append(logits)
    kind = "spike-driven Transformer" if spike_driven else "spiking Transformer"
    return g.finish(f"TFLM compact {kind}, statically unrolled T={steps}")


def _empty_sub_options(builder):
    tflite.SubOptionsStart(builder)
    tflite.SubOptionsAddFusedActivationFunction(builder, tflite.ActivationFunctionType.NONE)
    return tflite.SubOptionsEnd(builder)


def _empty_div_options(builder):
    tflite.DivOptionsStart(builder)
    tflite.DivOptionsAddFusedActivationFunction(builder, tflite.ActivationFunctionType.NONE)
    return tflite.DivOptionsEnd(builder)


def _empty_greater_equal_options(builder):
    tflite.GreaterEqualOptionsStart(builder)
    return tflite.GreaterEqualOptionsEnd(builder)


def _empty_select_options(builder):
    tflite.SelectV2OptionsStart(builder)
    return tflite.SelectV2OptionsEnd(builder)


def _reducer_options(builder, keep_dims: bool):
    tflite.ReducerOptionsStart(builder)
    tflite.ReducerOptionsAddKeepDims(builder, keep_dims)
    return tflite.ReducerOptionsEnd(builder)


def _reshape_options(builder, shape: list[int]):
    tflite.ReshapeOptionsStartNewShapeVector(builder, len(shape))
    for value in reversed(shape):
        builder.PrependInt32(value)
    new_shape = builder.EndVector()
    tflite.ReshapeOptionsStart(builder)
    tflite.ReshapeOptionsAddNewShape(builder, new_shape)
    return tflite.ReshapeOptionsEnd(builder)


def _fc_options_keep_dims(builder):
    tflite.FullyConnectedOptionsStart(builder)
    tflite.FullyConnectedOptionsAddFusedActivationFunction(builder, tflite.ActivationFunctionType.NONE)
    tflite.FullyConnectedOptionsAddWeightsFormat(builder, tflite.FullyConnectedOptionsWeightsFormat.DEFAULT)
    tflite.FullyConnectedOptionsAddKeepNumDims(builder, True)
    return tflite.FullyConnectedOptionsEnd(builder)


def _batch_matmul_options(builder, adj_y: bool = False):
    tflite.BatchMatMulOptionsStart(builder)
    tflite.BatchMatMulOptionsAddAdjX(builder, False)
    tflite.BatchMatMulOptionsAddAdjY(builder, adj_y)
    tflite.BatchMatMulOptionsAddAsymmetricQuantizeInputs(builder, False)
    return tflite.BatchMatMulOptionsEnd(builder)


def _softmax_options(builder):
    tflite.SoftmaxOptionsStart(builder)
    tflite.SoftmaxOptionsAddBeta(builder, 1.0)
    return tflite.SoftmaxOptionsEnd(builder)
