"""TFLite and ESP-IDF model exporters."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import flatbuffers
import numpy as np
import torch
import tflite

from .models import ConfigurableANN, ConfigurableSpikingJellySNN

def export_pytorch_to_tflite(model: Union[ConfigurableANN, ConfigurableSpikingJellySNN]) -> bytes:
    """
    Dynamically create a standards-compliant TFLite FlatBuffer for an
    arbitrary convolutional or fully connected ANN.
    """
    model.eval()
    bld = flatbuffers.Builder(2 * 1024 * 1024)

    tflite.BufferStart(bld)
    buf0 = tflite.BufferEnd(bld)
    buffers = [buf0]

    def make_buffer(arr: np.ndarray) -> int:
        raw_bytes = arr.tobytes(order="C")
        vec = bld.CreateByteVector(raw_bytes)
        tflite.BufferStart(bld)
        tflite.BufferAddData(bld, vec)
        return tflite.BufferEnd(bld)

    op_codes = []
    op_code_map = {}

    def get_opcode_idx(code):
        if code not in op_code_map:
            tflite.OperatorCodeStart(bld)
            tflite.OperatorCodeAddBuiltinCode(bld, code)
            op = tflite.OperatorCodeEnd(bld)
            op_codes.append(op)
            op_code_map[code] = len(op_codes) - 1
        return op_code_map[code]

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
    ops = []

    if getattr(model, "has_conv", False):
        t_input = make_tensor("input", [1, model.in_height, model.in_width, model.in_channels], tflite.TensorType.FLOAT32, 0)
        tensors.append(t_input)
        cur_tensor_idx = 0

        for c_idx, cl in enumerate(model.resolved_conv):
            seq = model.conv_modules[c_idx]
            conv_mod = seq[0]
            # PyTorch: (out_c, in_c, kh, kw) -> TFLite: (out_c, kh, kw, in_c)
            w = conv_mod.weight.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
            b = conv_mod.bias.detach().cpu().numpy().astype(np.float32)
            w_buf = make_buffer(w)
            b_buf = make_buffer(b)
            buffers.extend([w_buf, b_buf])
            w_buf_idx = len(buffers) - 2
            b_buf_idx = len(buffers) - 1

            t_w = make_tensor(f"conv{c_idx+1}_w", [cl["out_channels"], cl["kernel_h"], cl["kernel_w"], cl["in_channels"]], tflite.TensorType.FLOAT32, w_buf_idx)
            t_b = make_tensor(f"conv{c_idx+1}_b", [cl["out_channels"]], tflite.TensorType.FLOAT32, b_buf_idx)
            t_conv_out = make_tensor(f"conv{c_idx+1}_out", [1, cl["out_h"], cl["out_w"], cl["out_channels"]], tflite.TensorType.FLOAT32, 0)

            w_idx = len(tensors)
            b_idx = len(tensors) + 1
            conv_out_idx = len(tensors) + 2
            tensors.extend([t_w, t_b, t_conv_out])

            op_conv_idx = get_opcode_idx(tflite.BuiltinOperator.CONV_2D)
            tflite.Conv2DOptionsStart(bld)
            has_padding = any(cl["padding"]) if isinstance(cl["padding"], tuple) else cl["padding"] > 0
            tflite.Conv2DOptionsAddPadding(bld, tflite.Padding.SAME if has_padding else tflite.Padding.VALID)
            tflite.Conv2DOptionsAddStrideH(bld, cl["stride"])
            tflite.Conv2DOptionsAddStrideW(bld, cl["stride"])
            act_type = tflite.ActivationFunctionType.RELU if getattr(model, "activation_name", "relu") == "relu" else tflite.ActivationFunctionType.NONE
            tflite.Conv2DOptionsAddFusedActivationFunction(bld, act_type)
            opt_conv = tflite.Conv2DOptionsEnd(bld)

            tflite.OperatorStartInputsVector(bld, 3)
            bld.PrependInt32(b_idx)
            bld.PrependInt32(w_idx)
            bld.PrependInt32(cur_tensor_idx)
            in_conv = bld.EndVector()

            tflite.OperatorStartOutputsVector(bld, 1)
            bld.PrependInt32(conv_out_idx)
            out_conv = bld.EndVector()

            tflite.OperatorStart(bld)
            tflite.OperatorAddOpcodeIndex(bld, op_conv_idx)
            tflite.OperatorAddInputs(bld, in_conv)
            tflite.OperatorAddOutputs(bld, out_conv)
            tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.Conv2DOptions)
            tflite.OperatorAddBuiltinOptions(bld, opt_conv)
            ops.append(tflite.OperatorEnd(bld))
            cur_tensor_idx = conv_out_idx

            if cl["pool_h"] > 1 or cl["pool_w"] > 1:
                t_pool_out = make_tensor(f"pool{c_idx+1}_out", [1, cl["pool_out_h"], cl["pool_out_w"], cl["out_channels"]], tflite.TensorType.FLOAT32, 0)
                pool_out_idx = len(tensors)
                tensors.append(t_pool_out)

                pool_op_code = tflite.BuiltinOperator.MAX_POOL_2D if cl["pool_type"] == "max" else tflite.BuiltinOperator.AVERAGE_POOL_2D
                op_pool_idx = get_opcode_idx(pool_op_code)
                tflite.Pool2DOptionsStart(bld)
                tflite.Pool2DOptionsAddPadding(bld, tflite.Padding.VALID)
                tflite.Pool2DOptionsAddStrideH(bld, cl["pool_stride_h"])
                tflite.Pool2DOptionsAddStrideW(bld, cl["pool_stride_w"])
                tflite.Pool2DOptionsAddFilterHeight(bld, cl["pool_h"])
                tflite.Pool2DOptionsAddFilterWidth(bld, cl["pool_w"])
                tflite.Pool2DOptionsAddFusedActivationFunction(bld, tflite.ActivationFunctionType.NONE)
                opt_pool = tflite.Pool2DOptionsEnd(bld)

                tflite.OperatorStartInputsVector(bld, 1)
                bld.PrependInt32(cur_tensor_idx)
                in_pool = bld.EndVector()

                tflite.OperatorStartOutputsVector(bld, 1)
                bld.PrependInt32(pool_out_idx)
                out_pool = bld.EndVector()

                tflite.OperatorStart(bld)
                tflite.OperatorAddOpcodeIndex(bld, op_pool_idx)
                tflite.OperatorAddInputs(bld, in_pool)
                tflite.OperatorAddOutputs(bld, out_pool)
                tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.Pool2DOptions)
                tflite.OperatorAddBuiltinOptions(bld, opt_pool)
                ops.append(tflite.OperatorEnd(bld))
                cur_tensor_idx = pool_out_idx

        # Reshape / Flatten
        shape_arr = np.array([1, model.flat_size], dtype=np.int32)
        buf_shape = make_buffer(shape_arr)
        buffers.append(buf_shape)
        shape_buf_idx = len(buffers) - 1

        t_shape = make_tensor("flat_shape", [2], tflite.TensorType.INT32, shape_buf_idx)
        t_flat = make_tensor("flat_out", [1, model.flat_size], tflite.TensorType.FLOAT32, 0)
        shape_tensor_idx = len(tensors)
        flat_tensor_idx = len(tensors) + 1
        tensors.extend([t_shape, t_flat])

        op_reshape_idx = get_opcode_idx(tflite.BuiltinOperator.RESHAPE)
        tflite.ReshapeOptionsStartNewShapeVector(bld, 2)
        bld.PrependInt32(model.flat_size)
        bld.PrependInt32(1)
        new_shape_vec = bld.EndVector()
        tflite.ReshapeOptionsStart(bld)
        tflite.ReshapeOptionsAddNewShape(bld, new_shape_vec)
        opt_reshape = tflite.ReshapeOptionsEnd(bld)

        tflite.OperatorStartInputsVector(bld, 2)
        bld.PrependInt32(shape_tensor_idx)
        bld.PrependInt32(cur_tensor_idx)
        in_reshape = bld.EndVector()

        tflite.OperatorStartOutputsVector(bld, 1)
        bld.PrependInt32(flat_tensor_idx)
        out_reshape = bld.EndVector()

        tflite.OperatorStart(bld)
        tflite.OperatorAddOpcodeIndex(bld, op_reshape_idx)
        tflite.OperatorAddInputs(bld, in_reshape)
        tflite.OperatorAddOutputs(bld, out_reshape)
        tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.ReshapeOptions)
        tflite.OperatorAddBuiltinOptions(bld, opt_reshape)
        ops.append(tflite.OperatorEnd(bld))
        cur_tensor_idx = flat_tensor_idx

        # FC layers
        op_fc_idx = get_opcode_idx(tflite.BuiltinOperator.FULLY_CONNECTED)
        num_fc = len(model.fc_layers)
        for f_idx, fc in enumerate(model.fc_layers):
            is_last = (f_idx == num_fc - 1)
            use_relu = (not is_last) and (getattr(model, "activation_name", "relu") == "relu")

            w = fc.weight.detach().cpu().numpy().astype(np.float32)
            if f_idx == 0:
                # TFLite Conv2D produces NHWC data, while PyTorch's Flatten
                # presents the same activations to this layer in NCHW order.
                # Keep the TFLite graph in NHWC and reorder the first FC
                # layer's columns to match that layout.
                last_conv = model.resolved_conv[-1]
                channels = last_conv["out_channels"]
                height = last_conv["pool_out_h"]
                width = last_conv["pool_out_w"]
                w = w.reshape(fc.out_features, channels, height, width)
                w = w.transpose(0, 2, 3, 1).reshape(fc.out_features, -1)
            b = fc.bias.detach().cpu().numpy().astype(np.float32)
            w_buf = make_buffer(w)
            b_buf = make_buffer(b)
            buffers.extend([w_buf, b_buf])
            w_buf_idx = len(buffers) - 2
            b_buf_idx = len(buffers) - 1

            t_w = make_tensor(f"fc{f_idx+1}_w", [fc.out_features, fc.in_features], tflite.TensorType.FLOAT32, w_buf_idx)
            t_b = make_tensor(f"fc{f_idx+1}_b", [fc.out_features], tflite.TensorType.FLOAT32, b_buf_idx)
            out_name = "output" if is_last else f"fc{f_idx+1}_out"
            t_out = make_tensor(out_name, [1, fc.out_features], tflite.TensorType.FLOAT32, 0)

            w_idx = len(tensors)
            b_idx = len(tensors) + 1
            out_idx = len(tensors) + 2
            tensors.extend([t_w, t_b, t_out])

            tflite.FullyConnectedOptionsStart(bld)
            tflite.FullyConnectedOptionsAddFusedActivationFunction(bld, tflite.ActivationFunctionType.RELU if use_relu else tflite.ActivationFunctionType.NONE)
            opt_fc = tflite.FullyConnectedOptionsEnd(bld)

            tflite.OperatorStartInputsVector(bld, 3)
            bld.PrependInt32(b_idx)
            bld.PrependInt32(w_idx)
            bld.PrependInt32(cur_tensor_idx)
            in_fc = bld.EndVector()

            tflite.OperatorStartOutputsVector(bld, 1)
            bld.PrependInt32(out_idx)
            out_fc = bld.EndVector()

            tflite.OperatorStart(bld)
            tflite.OperatorAddOpcodeIndex(bld, op_fc_idx)
            tflite.OperatorAddInputs(bld, in_fc)
            tflite.OperatorAddOutputs(bld, out_fc)
            tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.FullyConnectedOptions)
            tflite.OperatorAddBuiltinOptions(bld, opt_fc)
            ops.append(tflite.OperatorEnd(bld))
            cur_tensor_idx = out_idx

    else:
        # Pure FC Model
        layers = list(model.fc_layers) if hasattr(model, "fc_layers") else list(model.layers)
        num_layers = len(layers)
        op_fc_idx = get_opcode_idx(tflite.BuiltinOperator.FULLY_CONNECTED)

        t_input = make_tensor("input", [1, model.input_size], tflite.TensorType.FLOAT32, 0)
        tensors.append(t_input)
        cur_tensor_idx = 0

        for i, layer in enumerate(layers):
            is_last = (i == num_layers - 1)
            use_relu = (not is_last) and (getattr(model, "activation_name", "relu") == "relu")

            w = layer.weight.detach().cpu().numpy().astype(np.float32)
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            w_buf = make_buffer(w)
            b_buf = make_buffer(b)
            buffers.extend([w_buf, b_buf])
            w_buf_idx = len(buffers) - 2
            b_buf_idx = len(buffers) - 1

            t_w = make_tensor(f"fc{i+1}_w", [layer.out_features, layer.in_features], tflite.TensorType.FLOAT32, w_buf_idx)
            t_b = make_tensor(f"fc{i+1}_b", [layer.out_features], tflite.TensorType.FLOAT32, b_buf_idx)
            out_name = "output" if is_last else f"fc{i+1}_out"
            t_out = make_tensor(out_name, [1, layer.out_features], tflite.TensorType.FLOAT32, 0)

            w_idx = len(tensors)
            b_idx = len(tensors) + 1
            out_idx = len(tensors) + 2
            tensors.extend([t_w, t_b, t_out])

            tflite.FullyConnectedOptionsStart(bld)
            tflite.FullyConnectedOptionsAddFusedActivationFunction(bld, tflite.ActivationFunctionType.RELU if use_relu else tflite.ActivationFunctionType.NONE)
            opt_fc = tflite.FullyConnectedOptionsEnd(bld)

            tflite.OperatorStartInputsVector(bld, 3)
            bld.PrependInt32(b_idx)
            bld.PrependInt32(w_idx)
            bld.PrependInt32(cur_tensor_idx)
            in_fc = bld.EndVector()

            tflite.OperatorStartOutputsVector(bld, 1)
            bld.PrependInt32(out_idx)
            out_fc = bld.EndVector()

            tflite.OperatorStart(bld)
            tflite.OperatorAddOpcodeIndex(bld, op_fc_idx)
            tflite.OperatorAddInputs(bld, in_fc)
            tflite.OperatorAddOutputs(bld, out_fc)
            tflite.OperatorAddBuiltinOptionsType(bld, tflite.BuiltinOptions.FullyConnectedOptions)
            tflite.OperatorAddBuiltinOptions(bld, opt_fc)
            ops.append(tflite.OperatorEnd(bld))
            cur_tensor_idx = out_idx

    final_output_idx = cur_tensor_idx

    # Subgraph
    tflite.SubGraphStartInputsVector(bld, 1)
    bld.PrependInt32(0)
    sub_in = bld.EndVector()

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

    # Model
    tflite.ModelStartOperatorCodesVector(bld, len(op_codes))
    for oc in reversed(op_codes):
        bld.PrependUOffsetTRelative(oc)
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

def export_c_files(
    model_entry: Dict[str, Any],
    output_dir: str = "main",
):
    """
    Generate model_data.h/cc and mnist_test_data.h in the target directory for ESP32.
    Supports ANNs (TFLM FlatBuffer) and SNNs (LIF parameters and weights).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    config = model_entry["config"]
    model = model_entry["model"]
    test_dataset = model_entry["test_dataset"]
    model_type = model_entry.get("type", "ann").lower()
    model_name = model_entry.get("name", "model")

    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})

    dataset_name = config.get("dataset", config.get("training", {}).get("dataset", "mnist"))
    if isinstance(dataset_name, dict):
        dataset_name = dataset_name.get("name", "mnist")
    dataset_name = str(dataset_name).strip().lower()

    input_size = int(model_cfg.get("input_size", 784))
    dataset_input_size = input_size
    temporal_input = bool(getattr(model, "temporal_input", False))
    if temporal_input:
        input_size = int(model.flat_size)
    output_size = int(model_cfg.get("output_size", 10))
    num_test_samples = int(eval_cfg.get("num_test_samples", 500))
    if dataset_name in ("speech_commands", "kws"):
        num_test_samples = min(num_test_samples, 100)
    if test_dataset is not None and hasattr(test_dataset, "__len__"):
        num_test_samples = min(num_test_samples, len(test_dataset))

    # 1. model_data.h
    header_path = out_path / "model_data.h"
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stddef.h>\n\n")
        f.write("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n")
        f.write("#define MODEL_TYPE_ANN 0\n")
        f.write("#define MODEL_TYPE_SNN 1\n\n")
        f.write("#define MODEL_TYPE_TRANSFORMER 2\n\n")
        f.write("#define NEURON_TYPE_LIF 0\n")
        f.write("#define NEURON_TYPE_IF  1\n\n")
        f.write("#define ENCODING_DIRECT    0\n")
        f.write("#define ENCODING_TTFS      1\n")
        f.write("#define ENCODING_POISSON   2\n")
        f.write("#define ENCODING_PERIODIC  3\n\n")
        f.write("#define EXECUTION_MODE_DENSE       0\n")
        f.write("#define EXECUTION_MODE_EVENT_BASED 1\n\n")
        f.write("#define SNN_MAX_FC_LAYERS   8\n")
        f.write("#define SNN_MAX_CONV_LAYERS 4\n\n")
        f.write("typedef struct {\n")
        f.write("    int in_channels;\n")
        f.write("    int out_channels;\n")
        f.write("    int kernel_h, kernel_w;\n")
        f.write("    int stride;\n")
        f.write("    int padding_h, padding_w;\n")
        f.write("    int pool_h, pool_w;\n")
        f.write("    int pool_stride_h, pool_stride_w;\n")
        f.write("    int pool_type;  // 0=max, 1=average\n")
        f.write("    int in_h, in_w;\n")
        f.write("    int out_h, out_w;\n")
        f.write("    int pool_out_h, pool_out_w;\n")
        f.write("} snn_conv_layer_cfg_t;\n\n")
        f.write("typedef struct {\n")
        f.write("    int input_size;\n")
        f.write("    int num_conv_layers;\n")
        f.write("    snn_conv_layer_cfg_t conv_layers[SNN_MAX_CONV_LAYERS];\n")
        f.write("    int flat_size;\n")
        f.write("    int hidden_layers_count;\n")
        f.write("    int fc_layer_sizes[SNN_MAX_FC_LAYERS + 1];\n")
        f.write("    int num_steps;\n")
        f.write("    float beta;\n")
        f.write("    float threshold;\n")
        f.write("    int output_size;\n")
        f.write("    int neuron_type;\n")
        f.write("    int encoding_type;\n")
        f.write("    int execution_mode;\n")
        f.write("    int temporal_input;\n")
        f.write("    float input_scale;\n")
        f.write("    int analog_readout;\n")
        f.write("} snn_config_t;\n\n")
        f.write("typedef struct {\n")
        f.write("    const float *qkv_w, *qkv_b, *attn_out_w, *attn_out_b;\n")
        f.write("    const float *norm1_w, *norm1_b, *ff1_w, *ff1_b;\n")
        f.write("    const float *ff2_w, *ff2_b, *norm2_w, *norm2_b;\n")
        f.write("} transformer_layer_weights_t;\n\n")
        f.write("typedef struct { int tokens, input_features, embed_dim, num_heads, num_layers, feedforward_dim, output_size, spiking, num_steps; float beta, threshold, input_scale; } transformer_config_t;\n\n")
        f.write("extern const int g_model_type;\n")
        f.write("extern const char* const g_model_name;\n\n")
        f.write("// ANN-specific TensorFlow Lite Micro FlatBuffer\n")
        f.write("extern const unsigned char g_model[];\n")
        f.write("extern const int g_model_len;\n\n")
        f.write("// SNN-specific LIF/IF neuron parameters and weights\n")
        f.write("extern const snn_config_t g_snn_config;\n")
        f.write("extern const float* const g_snn_conv_weights[];\n")
        f.write("extern const float* const g_snn_conv_biases[];\n")
        f.write("extern const float* const g_snn_fc_weights[];\n")
        f.write("extern const float* const g_snn_fc_biases[];\n\n")
        f.write("extern const transformer_config_t g_transformer_config;\n")
        f.write("extern const float* const g_transformer_input_w;\n")
        f.write("extern const float* const g_transformer_input_b;\n")
        f.write("extern const float* const g_transformer_position;\n")
        f.write("extern const transformer_layer_weights_t g_transformer_layers[];\n")
        f.write("extern const float* const g_transformer_norm_w;\n")
        f.write("extern const float* const g_transformer_norm_b;\n")
        f.write("extern const float* const g_transformer_classifier_w;\n")
        f.write("extern const float* const g_transformer_classifier_b;\n\n")
        f.write("extern const float* const g_speech_mel_filterbank;\n\n")
        f.write("#ifdef __cplusplus\n}\n#endif\n")

    # 2. model_data.cc
    cc_path = out_path / "model_data.cc"
    with open(cc_path, "w", encoding="utf-8") as f:
        f.write('#include "model_data.h"\n\n')
        f.write(f'const char* const g_model_name = "{model_name}";\n')

        tflite_bytes = model_entry.get("tflite_bytes", b"")
        is_transformer = bool(getattr(model, "is_compact_transformer", False) or getattr(model, "is_spiking_transformer", False))
        if is_transformer:
            f.write("const int g_model_type = MODEL_TYPE_TRANSFORMER;\n\n")
        elif model_type == "ann":
            f.write("const int g_model_type = MODEL_TYPE_ANN;\n\n")
        else:
            f.write("const int g_model_type = MODEL_TYPE_SNN;\n\n")

        f.write('alignas(16) const unsigned char g_model[] = {\n')
        if not tflite_bytes:
            f.write('  0x00,\n')
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

        if is_transformer:
            f.write("const snn_config_t g_snn_config = {};\n")
            f.write("const float* const g_snn_conv_weights[] = {0};\n")
            f.write("const float* const g_snn_conv_biases[] = {0};\n")
            f.write("const float* const g_snn_fc_weights[] = {0};\n")
            f.write("const float* const g_snn_fc_biases[] = {0};\n\n")

            def emit_transformer_array(symbol: str, tensor: torch.Tensor) -> None:
                values = tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)
                f.write(f"static const float {symbol}[{values.size}] = {{\n")
                for start in range(0, values.size, 8):
                    literals = []
                    for value in values[start:start + 8]:
                        literal = f"{float(value):.9g}"
                        if "." not in literal and "e" not in literal.lower():
                            literal += ".0"
                        literals.append(literal + "f")
                    f.write("  " + ", ".join(literals) + ",\n")
                f.write("};\n")

            emit_transformer_array("s_transformer_input_w", model.input_projection.weight)
            emit_transformer_array("s_transformer_input_b", model.input_projection.bias)
            emit_transformer_array("s_transformer_position", model.position_embedding)
            emit_transformer_array("s_transformer_norm_w", model.norm.weight)
            emit_transformer_array("s_transformer_norm_b", model.norm.bias)
            emit_transformer_array("s_transformer_classifier_w", model.classifier.weight)
            emit_transformer_array("s_transformer_classifier_b", model.classifier.bias)

            layer_symbols = []
            for layer_idx, layer in enumerate(model.encoder.layers):
                names = {
                    "qkv_w": layer.self_attn.in_proj_weight,
                    "qkv_b": layer.self_attn.in_proj_bias,
                    "attn_out_w": layer.self_attn.out_proj.weight,
                    "attn_out_b": layer.self_attn.out_proj.bias,
                    "norm1_w": layer.norm1.weight,
                    "norm1_b": layer.norm1.bias,
                    "ff1_w": layer.linear1.weight,
                    "ff1_b": layer.linear1.bias,
                    "ff2_w": layer.linear2.weight,
                    "ff2_b": layer.linear2.bias,
                    "norm2_w": layer.norm2.weight,
                    "norm2_b": layer.norm2.bias,
                }
                symbols = {}
                for key, tensor in names.items():
                    symbol = f"s_transformer_l{layer_idx}_{key}"
                    emit_transformer_array(symbol, tensor)
                    symbols[key] = symbol
                layer_symbols.append(symbols)

            tokens = model.in_width
            features = model.in_height
            embed_dim = model.input_projection.out_features
            heads = model.encoder.layers[0].self_attn.num_heads
            layer_count = len(model.encoder.layers)
            ffn_dim = model.encoder.layers[0].linear1.out_features
            if (tokens, features, embed_dim, heads, layer_count, ffn_dim, output_size) != (16, 16, 32, 4, 2, 64, 35):
                raise ValueError("ESP32 Transformer export currently supports 16x16 input, D=32, 4 heads, 2 layers, FFN=64, and 35 classes")
            f.write("\nconst transformer_config_t g_transformer_config = {")
            is_spiking_transformer = bool(getattr(model, "is_spiking_transformer", False))
            transformer_steps = int(getattr(model, "num_steps", 1)) if is_spiking_transformer else 1
            transformer_beta = float(getattr(model, "beta", 0.0)) if is_spiking_transformer else 0.0
            transformer_threshold = float(getattr(model, "threshold", 1.0)) if is_spiking_transformer else 1.0
            transformer_input_scale = float(getattr(model, "input_scale", 1.0)) if is_spiking_transformer else 1.0
            f.write(f"{tokens}, {features}, {embed_dim}, {heads}, {layer_count}, {ffn_dim}, {output_size}, "
                    f"{1 if is_spiking_transformer else 0}, {transformer_steps}, {transformer_beta:.6f}f, "
                    f"{transformer_threshold:.6f}f, {transformer_input_scale:.6f}f" + "};\n")
            f.write("const float* const g_transformer_input_w = s_transformer_input_w;\n")
            f.write("const float* const g_transformer_input_b = s_transformer_input_b;\n")
            f.write("const float* const g_transformer_position = s_transformer_position;\n")
            f.write("const float* const g_transformer_norm_w = s_transformer_norm_w;\n")
            f.write("const float* const g_transformer_norm_b = s_transformer_norm_b;\n")
            f.write("const float* const g_transformer_classifier_w = s_transformer_classifier_w;\n")
            f.write("const float* const g_transformer_classifier_b = s_transformer_classifier_b;\n")
            f.write("const transformer_layer_weights_t g_transformer_layers[] = {\n")
            for symbols in layer_symbols:
                f.write("    {" + ", ".join(symbols[key] for key in ("qkv_w", "qkv_b", "attn_out_w", "attn_out_b", "norm1_w", "norm1_b", "ff1_w", "ff1_b", "ff2_w", "ff2_b", "norm2_w", "norm2_b")) + "},\n")
            f.write("};\n")
        elif model_type == "ann":
            f.write("const snn_config_t g_snn_config = {};\n")
            f.write("const float* const g_snn_conv_weights[] = {0};\n")
            f.write("const float* const g_snn_conv_biases[] = {0};\n")
            f.write("const float* const g_snn_fc_weights[] = {0};\n")
            f.write("const float* const g_snn_fc_biases[] = {0};\n")
        elif model_type == "snn":
            num_conv = len(getattr(model, "conv_layers_list", []))
            if num_conv > 0:
                for c_idx, conv in enumerate(model.conv_layers_list):
                    w = conv.weight.detach().cpu().numpy().astype(np.float32)
                    b = conv.bias.detach().cpu().numpy().astype(np.float32)
                    f.write(f"// SNN Conv {c_idx}: shape {w.shape}\n")
                    f.write(f"static const float s_snn_conv_w{c_idx}[{w.size}] = {{\n")
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

                    f.write(f"static const float s_snn_conv_b{c_idx}[{b.size}] = {{\n  ")
                    f.write(", ".join(f"{val:.7f}f" for val in b))
                    f.write("\n};\n\n")

                f.write("const float* const g_snn_conv_weights[] = {\n")
                for c_idx in range(num_conv):
                    f.write(f"    s_snn_conv_w{c_idx},\n")
                f.write("};\n\n")

                f.write("const float* const g_snn_conv_biases[] = {\n")
                for c_idx in range(num_conv):
                    f.write(f"    s_snn_conv_b{c_idx},\n")
                f.write("};\n\n")
            else:
                f.write("const float* const g_snn_conv_weights[] = {0};\n")
                f.write("const float* const g_snn_conv_biases[] = {0};\n\n")

            # FC layers
            fcs = model.fcs
            num_fc = len(fcs)
            for l_idx, fc in enumerate(fcs):
                w = fc.weight.detach().cpu().numpy().astype(np.float32)
                b = fc.bias.detach().cpu().numpy().astype(np.float32)

                f.write(f"// SNN FC layer {l_idx}: {w.shape[1]} -> {w.shape[0]}\n")
                f.write(f"static const float s_snn_fc_w{l_idx}[{w.size}] = {{\n")
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

                f.write(f"static const float s_snn_fc_b{l_idx}[{b.size}] = {{\n  ")
                f.write(", ".join(f"{val:.7f}f" for val in b))
                f.write("\n};\n\n")

            f.write("const float* const g_snn_fc_weights[] = {\n")
            for l_idx in range(num_fc):
                f.write(f"    s_snn_fc_w{l_idx},\n")
            f.write("};\n\n")

            f.write("const float* const g_snn_fc_biases[] = {\n")
            for l_idx in range(num_fc):
                f.write(f"    s_snn_fc_b{l_idx},\n")
            f.write("};\n\n")

            n_type_code = 1 if getattr(model, "neuron_type", "lif") == "if" else 0
            enc_code = get_encoding_code(getattr(model, "encoding_type", "direct"))
            exec_mode_code = 1 if getattr(model, "execution_mode", "dense") in ("event_based", "event_driven", "event") else 0
            fc_layer_sizes = [model.flat_size] + model.hidden_layers + [output_size]
            fc_sizes_str = ", ".join(str(s) for s in fc_layer_sizes)

            f.write("const snn_config_t g_snn_config = {\n")
            f.write(f"    .input_size = {input_size},\n")
            f.write(f"    .num_conv_layers = {num_conv},\n")
            f.write("    .conv_layers = {\n")
            if num_conv > 0:
                for cl in model.resolved_conv:
                    pad_h, pad_w = cl["padding"] if isinstance(cl["padding"], tuple) else (cl["padding"], cl["padding"])
                    f.write(
                        f"        {{{cl['in_channels']}, {cl['out_channels']}, {cl['kernel_h']}, {cl['kernel_w']}, "
                        f"{cl['stride']}, {pad_h}, {pad_w}, {cl['pool_h']}, {cl['pool_w']}, "
                        f"{cl['pool_stride_h']}, {cl['pool_stride_w']}, {1 if cl['pool_type'] == 'avg' else 0}, "
                        f"{cl['in_h']}, {cl['in_w']}, {cl['out_h']}, {cl['out_w']}, {cl['pool_out_h']}, {cl['pool_out_w']}}},\n"
                    )
            f.write("    },\n")
            f.write(f"    .flat_size = {model.flat_size},\n")
            f.write(f"    .hidden_layers_count = {len(model.hidden_layers)},\n")
            f.write(f"    .fc_layer_sizes = {{{fc_sizes_str}}},\n")
            f.write(f"    .num_steps = {model.num_steps},\n")
            f.write(f"    .beta = {model.beta:.4f}f,\n")
            f.write(f"    .threshold = {model.threshold:.4f}f,\n")
            f.write(f"    .output_size = {output_size},\n")
            f.write(f"    .neuron_type = {n_type_code},\n")
            f.write(f"    .encoding_type = {enc_code},\n")
            f.write(f"    .execution_mode = {exec_mode_code},\n")
            f.write(f"    .temporal_input = {1 if temporal_input else 0},\n")
            f.write(f"    .input_scale = {getattr(model, 'input_scale', 1.0):.6f}f,\n")
            f.write(f"    .analog_readout = {1 if getattr(model, 'output_readout', 'spike') == 'analog' else 0},\n")
            f.write("};\n")

        if dataset_name in ("speech_commands", "kws"):
            hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
            mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
            mel_points = np.linspace(hz_to_mel(20.0), hz_to_mel(7600.0), 42)
            hz_points = mel_to_hz(mel_points)
            fft_frequencies = np.linspace(0.0, 8000.0, 201)
            filterbank = np.zeros((40, 201), dtype=np.float32)
            for mel_idx in range(40):
                lower, center, upper = hz_points[mel_idx:mel_idx + 3]
                filterbank[mel_idx] = np.maximum(
                    0.0,
                    np.minimum((fft_frequencies - lower) / (center - lower),
                               (upper - fft_frequencies) / (upper - center)),
                )
            values = filterbank.reshape(-1)
            f.write(f"static const float s_speech_mel_filterbank[{values.size}] = {{\n")
            for start in range(0, values.size, 8):
                literals = []
                for value in values[start:start + 8]:
                    literal = f"{float(value):.9g}"
                    if "." not in literal and "e" not in literal.lower():
                        literal += ".0"
                    literals.append(literal + "f")
                f.write("  " + ", ".join(literals) + ",\n")
            f.write("};\nconst float* const g_speech_mel_filterbank = s_speech_mel_filterbank;\n")
        else:
            f.write("const float* const g_speech_mel_filterbank = nullptr;\n")

    # 3. mnist_test_data.h
    test_header_path = out_path / "mnist_test_data.h"
    with open(test_header_path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define DATASET_NAME \"{dataset_name}\"\n")
        f.write(f"#define DATASET_INPUT_SIZE {dataset_input_size}\n")
        f.write(f"#define DATASET_NUM_CLASSES {output_size}\n")
        f.write(f"#define HAS_SPEECH_AUDIO {1 if dataset_name in ('speech_commands', 'kws') else 0}\n")
        if dataset_name in ("speech_commands", "kws"):
            f.write("#define SPEECH_AUDIO_SAMPLES 16000\n")
        f.write("\n")
        f.write(f"#define MNIST_TEST_SUBSET {num_test_samples}\n")
        f.write(f"#define MNIST_IMAGE_SIZE {dataset_input_size}\n")
        f.write(f"#define MNIST_NUM_OUTPUTS {output_size}\n\n")

        labels = [int(test_dataset[i][1]) for i in range(num_test_samples)]
        f.write("static const uint8_t mnist_test_labels[MNIST_TEST_SUBSET] = {\n  ")
        f.write(", ".join(str(l) for l in labels))
        f.write("\n};\n\n")

        if dataset_name in ("speech_commands", "kws"):
            if not hasattr(test_dataset, "load_normalized_audio"):
                raise TypeError("Speech Commands ESP32 export requires load_normalized_audio() on the dataset")
            f.write("static const int16_t speech_test_audio[MNIST_TEST_SUBSET][SPEECH_AUDIO_SAMPLES] = {\n")
            for sample_idx in range(num_test_samples):
                pcm = np.rint(test_dataset.load_normalized_audio(sample_idx) * 32768.0)
                pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
                f.write("  {\n")
                for row in range(0, pcm.size, 16):
                    f.write("    " + ", ".join(str(int(v)) for v in pcm[row:row + 16]) + ",\n")
                f.write("  },\n")
            f.write("};\n\n")

        f.write("static const float mnist_test_images[MNIST_TEST_SUBSET][MNIST_IMAGE_SIZE] = {\n")
        for i in range(num_test_samples):
            img_tensor, _ = test_dataset[i]
            if temporal_input:
                # Store temporal SNN inputs in time-major order so each
                # firmware step reads a contiguous feature vector.
                img_np = img_tensor.reshape(1, model.in_channels * model.in_height, model.in_width)
                img_np = img_np.permute(2, 1, 0).reshape(-1).numpy()
            else:
                img_np = img_tensor.view(-1).numpy()
            f.write("  {\n")
            for row in range(0, dataset_input_size, 14):
                chunk = img_np[row:min(row + 14, dataset_input_size)]
                f.write("    " + ", ".join(f"{v:.4f}f" for v in chunk) + ",\n")
            f.write("  },\n")
        f.write("};\n")

    if getattr(model, "is_compact_transformer", False) or getattr(model, "is_spiking_transformer", False):
        size_desc = f"{model_entry.get('model_size_bytes', 0):,} bytes (FP32 weights)"
    else:
        size_desc = f"{len(model_entry.get('tflite_bytes', b'')):,} Bytes" if model_type == "ann" else f"{model_entry.get('model_size_bytes', 0):,} Bytes"
    print(f"[Export] Exported model '{model_name}' ({model_type.upper()}, {size_desc}) and {num_test_samples} test samples to '{output_dir}/':")
    print(f"  - {header_path}")
    print(f"  - {cc_path}")
    print(f"  - {test_header_path}\n")

def export_model_entry_to_main(model_entry: Dict[str, Any], output_dir: str = "main"):
    """
    Export a single trained model to 'main/model_data.h/cc'.
    """
    export_c_files(model_entry, output_dir=output_dir)
