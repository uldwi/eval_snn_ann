"""TFLite Micro graph export for feed-forward and convolutional SNNs.

The temporal dimension is statically unrolled. IF/LIF state updates are
expressed with TFLite MUL/ADD/GREATER/SELECT_V2 operators, so the generated
model uses the normal TFLM kernels and needs no custom runtime operator.
"""
from __future__ import annotations

from typing import Any

import flatbuffers
import numpy as np
import tflite


class _Graph:
    def __init__(self):
        self.builder = flatbuffers.Builder(1024 * 1024)
        self.buffers: list[int] = []
        self.tensors: list[int] = []
        self.operators: list[int] = []
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self.shapes: list[list[int]] = []
        self.opcodes: list[int] = []
        self.opcode_indices: dict[int, int] = {}
        self.buffer(np.empty((0,), dtype=np.uint8))

    def buffer(self, values: np.ndarray) -> int:
        b = self.builder
        data = b.CreateByteVector(np.ascontiguousarray(values).tobytes()) if values.size else 0
        tflite.BufferStart(b)
        if data:
            tflite.BufferAddData(b, data)
        result = tflite.BufferEnd(b)
        self.buffers.append(result)
        return len(self.buffers) - 1

    def tensor(self, name: str, shape: list[int], dtype: int = tflite.TensorType.FLOAT32,
               data: np.ndarray | None = None) -> int:
        buf = self.buffer(data) if data is not None else 0
        b = self.builder
        tflite.TensorStartShapeVector(b, len(shape))
        for dim in reversed(shape):
            b.PrependInt32(int(dim))
        dims = b.EndVector()
        name_ref = b.CreateString(name)
        tflite.TensorStart(b)
        tflite.TensorAddShape(b, dims)
        tflite.TensorAddType(b, dtype)
        tflite.TensorAddBuffer(b, buf)
        tflite.TensorAddName(b, name_ref)
        self.tensors.append(tflite.TensorEnd(b))
        self.shapes.append([int(dim) for dim in shape])
        return len(self.tensors) - 1

    def opcode(self, builtin: int) -> int:
        if builtin not in self.opcode_indices:
            b = self.builder
            tflite.OperatorCodeStart(b)
            tflite.OperatorCodeAddBuiltinCode(b, builtin)
            self.opcodes.append(tflite.OperatorCodeEnd(b))
            self.opcode_indices[builtin] = len(self.opcodes) - 1
        return self.opcode_indices[builtin]

    def op(self, builtin: int, inputs: list[int], output: int,
           options_type: int = tflite.BuiltinOptions.NONE, options: int = 0) -> None:
        b = self.builder
        tflite.OperatorStartInputsVector(b, len(inputs))
        for tensor_idx in reversed(inputs):
            b.PrependInt32(tensor_idx)
        in_vec = b.EndVector()
        tflite.OperatorStartOutputsVector(b, 1)
        b.PrependInt32(output)
        out_vec = b.EndVector()
        opcode_idx = self.opcode(builtin)
        tflite.OperatorStart(b)
        tflite.OperatorAddOpcodeIndex(b, opcode_idx)
        tflite.OperatorAddInputs(b, in_vec)
        tflite.OperatorAddOutputs(b, out_vec)
        if options:
            tflite.OperatorAddBuiltinOptionsType(b, options_type)
            tflite.OperatorAddBuiltinOptions(b, options)
        self.operators.append(tflite.OperatorEnd(b))

    def finish(self, description: str) -> bytes:
        b = self.builder
        tflite.SubGraphStartInputsVector(b, len(self.inputs))
        for idx in reversed(self.inputs):
            b.PrependInt32(idx)
        inputs = b.EndVector()
        tflite.SubGraphStartOutputsVector(b, len(self.outputs))
        for idx in reversed(self.outputs):
            b.PrependInt32(idx)
        outputs = b.EndVector()
        tflite.SubGraphStartTensorsVector(b, len(self.tensors))
        for item in reversed(self.tensors):
            b.PrependUOffsetTRelative(item)
        tensors = b.EndVector()
        tflite.SubGraphStartOperatorsVector(b, len(self.operators))
        for item in reversed(self.operators):
            b.PrependUOffsetTRelative(item)
        operators = b.EndVector()
        tflite.SubGraphStart(b)
        tflite.SubGraphAddTensors(b, tensors)
        tflite.SubGraphAddInputs(b, inputs)
        tflite.SubGraphAddOutputs(b, outputs)
        tflite.SubGraphAddOperators(b, operators)
        subgraph = tflite.SubGraphEnd(b)

        tflite.ModelStartOperatorCodesVector(b, len(self.opcodes))
        for item in reversed(self.opcodes):
            b.PrependUOffsetTRelative(item)
        opcodes = b.EndVector()
        tflite.ModelStartSubgraphsVector(b, 1)
        b.PrependUOffsetTRelative(subgraph)
        subgraphs = b.EndVector()
        tflite.ModelStartBuffersVector(b, len(self.buffers))
        for item in reversed(self.buffers):
            b.PrependUOffsetTRelative(item)
        buffers = b.EndVector()
        desc = b.CreateString(description)
        tflite.ModelStart(b)
        tflite.ModelAddVersion(b, 3)
        tflite.ModelAddOperatorCodes(b, opcodes)
        tflite.ModelAddSubgraphs(b, subgraphs)
        tflite.ModelAddBuffers(b, buffers)
        tflite.ModelAddDescription(b, desc)
        model = tflite.ModelEnd(b)
        b.Finish(model, b"TFL3")
        return bytes(b.Output())


def export_spiking_snn_to_tflite(model: Any) -> bytes:
    """Export a standard ConfigurableSpikingJellySNN as a static TFLM graph.

    One graph input is provided for each inference step. The ESP32 frontend
    performs only the input encoding (DIRECT/TTFS/PERIODIC) and fills those
    inputs. Neuron dynamics, convolutions, pooling, dense layers and readout
    run inside TFLite Micro.
    """
    if getattr(model, "is_spiking_transformer", False) or getattr(model, "is_compact_transformer", False):
        raise ValueError("TFLM SNN export currently supports standard FC/CNN SNNs, not Transformers.")
    if getattr(model, "temporal_input", False):
        raise ValueError("TFLM SNN export does not yet support temporal feature-sequence input.")

    steps = int(model.inference_steps)
    graph = _Graph()
    g = graph
    b = g.builder
    input_shape = ([1, model.in_height, model.in_width, model.in_channels]
                   if model.has_conv else [1, model.input_size])
    step_inputs = [g.tensor(f"encoded_input_t{t}", input_shape) for t in range(steps)]
    g.inputs.extend(step_inputs)

    scalar_zero = g.tensor("scalar_zero", [1], data=np.array([0.0], dtype=np.float32))
    scalar_one = g.tensor("scalar_one", [1], data=np.array([1.0], dtype=np.float32))
    scalar_threshold = g.tensor("threshold", [1], data=np.array([model.threshold], dtype=np.float32))
    scalar_beta = g.tensor("beta", [1], data=np.array([model.beta], dtype=np.float32))

    def binary_op(builtin: int, a: int, c: int, name: str,
                  opt_type: int = tflite.BuiltinOptions.NONE, options: int = 0) -> int:
        shape = _shape_of(g, a)
        out = g.tensor(name, shape)
        g.op(builtin, [a, c], out, opt_type, options)
        return out

    def neuron(current: int, previous: int | None, name: str) -> tuple[int, int]:
        if previous is None:
            previous = g.tensor(f"{name}_mem0", _shape_of(g, current),
                                data=np.zeros(_shape_of(g, current), dtype=np.float32))
        if float(model.beta) == 1.0:
            decayed = previous
        else:
            decayed = binary_op(tflite.BuiltinOperator.MUL, previous, scalar_beta,
                                f"{name}_decay", tflite.BuiltinOptions.MulOptions,
                                _empty_mul_options(b))
        potential = binary_op(tflite.BuiltinOperator.ADD, decayed, current,
                              f"{name}_potential", tflite.BuiltinOptions.AddOptions,
                              _empty_add_options(b))
        condition = g.tensor(f"{name}_thresholded", _shape_of(g, potential), tflite.TensorType.BOOL)
        g.op(tflite.BuiltinOperator.GREATER_EQUAL, [potential, scalar_threshold], condition,
             tflite.BuiltinOptions.GreaterEqualOptions, _empty_greater_equal_options(b))
        spikes = g.tensor(f"{name}_spikes", _shape_of(g, potential))
        g.op(tflite.BuiltinOperator.SELECT_V2, [condition, scalar_one, scalar_zero], spikes,
             tflite.BuiltinOptions.SelectV2Options, _empty_select_v2_options(b))
        membrane = g.tensor(f"{name}_membrane", _shape_of(g, potential))
        g.op(tflite.BuiltinOperator.SELECT_V2, [condition, scalar_zero, potential], membrane,
             tflite.BuiltinOptions.SelectV2Options, _empty_select_v2_options(b))
        return spikes, membrane

    conv_weights: list[int] = []
    conv_biases: list[int] = []
    for idx, conv in enumerate(model.conv_layers_list):
        cl = model.resolved_conv[idx]
        weight = conv.weight.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
        conv_weights.append(g.tensor(f"conv{idx}_weights", list(weight.shape), data=weight))
        bias = conv.bias.detach().cpu().numpy().astype(np.float32)
        conv_biases.append(g.tensor(f"conv{idx}_bias", list(bias.shape), data=bias))

    fc_weights: list[int] = []
    fc_biases: list[int] = []
    for idx, fc in enumerate(model.fcs):
        weight = fc.weight.detach().cpu().numpy().astype(np.float32)
        if idx == 0 and model.has_conv:
            last = model.resolved_conv[-1]
            weight = weight.reshape(fc.out_features, last["out_channels"], last["pool_out_h"], last["pool_out_w"])
            weight = weight.transpose(0, 2, 3, 1).reshape(fc.out_features, -1)
        fc_weights.append(g.tensor(f"fc{idx}_weights", list(weight.shape), data=weight))
        bias = fc.bias.detach().cpu().numpy().astype(np.float32)
        fc_biases.append(g.tensor(f"fc{idx}_bias", list(bias.shape), data=bias))

    conv_mem: list[int | None] = [None] * len(model.conv_layers_list)
    fc_mem: list[int | None] = [None] * len(model.fcs)
    readout: int | None = None
    for t, step_input in enumerate(step_inputs):
        current = step_input
        if model.has_conv:
            for idx, cl in enumerate(model.resolved_conv):
                conv = model.conv_layers_list[idx]
                conv_shape = [1, cl["out_h"], cl["out_w"], cl["out_channels"]]
                conv_out = g.tensor(f"conv{idx}_t{t}", conv_shape)
                options = _conv_options(b, cl)
                g.op(tflite.BuiltinOperator.CONV_2D,
                     [current, conv_weights[idx], conv_biases[idx]], conv_out,
                     tflite.BuiltinOptions.Conv2DOptions, options)
                current, conv_mem[idx] = neuron(conv_out, conv_mem[idx], f"conv{idx}_t{t}")
                if cl["pool_h"] > 1 or cl["pool_w"] > 1:
                    pool_shape = [1, cl["pool_out_h"], cl["pool_out_w"], cl["out_channels"]]
                    pooled = g.tensor(f"pool{idx}_t{t}", pool_shape)
                    pool_opts = _pool_options(b, cl)
                    pool_code = (tflite.BuiltinOperator.MAX_POOL_2D if cl["pool_type"] == "max"
                                 else tflite.BuiltinOperator.AVERAGE_POOL_2D)
                    g.op(pool_code, [current], pooled, tflite.BuiltinOptions.Pool2DOptions, pool_opts)
                    current = pooled
            flat = g.tensor(f"flatten_t{t}", [1, model.flat_size])
            shape_const = g.tensor(f"flatten_shape_t{t}", [2], tflite.TensorType.INT32,
                                   np.array([1, model.flat_size], dtype=np.int32))
            g.op(tflite.BuiltinOperator.RESHAPE, [current, shape_const], flat,
                 tflite.BuiltinOptions.ReshapeOptions, _reshape_options(b, [1, model.flat_size]))
            current = flat

        for idx, fc in enumerate(model.fcs):
            out = g.tensor(f"fc{idx}_t{t}", [1, fc.out_features])
            fc_options = _fc_options(b)
            g.op(tflite.BuiltinOperator.FULLY_CONNECTED,
                 [current, fc_weights[idx], fc_biases[idx]], out,
                 tflite.BuiltinOptions.FullyConnectedOptions, fc_options)
            is_analog_output = model.output_readout == "analog" and idx == len(model.fcs) - 1
            if is_analog_output:
                current = out
            else:
                current, fc_mem[idx] = neuron(out, fc_mem[idx], f"fc{idx}_t{t}")

        if readout is None:
            readout = current
        else:
            summed = g.tensor(f"readout_sum_t{t}", [1, model.output_size])
            add_opts = _empty_add_options(b)
            g.op(tflite.BuiltinOperator.ADD, [readout, current], summed,
                 tflite.BuiltinOptions.AddOptions, add_opts)
            readout = summed

    assert readout is not None
    g.outputs.append(readout)
    return g.finish(f"TFLM unrolled {model.get_structure_summary()}")


def _shape_of(graph: _Graph, tensor_idx: int) -> list[int]:
    # Shapes are tracked alongside FlatBuffer tensors to avoid reparsing tables.
    return graph.shapes[tensor_idx]


def _empty_add_options(builder):
    tflite.AddOptionsStart(builder)
    tflite.AddOptionsAddFusedActivationFunction(builder, tflite.ActivationFunctionType.NONE)
    return tflite.AddOptionsEnd(builder)


def _empty_mul_options(builder):
    tflite.MulOptionsStart(builder)
    tflite.MulOptionsAddFusedActivationFunction(builder, tflite.ActivationFunctionType.NONE)
    return tflite.MulOptionsEnd(builder)


def _empty_greater_equal_options(builder):
    tflite.GreaterEqualOptionsStart(builder)
    return tflite.GreaterEqualOptionsEnd(builder)


def _empty_select_v2_options(builder):
    tflite.SelectV2OptionsStart(builder)
    return tflite.SelectV2OptionsEnd(builder)


def _fc_options(builder):
    tflite.FullyConnectedOptionsStart(builder)
    tflite.FullyConnectedOptionsAddFusedActivationFunction(builder, tflite.ActivationFunctionType.NONE)
    tflite.FullyConnectedOptionsAddWeightsFormat(builder, tflite.FullyConnectedOptionsWeightsFormat.DEFAULT)
    tflite.FullyConnectedOptionsAddKeepNumDims(builder, False)
    return tflite.FullyConnectedOptionsEnd(builder)


def _conv_options(builder, cl: dict[str, Any]) -> int:
    padding = cl["padding"]
    if isinstance(padding, tuple):
        if padding[0] != padding[1]:
            raise ValueError("TFLM SNN export currently requires symmetric Conv2D padding.")
        padding = padding[0]
    if int(cl["stride"]) != 1:
        raise ValueError("TFLM SNN export currently requires Conv2D stride 1.")
    tflite.Conv2DOptionsStart(builder)
    tflite.Conv2DOptionsAddPadding(builder, tflite.Padding.SAME if int(padding) else tflite.Padding.VALID)
    tflite.Conv2DOptionsAddStrideH(builder, int(cl["stride"]))
    tflite.Conv2DOptionsAddStrideW(builder, int(cl["stride"]))
    tflite.Conv2DOptionsAddDilationHFactor(builder, 1)
    tflite.Conv2DOptionsAddDilationWFactor(builder, 1)
    tflite.Conv2DOptionsAddFusedActivationFunction(builder, tflite.ActivationFunctionType.NONE)
    return tflite.Conv2DOptionsEnd(builder)


def _pool_options(builder, cl: dict[str, Any]) -> int:
    tflite.Pool2DOptionsStart(builder)
    tflite.Pool2DOptionsAddPadding(builder, tflite.Padding.VALID)
    tflite.Pool2DOptionsAddStrideH(builder, int(cl["pool_stride_h"]))
    tflite.Pool2DOptionsAddStrideW(builder, int(cl["pool_stride_w"]))
    tflite.Pool2DOptionsAddFilterHeight(builder, int(cl["pool_h"]))
    tflite.Pool2DOptionsAddFilterWidth(builder, int(cl["pool_w"]))
    tflite.Pool2DOptionsAddFusedActivationFunction(builder, tflite.ActivationFunctionType.NONE)
    return tflite.Pool2DOptionsEnd(builder)


def _reshape_options(builder, shape: list[int]) -> int:
    tflite.ReshapeOptionsStartNewShapeVector(builder, len(shape))
    for dim in reversed(shape):
        builder.PrependInt32(dim)
    dims = builder.EndVector()
    tflite.ReshapeOptionsStart(builder)
    tflite.ReshapeOptionsAddNewShape(builder, dims)
    return tflite.ReshapeOptionsEnd(builder)
