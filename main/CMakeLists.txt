idf_component_register(SRCS "eval_snn_ann.cc" "model_data.cc"
                    INCLUDE_DIRS "."
                    REQUIRES esp-tflite-micro driver esp_timer)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3)

# Select SNN kernel variants; override with -DSNN_SEL_...=<0|1>.
# Measured on ESP32-C6 (500 MNIST images, TTFS, dense; average of three
# measurement cycles; predictions and spike counts were identical in all variants):
#
#   Variante                          ms/Inferenz
#   FP32 FC, explicit decay               342.2   (baseline)
#   + INT8 FC + lazy decay                297.4
#   + beta==1 convolution                 258.1
#   + tap index table                     226.1
#   + branch instead of branchless pass   204.4
#   + output-stationary scatter           196.0   (default)
#
#   SNN_SEL_LAYOUT_NHWC  1 = NHWC layout (oc minor). Consistently slower on C6
#                            than NCHW (345 vs. 342 ms): 1 = disabled
#   SNN_SEL_INT8         INT8 convolution weights. Without SNN_SEL_CONV_ACC32,
#                            each weight must be dequantized; on a core without
#                            vector instructions, fcvt.w.s becomes the
#                            bottleneck. With ACC32: 297.9 -> 292.9 ms and
#                            69 KB less free heap: 0 = disabled
#   SNN_SEL_INT8_FC      INT8 fully connected weights. The gain comes not from
#                            the arithmetic itself, but from keeping weights
#                            in SRAM instead of the flash cache: FC time
#                            1882 ms -> 171 ms: 1 = enabled
#   SNN_SEL_CONV_ACC32   Separate INT32 accumulator per convolution neuron;
#                            costs 4 bytes per neuron (75 KB in this model):
#                            0 = disabled
#   SNN_SEL_LAZY_DECAY   Apply decay/bias lazily in closed form instead of in
#                            a separate pass: 350 ms -> 293 ms: 1 = enabled
#   SNN_SEL_DENSE_REF    0 = disabled; otherwise number of test samples for
#                            self-test against dense reference path (verification
#                            only; much slower): 0 = disabled
#   SNN_PROFILE          Timing breakdown for conv/scatter/threshold/pool/FC
#                            (adds esp_timer calls per layer and time step):
#                            0 = disabled
#
# Unless overridden by SNN_SEL_*, a beta==1 special path is also enabled:
# when beta = 1, decay is a no-op and bias can be separated from state
# (mu = mem - b*last, v = mu + b*stamp). This eliminates decay updates during
# scatter and lets threshold checks use one access per neuron. Other switches
# remain available for comparison.
set(SNN_SEL_LAYOUT_NHWC 0 CACHE STRING "NHWC layout")
set(SNN_SEL_INT8        0 CACHE STRING "INT8 convolution weights")
set(SNN_SEL_INT8_FC     0 CACHE STRING "INT8 fully connected weights")
set(SNN_SEL_CONV_ACC32  0 CACHE STRING "INT32 accumulator per convolution neuron")
set(SNN_SEL_LAZY_DECAY  0 CACHE STRING "Lazy Decay")
set(SNN_SEL_DENSE_REF   0 CACHE STRING "Samples for dense reference path self-test")
set(SNN_PROFILE         0 CACHE STRING "Timing breakdown")

target_compile_definitions(${COMPONENT_LIB} PRIVATE
    SNN_SEL_LAYOUT_NHWC=${SNN_SEL_LAYOUT_NHWC}
    SNN_SEL_INT8=${SNN_SEL_INT8}
    SNN_SEL_INT8_FC=${SNN_SEL_INT8_FC}
    SNN_SEL_CONV_ACC32=${SNN_SEL_CONV_ACC32}
    SNN_SEL_LAZY_DECAY=${SNN_SEL_LAZY_DECAY}
    SNN_SEL_DENSE_REF=${SNN_SEL_DENSE_REF}
    SNN_PROFILE=${SNN_PROFILE})
