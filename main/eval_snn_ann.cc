#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_random.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"

#ifndef HOST_TEST
// TensorFlow Lite Micro (for ANN)
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif

#include "model_data.h"
#include "mnist_test_data.h"

#ifndef HAS_SPEECH_AUDIO
#define HAS_SPEECH_AUDIO 0
#endif

#define TRIGGER_GPIO GPIO_NUM_4

static const char *TAG = "EVAL_SNN_ANN";

// Tensor arena for TensorFlow Lite Micro (ANN only)
constexpr size_t kTensorArenaSize = 192 * 1024;
static uint8_t *tensor_arena = nullptr;

static int argmax_float(const float *arr, int len) {
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < len; ++i) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// ESP32-S3 FP32 inference for the compact Transformer defined in Python.
// Export dimensions are intentionally fixed so the workspace
// remains static and large temporary tensors do not use the task stack.
static constexpr int KWS_TOKENS = 16;
static constexpr int KWS_FEATURES = 16;
static constexpr int KWS_EMBED = 32;
static constexpr int KWS_HEADS = 4;
static constexpr int KWS_HEAD_DIM = KWS_EMBED / KWS_HEADS;
static constexpr int KWS_FFN = 64;
static constexpr int KWS_LAYERS = 2;
static constexpr int KWS_CLASSES = 35;

#if HAS_SPEECH_AUDIO
static constexpr int SPEECH_SAMPLE_RATE = 16000;
static constexpr int SPEECH_NFFT = 400;
static constexpr int SPEECH_BINS = 201;
static constexpr int SPEECH_HOP = 160;
static constexpr int SPEECH_FRAMES = 101;
static constexpr int SPEECH_MELS = 40;

struct SpeechComplex { float re, im; };
static float s_speech_hann[SPEECH_NFFT];
static SpeechComplex s_speech_twiddle[SPEECH_NFFT];
static SpeechComplex s_speech_fft_input[SPEECH_NFFT];
static SpeechComplex s_speech_fft_output[SPEECH_NFFT];
static SpeechComplex s_speech_fft_work[8][SPEECH_NFFT];
static float s_speech_mel_power[SPEECH_MELS * SPEECH_FRAMES];
static float s_speech_features[KWS_FEATURES * KWS_TOKENS];
static bool s_speech_frontend_ready = false;

static void init_speech_frontend(void) {
    constexpr float pi = 3.14159265358979323846f;
    for (int n = 0; n < SPEECH_NFFT; ++n) {
        s_speech_hann[n] = 0.5f - 0.5f * cosf((2.0f * pi * n) / SPEECH_NFFT);
        const float angle = (-2.0f * pi * n) / SPEECH_NFFT;
        s_speech_twiddle[n] = {cosf(angle), sinf(angle)};
    }
    s_speech_frontend_ready = true;
}

static void speech_fft400(const SpeechComplex *src, int stride, int n,
                          SpeechComplex *dst, int level) {
    if (n == 1) {
        dst[0] = *src;
        return;
    }
    const int radix = (n % 2 == 0) ? 2 : 5;
    const int sub_n = n / radix;
    SpeechComplex *sub = s_speech_fft_work[level];
    for (int j = 0; j < radix; ++j) {
        speech_fft400(src + j * stride, stride * radix, sub_n,
                      sub + j * sub_n, level + 1);
    }
    const int twiddle_scale = SPEECH_NFFT / n;
    for (int q = 0; q < radix; ++q) {
        for (int k = 0; k < sub_n; ++k) {
            const int out_idx = k + q * sub_n;
            float real = 0.0f;
            float imag = 0.0f;
            for (int j = 0; j < radix; ++j) {
                const int phase = (j * out_idx * twiddle_scale) % SPEECH_NFFT;
                const SpeechComplex a = sub[j * sub_n + k];
                const SpeechComplex w = s_speech_twiddle[phase];
                real += a.re * w.re - a.im * w.im;
                imag += a.re * w.im + a.im * w.re;
            }
            dst[out_idx] = {real, imag};
        }
    }
}

static int speech_reflect_index(int index) {
    while (index < 0 || index >= SPEECH_SAMPLE_RATE) {
        if (index < 0) index = -index;
        else index = 2 * SPEECH_SAMPLE_RATE - 2 - index;
    }
    return index;
}

static void speech_audio_to_mel(const int16_t *audio, float *features) {
    if (!s_speech_frontend_ready) init_speech_frontend();
    float mel_max = -100.0f;
    for (int frame = 0; frame < SPEECH_FRAMES; ++frame) {
        const int center = frame * SPEECH_HOP;
        for (int n = 0; n < SPEECH_NFFT; ++n) {
            const int sample_idx = speech_reflect_index(center + n - SPEECH_NFFT / 2);
            s_speech_fft_input[n] = {
                (static_cast<float>(audio[sample_idx]) / 32768.0f) * s_speech_hann[n], 0.0f
            };
        }
        speech_fft400(s_speech_fft_input, 1, SPEECH_NFFT, s_speech_fft_output, 0);
        for (int mel = 0; mel < SPEECH_MELS; ++mel) {
            float power = 0.0f;
            const float *filter = g_speech_mel_filterbank + mel * SPEECH_BINS;
            for (int bin = 0; bin < SPEECH_BINS; ++bin) {
                const SpeechComplex value = s_speech_fft_output[bin];
                power += (value.re * value.re + value.im * value.im) * filter[bin];
            }
            const float log_power = 10.0f * log10f(fmaxf(power, 1.0e-10f));
            s_speech_mel_power[mel * SPEECH_FRAMES + frame] = log_power;
            if (log_power > mel_max) mel_max = log_power;
        }
    }

    for (int mel = 0; mel < SPEECH_MELS; ++mel) {
        for (int frame = 0; frame < SPEECH_FRAMES; ++frame) {
            float value = s_speech_mel_power[mel * SPEECH_FRAMES + frame] - mel_max;
            value = fmaxf(-80.0f, fminf(0.0f, value));
            s_speech_mel_power[mel * SPEECH_FRAMES + frame] = (value + 80.0f) / 80.0f;
        }
    }

    for (int out_mel = 0; out_mel < KWS_FEATURES; ++out_mel) {
        const int m_start = (out_mel * SPEECH_MELS) / KWS_FEATURES;
        const int m_end = ((out_mel + 1) * SPEECH_MELS + KWS_FEATURES - 1) / KWS_FEATURES;
        for (int out_time = 0; out_time < KWS_TOKENS; ++out_time) {
            const int t_start = (out_time * SPEECH_FRAMES) / KWS_TOKENS;
            const int t_end = ((out_time + 1) * SPEECH_FRAMES + KWS_TOKENS - 1) / KWS_TOKENS;
            float sum = 0.0f;
            for (int m = m_start; m < m_end; ++m) {
                for (int t = t_start; t < t_end; ++t) {
                    sum += s_speech_mel_power[m * SPEECH_FRAMES + t];
                }
            }
            const int area = (m_end - m_start) * (t_end - t_start);
            features[out_mel * KWS_TOKENS + out_time] = sum / static_cast<float>(area);
        }
    }
}
#endif

static float s_tr_x[KWS_TOKENS * KWS_EMBED];
static float s_tr_norm[KWS_TOKENS * KWS_EMBED];
static float s_tr_qkv[KWS_TOKENS * 3 * KWS_EMBED];
static float s_tr_attn[KWS_TOKENS * KWS_EMBED];
static float s_tr_proj[KWS_TOKENS * KWS_EMBED];
static float s_tr_ff1[KWS_TOKENS * KWS_FFN];
static float s_tr_ff2[KWS_TOKENS * KWS_EMBED];
static float s_tr_scores[KWS_TOKENS * KWS_TOKENS];
static float s_tr_logits[KWS_CLASSES];
static uint32_t s_transformer_spikes = 0;
static uint32_t s_transformer_synops = 0;
static float s_st_mem_input[KWS_TOKENS * KWS_EMBED];
static float s_st_mem_qkv[KWS_LAYERS][3 * KWS_TOKENS * KWS_EMBED];
static float s_st_mem_attn[KWS_LAYERS][KWS_TOKENS * KWS_EMBED];
static float s_st_mem_ff[KWS_LAYERS][KWS_TOKENS * KWS_EMBED];
static float s_st_spike_accum[KWS_TOKENS * KWS_EMBED];

static void transformer_linear(const float *input, int rows, int in_dim,
                               const float *weight, const float *bias,
                               int out_dim, float *output) {
    for (int r = 0; r < rows; ++r) {
        const float *x = input + r * in_dim;
        float *y = output + r * out_dim;
        for (int o = 0; o < out_dim; ++o) {
            float sum = bias ? bias[o] : 0.0f;
            const float *w = weight + o * in_dim;
            for (int i = 0; i < in_dim; ++i) sum += w[i] * x[i];
            y[o] = sum;
        }
    }
}

static void transformer_layer_norm(const float *input, int rows,
                                   const float *weight, const float *bias,
                                   float *output) {
    constexpr float eps = 1.0e-5f;
    for (int r = 0; r < rows; ++r) {
        const float *x = input + r * KWS_EMBED;
        float *y = output + r * KWS_EMBED;
        float mean = 0.0f;
        for (int i = 0; i < KWS_EMBED; ++i) mean += x[i];
        mean /= static_cast<float>(KWS_EMBED);
        float variance = 0.0f;
        for (int i = 0; i < KWS_EMBED; ++i) {
            const float delta = x[i] - mean;
            variance += delta * delta;
        }
        variance /= static_cast<float>(KWS_EMBED);
        const float inv_std = 1.0f / sqrtf(variance + eps);
        for (int i = 0; i < KWS_EMBED; ++i) y[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

static inline float transformer_lif(float current, float &membrane,
                                    float beta, float threshold) {
    // Match SpikingJelly LIFNode(decay_input=False):
    // v[t] = beta * v[t-1] + current.  Scaling the input by (1-beta)
    // changes the learned network's firing rates and breaks host parity.
    membrane = beta * membrane + current;
    if (membrane >= threshold) {
        membrane = 0.0f;
        return 1.0f;
    }
    return 0.0f;
}

static int run_spiking_transformer(const float *mel) {
    const float beta = g_transformer_config.beta;
    const float threshold = g_transformer_config.threshold;
    const float input_scale = g_transformer_config.input_scale > 0.0f
                                  ? g_transformer_config.input_scale : 1.0f;
    const int steps = g_transformer_config.num_steps;
    memset(s_st_mem_input, 0, sizeof(s_st_mem_input));
    memset(s_st_mem_qkv, 0, sizeof(s_st_mem_qkv));
    memset(s_st_mem_attn, 0, sizeof(s_st_mem_attn));
    memset(s_st_mem_ff, 0, sizeof(s_st_mem_ff));
    memset(s_st_spike_accum, 0, sizeof(s_st_spike_accum));
    s_transformer_spikes = 0;
    s_transformer_synops = 0;

    for (int step = 0; step < steps; ++step) {
        for (int t = 0; t < KWS_TOKENS; ++t) {
            for (int d = 0; d < KWS_EMBED; ++d) {
                float current = g_transformer_input_b[d];
                const float *w = g_transformer_input_w + d * KWS_FEATURES;
                for (int f = 0; f < KWS_FEATURES; ++f) current += w[f] * mel[f * KWS_TOKENS + t];
                current += g_transformer_position[t * KWS_EMBED + d];
                current *= input_scale;
                float &mem = s_st_mem_input[t * KWS_EMBED + d];
                const float spike = transformer_lif(current, mem, beta, threshold);
                s_tr_x[t * KWS_EMBED + d] = spike;
                s_transformer_spikes += static_cast<uint32_t>(spike);
            }
        }

        for (int layer_idx = 0; layer_idx < KWS_LAYERS; ++layer_idx) {
            const transformer_layer_weights_t &layer = g_transformer_layers[layer_idx];
            transformer_layer_norm(s_tr_x, KWS_TOKENS, layer.norm1_w, layer.norm1_b, s_tr_norm);
            transformer_linear(s_tr_norm, KWS_TOKENS, KWS_EMBED, layer.qkv_w, layer.qkv_b,
                               3 * KWS_EMBED, s_tr_qkv);
            uint32_t qkv_in_spikes = 0;
            for (int i = 0; i < KWS_TOKENS * KWS_EMBED; ++i) qkv_in_spikes += static_cast<uint32_t>(s_tr_x[i]);
            s_transformer_synops += qkv_in_spikes * (3 * KWS_EMBED);
            for (int i = 0; i < 3 * KWS_TOKENS * KWS_EMBED; ++i) {
                s_tr_qkv[i] = transformer_lif(s_tr_qkv[i], s_st_mem_qkv[layer_idx][i], beta, threshold);
                s_transformer_spikes += static_cast<uint32_t>(s_tr_qkv[i]);
            }

            for (int h = 0; h < KWS_HEADS; ++h) {
                const int head_offset = h * KWS_HEAD_DIM;
                for (int q = 0; q < KWS_TOKENS; ++q) {
                    float *scores = s_tr_scores + q * KWS_TOKENS;
                    float max_score = -INFINITY;
                    for (int k = 0; k < KWS_TOKENS; ++k) {
                        float score = 0.0f;
                        for (int d = 0; d < KWS_HEAD_DIM; ++d) {
                            score += s_tr_qkv[q * (3 * KWS_EMBED) + head_offset + d] *
                                     s_tr_qkv[k * (3 * KWS_EMBED) + KWS_EMBED + head_offset + d];
                        }
                        scores[k] = score / sqrtf(static_cast<float>(KWS_HEAD_DIM));
                        if (scores[k] > max_score) max_score = scores[k];
                    }
                    float denom = 0.0f;
                    for (int k = 0; k < KWS_TOKENS; ++k) {
                        scores[k] = expf(scores[k] - max_score);
                        denom += scores[k];
                    }
                    for (int d = 0; d < KWS_HEAD_DIM; ++d) {
                        float value = 0.0f;
                        for (int k = 0; k < KWS_TOKENS; ++k) {
                            value += (scores[k] / denom) *
                                     s_tr_qkv[k * (3 * KWS_EMBED) + 2 * KWS_EMBED + head_offset + d];
                        }
                        s_tr_attn[q * KWS_EMBED + head_offset + d] = value;
                    }
                }
            }

            transformer_linear(s_tr_attn, KWS_TOKENS, KWS_EMBED,
                               layer.attn_out_w, layer.attn_out_b, KWS_EMBED, s_tr_proj);
            for (int i = 0; i < KWS_TOKENS * KWS_EMBED; ++i) {
                const float spike = transformer_lif(s_tr_x[i] + s_tr_proj[i],
                                                     s_st_mem_attn[layer_idx][i], beta, threshold);
                s_tr_x[i] = spike;
                s_transformer_spikes += static_cast<uint32_t>(spike);
            }

            uint32_t mlp_in_spikes = 0;
            for (int i = 0; i < KWS_TOKENS * KWS_EMBED; ++i) mlp_in_spikes += static_cast<uint32_t>(s_tr_x[i]);
            s_transformer_synops += mlp_in_spikes * KWS_FFN;
            transformer_layer_norm(s_tr_x, KWS_TOKENS, layer.norm2_w, layer.norm2_b, s_tr_norm);
            transformer_linear(s_tr_norm, KWS_TOKENS, KWS_EMBED, layer.ff1_w, layer.ff1_b,
                               KWS_FFN, s_tr_ff1);
            constexpr float inv_sqrt2 = 0.7071067811865475f;
            for (int i = 0; i < KWS_TOKENS * KWS_FFN; ++i) {
                s_tr_ff1[i] = 0.5f * s_tr_ff1[i] * (1.0f + erff(s_tr_ff1[i] * inv_sqrt2));
            }
            transformer_linear(s_tr_ff1, KWS_TOKENS, KWS_FFN, layer.ff2_w, layer.ff2_b,
                               KWS_EMBED, s_tr_ff2);
            for (int i = 0; i < KWS_TOKENS * KWS_EMBED; ++i) {
                const float spike = transformer_lif(s_tr_x[i] + s_tr_ff2[i],
                                                     s_st_mem_ff[layer_idx][i], beta, threshold);
                s_tr_x[i] = spike;
                s_transformer_spikes += static_cast<uint32_t>(spike);
            }
        }
        for (int i = 0; i < KWS_TOKENS * KWS_EMBED; ++i) s_st_spike_accum[i] += s_tr_x[i];
    }

    const float denom = static_cast<float>(steps * KWS_TOKENS);
    for (int d = 0; d < KWS_EMBED; ++d) {
        float pooled = 0.0f;
        for (int t = 0; t < KWS_TOKENS; ++t) pooled += s_st_spike_accum[t * KWS_EMBED + d];
        s_tr_norm[d] = pooled / denom;
    }
    transformer_layer_norm(s_tr_norm, 1, g_transformer_norm_w, g_transformer_norm_b, s_tr_proj);
    transformer_linear(s_tr_proj, 1, KWS_EMBED, g_transformer_classifier_w,
                       g_transformer_classifier_b, KWS_CLASSES, s_tr_logits);
    return argmax_float(s_tr_logits, KWS_CLASSES);
}

static int run_transformer_inference(const float *mel) {
    if (g_transformer_config.spiking) return run_spiking_transformer(mel);
    // Input is (frequency, time); the ANN forward pass uses time as the token axis.
    for (int t = 0; t < KWS_TOKENS; ++t) {
        float *token = s_tr_x + t * KWS_EMBED;
        for (int d = 0; d < KWS_EMBED; ++d) {
            float value = g_transformer_input_b[d];
            const float *w = g_transformer_input_w + d * KWS_FEATURES;
            for (int f = 0; f < KWS_FEATURES; ++f) value += w[f] * mel[f * KWS_TOKENS + t];
            token[d] = value + g_transformer_position[t * KWS_EMBED + d];
        }
    }

    for (int layer_idx = 0; layer_idx < KWS_LAYERS; ++layer_idx) {
        const transformer_layer_weights_t &layer = g_transformer_layers[layer_idx];
        transformer_layer_norm(s_tr_x, KWS_TOKENS, layer.norm1_w, layer.norm1_b, s_tr_norm);
        transformer_linear(s_tr_norm, KWS_TOKENS, KWS_EMBED, layer.qkv_w, layer.qkv_b,
                           3 * KWS_EMBED, s_tr_qkv);

        for (int h = 0; h < KWS_HEADS; ++h) {
            const int head_offset = h * KWS_HEAD_DIM;
            for (int q = 0; q < KWS_TOKENS; ++q) {
                float *scores = s_tr_scores + q * KWS_TOKENS;
                float max_score = -INFINITY;
                for (int k = 0; k < KWS_TOKENS; ++k) {
                    float score = 0.0f;
                    for (int d = 0; d < KWS_HEAD_DIM; ++d) {
                        const float qv = s_tr_qkv[q * (3 * KWS_EMBED) + head_offset + d];
                        const float kv = s_tr_qkv[k * (3 * KWS_EMBED) + KWS_EMBED + head_offset + d];
                        score += qv * kv;
                    }
                    scores[k] = score / sqrtf(static_cast<float>(KWS_HEAD_DIM));
                    if (scores[k] > max_score) max_score = scores[k];
                }
                float denom = 0.0f;
                for (int k = 0; k < KWS_TOKENS; ++k) {
                    scores[k] = expf(scores[k] - max_score);
                    denom += scores[k];
                }
                const float inv_denom = 1.0f / denom;
                for (int d = 0; d < KWS_HEAD_DIM; ++d) {
                    float value = 0.0f;
                    for (int k = 0; k < KWS_TOKENS; ++k) {
                        const float probability = scores[k] * inv_denom;
                        const float vv = s_tr_qkv[k * (3 * KWS_EMBED) + 2 * KWS_EMBED + head_offset + d];
                        value += probability * vv;
                    }
                    s_tr_attn[q * KWS_EMBED + head_offset + d] = value;
                }
            }
        }

        transformer_linear(s_tr_attn, KWS_TOKENS, KWS_EMBED,
                           layer.attn_out_w, layer.attn_out_b, KWS_EMBED, s_tr_proj);
        for (int i = 0; i < KWS_TOKENS * KWS_EMBED; ++i) s_tr_x[i] += s_tr_proj[i];

        transformer_layer_norm(s_tr_x, KWS_TOKENS, layer.norm2_w, layer.norm2_b, s_tr_norm);
        transformer_linear(s_tr_norm, KWS_TOKENS, KWS_EMBED,
                           layer.ff1_w, layer.ff1_b, KWS_FFN, s_tr_ff1);
        constexpr float inv_sqrt2 = 0.7071067811865475f;
        for (int i = 0; i < KWS_TOKENS * KWS_FFN; ++i) {
            s_tr_ff1[i] = 0.5f * s_tr_ff1[i] * (1.0f + erff(s_tr_ff1[i] * inv_sqrt2));
        }
        transformer_linear(s_tr_ff1, KWS_TOKENS, KWS_FFN,
                           layer.ff2_w, layer.ff2_b, KWS_EMBED, s_tr_ff2);
        for (int i = 0; i < KWS_TOKENS * KWS_EMBED; ++i) s_tr_x[i] += s_tr_ff2[i];
    }

    for (int d = 0; d < KWS_EMBED; ++d) {
        float mean = 0.0f;
        for (int t = 0; t < KWS_TOKENS; ++t) mean += s_tr_x[t * KWS_EMBED + d];
        s_tr_norm[d] = mean / static_cast<float>(KWS_TOKENS);
    }
    transformer_layer_norm(s_tr_norm, 1, g_transformer_norm_w, g_transformer_norm_b, s_tr_proj);
    transformer_linear(s_tr_proj, 1, KWS_EMBED, g_transformer_classifier_w,
                       g_transformer_classifier_b, KWS_CLASSES, s_tr_logits);
    return argmax_float(s_tr_logits, KWS_CLASSES);
}

// ---------------------------------------------------------------------
// Kernel variants
//
//   SNN_SEL_LAYOUT_NHWC  0 = NCHW (channel-major, oc*plane)
//                        1 = NHWC (pixel-major, oc innermost -> contiguous
//                            scatter writes)
//   SNN_SEL_INT8         0 = FP32 weights, FP32 accumulation
//                        1 = INT8 weights (scaled per output channel),
//                            convolution requires an INT32 accumulator (SNN_SEL_CONV_ACC32),
//                            otherwise dequantization per weight on
//                            a core without vector instructions becomes the bottleneck
//   SNN_SEL_INT8_FC      INT8 for FC weights only (benefit: weights
//                        reside in SRAM instead of the flash cache)
//   SNN_SEL_CONV_ACC32   INT8 only: additional INT32 accumulator per
//                        convolution neuron (4 extra bytes per neuron) instead of
//                        dequantization per weight during scatter
//   SNN_SEL_LAZY_DECAY   0 = explicit decay/bias pass at each time step
//                        1 = lazy decay in closed form; decay is applied
//                            when a neuron is accessed
//   SNN_SEL_DENSE_REF    0 = disabled; > 0 = number of test samples for which
//                        the additional dense reference path is checked against
//                        the optimized path. The reference path is much
//                        slower, so use it only for self-tests.
//
// With SNN_SEL_RUNTIME=1, all variants are compiled and selected at runtime.
// selectable through snn_select_variants() (host verification).
// ---------------------------------------------------------------------
#ifndef SNN_SEL_LAYOUT_NHWC
#define SNN_SEL_LAYOUT_NHWC 0
#endif
#ifndef SNN_SEL_INT8
#define SNN_SEL_INT8 0
#endif
#ifndef SNN_SEL_INT8_FC
#define SNN_SEL_INT8_FC SNN_SEL_INT8
#endif
#ifndef SNN_SEL_CONV_ACC32
#define SNN_SEL_CONV_ACC32 0
#endif
#ifndef SNN_SEL_LAZY_DECAY
#define SNN_SEL_LAZY_DECAY 0
#endif
#ifndef SNN_SEL_DENSE_REF
#define SNN_SEL_DENSE_REF 0
#endif
#ifndef SNN_SEL_RUNTIME
#define SNN_SEL_RUNTIME 0
#endif
#ifndef SNN_PROFILE
#define SNN_PROFILE 0
#endif

#if SNN_PROFILE
static uint64_t s_cycle_delta;
static uint64_t s_prof[6];   // Conv-Scatter, Conv-Schwellenpass, Pool, FC in Mikrosekunden, Events, MACs
#define PROF_T(v)    const int64_t v = esp_timer_get_time()
#define PROF_ADD(v, slot) s_prof[slot] += (uint64_t)(esp_timer_get_time() - (v))
extern "C" void snn_prof_reset(void) { for (int i = 0; i < 6; ++i) s_prof[i] = 0; }
extern "C" void snn_prof_get(uint64_t *c, uint64_t *p, uint64_t *f) { *c = s_prof[0] + s_prof[1]; *p = s_prof[2]; *f = s_prof[3]; }
extern "C" void snn_prof_get_split(uint64_t *cs, uint64_t *ce) { *cs = s_prof[0]; *ce = s_prof[1]; }
extern "C" void snn_prof_get_counts(uint64_t *ev, uint64_t *macs) { *ev = s_prof[4]; *macs = s_prof[5]; }
#else
#define PROF_T(v)    ((void)0)
#define PROF_ADD(v, slot) ((void)0)
#endif

#if SNN_SEL_RUNTIME
static int s_opt_nhwc  = SNN_SEL_LAYOUT_NHWC;
static int s_opt_convq = SNN_SEL_INT8;
static int s_opt_acc32 = SNN_SEL_CONV_ACC32;
static int s_opt_lazy  = SNN_SEL_LAZY_DECAY;
static int s_opt_fcq   = SNN_SEL_INT8_FC;
#endif

// Active configuration as compile-time constants for production builds
#define SNN_CFG_NHWC  SNN_SEL_LAYOUT_NHWC
#define SNN_CFG_INT8  SNN_SEL_INT8
#define SNN_CFG_INT8_FC SNN_SEL_INT8_FC
#define SNN_CFG_ACC32 SNN_SEL_CONV_ACC32
#define SNN_CFG_LAZY  SNN_SEL_LAZY_DECAY

// ---------------------------------------------------------------------
// Vorberechnete Layer-Geometrie (vermeidet Divisionen und
// Indexarithmetik in den innersten Schleifen)
// ---------------------------------------------------------------------
typedef struct {
    int k_h, k_w, k2;
    int in_c, out_c;
    int in_h, in_w, in_hw;
    int out_h, out_w, oc_plane;
    int w_oc_stride;
    int pool_h, pool_w, pool_stride_h, pool_stride_w, pool_out_h, pool_out_w;
    int pool_type;
    int n_neurons, n_pool;
    bool has_pool;
    bool has_neg_bias;   // mindestens ein Output-Channel mit Bias < 0
} snn_geom_t;

static snn_geom_t s_geom[SNN_MAX_CONV_LAYERS] = {};

// Neuron states and event lists
static float    *s_snn_conv_mem[SNN_MAX_CONV_LAYERS] = {nullptr};
static uint16_t *s_snn_conv_events[SNN_MAX_CONV_LAYERS] = {nullptr};
static uint16_t *s_snn_conv_pool_events[SNN_MAX_CONV_LAYERS] = {nullptr};
static float *s_snn_conv_pool_scales[SNN_MAX_CONV_LAYERS] = {nullptr};
static float    *s_snn_fc_mem[SNN_MAX_FC_LAYERS] = {nullptr};
static uint16_t *s_snn_fc_events[SNN_MAX_FC_LAYERS] = {nullptr};

// Sparse tables: for each (ic, kh, kw) group, process only weights
// not zeroed by pruning. Weights are stored
// contiguously by group and sorted by oc so that the
// oc dimension is addressed contiguously. delta[] is the precomputed
// offset in membrane state (oc * oc_plane for NCHW); oc[] is the output channel
// for NHWC and for bias and scale accesses.
typedef struct {
    uint32_t *off;      // Gruppenanfang, Laenge groups + 1
    uint8_t  *oc;       // output channel for this entry
    float    *wf;       // FP32 weights
    int8_t   *wq;       // INT8 weights (scaled per output channel)
    bool      is_dense; // Direkt aus Flash
} sparse_tbl_t;

static sparse_tbl_t s_tbl[SNN_MAX_CONV_LAYERS];

// Quantized weights: int8 with per-output-channel scaling
static float   *s_wsc_conv[SNN_MAX_CONV_LAYERS] = {nullptr};
static float   *s_wsc_conv_in[SNN_MAX_CONV_LAYERS] = {nullptr};
static int8_t  *s_fc_wq[SNN_MAX_FC_LAYERS] = {nullptr};
static float   *s_wsc_fc[SNN_MAX_FC_LAYERS] = {nullptr};
static float   *s_wsc_fc_in[SNN_MAX_FC_LAYERS] = {nullptr};

// INT32-Akkumulator (nur Conv, nur wenn SNN_CFG_ACC32)
static int32_t *s_conv_acc[SNN_MAX_CONV_LAYERS] = {nullptr};

// Lazy decay: timestamp of the last membrane access and tables for
// closed-form update: mem(dt) = beta^dt * mem + bias * (1 - beta^dt)/(1 - beta)
static uint8_t *s_last_t[SNN_MAX_CONV_LAYERS] = {nullptr};
static uint8_t *s_last_t_fc[SNN_MAX_FC_LAYERS] = {nullptr};
static float   *s_pow_beta = nullptr;   // [dt] = beta^dt
static float   *s_bias_sum = nullptr;   // [dt] = (1 - beta^dt) / (1 - beta)

// NHWC -> NCHW reindexing for FC input (weights remain unchanged)
static uint16_t *s_nhwc2nchw[SNN_MAX_CONV_LAYERS] = {nullptr};
// Precomputed destination index for each input position and kernel tap (0xFFFF = out of bounds).
// This removes 2*k*k divisions and modulo operations per event during scatter.
static uint16_t *s_tap_off[SNN_MAX_CONV_LAYERS] = {nullptr};

// Output-stationary scatter: events are grouped by input channel,
// then the inner loop processes events for one
// channel. This makes mem[] access sequential rather than scattered
// and keeps the weight in a register.
static uint16_t *s_ev_bucket = nullptr;   // nach Kanal sortierte Events
static uint16_t *s_ev_row     = nullptr;   // row index per event in s_tap_off
static uint16_t *s_ev_ofs     = nullptr;   // valid destination indices per (channel, tap)
static uint32_t *s_ev_start  = nullptr;   // Startindex je Kanal
static int      s_max_in_ev  = 0;

// Check only neurons touched in this time step: if a step starts with
// mem < thr (always true after the previous step: either it spiked ->
// mem = 0, or it did not spike -> mem < thr), a neuron without
// synaptic input reaches threshold only if bias > 0. With beta == 1 and
// bias < 0, the membrane decreases without input, so untouched
// neurons cannot spike and threshold checks for those channels can use the
// sparse list (last[] == t). This is determined per output channel
// in each layer because biases may have mixed signs.

// Lazy decay active (can be toggled at runtime in host builds)
static int s_lazy_active = SNN_CFG_LAZY;
// s_last_t is also used by the beta==1 variant to mark
// neurons touched during scatter (for channels with bias < 0).
static bool need_stamp_tbl(void) {
    return (SNN_CFG_LAZY != 0);
}
#if SNN_SEL_RUNTIME
#define LAZY_ON  (s_lazy_active)
#define ALLOC_LAZY 1
#else
#define LAZY_ON  SNN_CFG_LAZY
#define ALLOC_LAZY SNN_CFG_LAZY
#endif
#if SNN_SEL_RUNTIME
#define CONV_QUANT_ON ((s_conv_variant & 2) != 0)
#define FC_QUANT_ON   ((s_conv_variant & 8) != 0)
#define ACC32_ON      (((s_conv_variant & 2) != 0) && ((s_conv_variant & 4) != 0))
#define NHWC_ON       ((s_conv_variant & 1) != 0)
#define ALLOC_QUANT 1
#define ALLOC_ACC32 1
#else
#define CONV_QUANT_ON SNN_CFG_INT8
#define FC_QUANT_ON   SNN_CFG_INT8_FC
#define ACC32_ON      (SNN_CFG_INT8 && SNN_CFG_ACC32)
#define NHWC_ON       SNN_CFG_NHWC
#define ALLOC_QUANT   (SNN_CFG_INT8 || SNN_CFG_INT8_FC)
#define ALLOC_ACC32   (SNN_CFG_INT8 && SNN_CFG_ACC32)
#endif

static int s_snn_spk_sum[128] = {0};
static float s_snn_readout_sum[128] = {0.0f};
static int16_t s_snn_in_tf[MNIST_IMAGE_SIZE];
static uint16_t *s_snn_in_events = nullptr;
static uint16_t *s_snn_flat_events = nullptr;
static float *s_snn_flat_scales = nullptr;
static uint8_t  *s_snn_pool_visited = nullptr;
static int s_conv_variant = SNN_CFG_NHWC | (SNN_CFG_INT8 << 1) | (SNN_CFG_ACC32 << 2);

static void init_geometry(void) {
    const snn_config_t &cfg = g_snn_config;
    for (int l = 0; l < cfg.num_conv_layers; ++l) {
        const snn_conv_layer_cfg_t &cl = cfg.conv_layers[l];
        snn_geom_t &g = s_geom[l];
        g.k_h     = cl.kernel_h;
        g.k_w     = cl.kernel_w;
        g.k2      = cl.kernel_h * cl.kernel_w;
        g.in_c    = cl.in_channels;
        g.out_c   = cl.out_channels;
        g.in_h    = cl.in_h;
        g.in_w    = cl.in_w;
        g.in_hw   = cl.in_h * cl.in_w;
        g.out_h   = cl.out_h;
        g.out_w   = cl.out_w;
        g.oc_plane = cl.out_h * cl.out_w;
        g.w_oc_stride = cl.in_channels * g.k2;
        g.pool_h = cl.pool_h;
        g.pool_w = cl.pool_w;
        g.pool_stride_h = (cl.pool_stride_h > 0) ? cl.pool_stride_h : cl.pool_h;
        g.pool_stride_w = (cl.pool_stride_w > 0) ? cl.pool_stride_w : cl.pool_w;
        g.pool_type = cl.pool_type;
        g.pool_out_h = cl.pool_out_h;
        g.pool_out_w = cl.pool_out_w;
        g.n_neurons  = cl.out_channels * g.oc_plane;
        g.n_pool     = cl.out_channels * cl.pool_out_h * cl.pool_out_w;
        g.has_pool   = cl.pool_h > 1 || cl.pool_w > 1;
        const float *bc = g_snn_conv_biases[l];
        g.has_neg_bias = false;
        if (bc) {
            for (int oc = 0; oc < g.out_c; ++oc) {
                if (bc[oc] < 0.0f) { g.has_neg_bias = true; break; }
            }
        }
    }
}

// Skalierung und Quantisierung einer Zeile (pro Output-Channel).
// Rueckgabe: true, wenn die Zeile mindestens einen von null verschiedenen Wert hat.
static bool quantize_row(const float *w, int n, int8_t *q, float *scale) {
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = (w[i] < 0.0f) ? -w[i] : w[i];
        if (a > mx) mx = a;
    }
    if (mx == 0.0f) {
        *scale = 1.0f;
        memset(q, 0, (size_t)n);
        return false;
    }
    const float inv = 127.0f / mx;
    *scale = mx / 127.0f;
    for (int i = 0; i < n; ++i) {
        int v = (int)lroundf(w[i] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
    return true;
}

// Build compact nonzero weights for each (ic, kh, kw) group.
// need_f/need_q select which weight variant to build.
static bool build_sparse_tables(void) {
    const snn_config_t &cfg = g_snn_config;
    const bool need_f = !CONV_QUANT_ON;
    const bool need_q = CONV_QUANT_ON;

    for (int l = 0; l < cfg.num_conv_layers; ++l) {
        const snn_geom_t &g = s_geom[l];
        const float *w = g_snn_conv_weights[l];
        const int groups = g.in_c * g.k2;
        const int row_len = groups;

        sparse_tbl_t &tb = s_tbl[l];
        tb.off = nullptr;
        tb.oc = nullptr;
        tb.wf = nullptr;
        tb.wq = nullptr;
        tb.is_dense = false;

        // Erstlauf: jede Output-Zeile quantisieren, damit durch Quantisierung
        // Drop weights that are exactly zero from the sparse list as well.
        float *wsc = nullptr;
        int8_t *allq = nullptr;
        if (need_q) {
            wsc = (float*)heap_caps_calloc((size_t)g.out_c, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            allq = (int8_t*)heap_caps_calloc((size_t)g.out_c * row_len, sizeof(int8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!wsc || !allq) {
                if (wsc) free(wsc);
                if (allq) free(allq);
                return false;
            }
            for (int oc = 0; oc < g.out_c; ++oc) {
                (void)quantize_row(w + (size_t)oc * g.w_oc_stride, row_len,
                                   allq + (size_t)oc * row_len, &wsc[oc]);
            }
        }

        uint32_t total = 0;
        for (int gi = 0; gi < groups; ++gi) {
            for (int oc = 0; oc < g.out_c; ++oc) {
                const bool keep = need_q ? (allq[(size_t)oc * row_len + gi] != 0)
                                         : (w[(size_t)oc * g.w_oc_stride + gi] != 0.0f);
                if (keep) ++total;
            }
        }

        // Wenn unquantisiert und vollbesetzt: Direkt aus Flash lesen, spart Allokationen im RAM.
        if (need_f && total == (uint32_t)(groups * g.out_c)) {
            tb.is_dense = true;
        } else {
            uint32_t *off = (uint32_t*)heap_caps_calloc(groups + 1, sizeof(uint32_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!off) {
                if (allq) free(allq);
                if (wsc) free(wsc);
                return false;
            }

            uint32_t cur = 0;
            for (int gi = 0; gi < groups; ++gi) {
                off[gi] = cur;
                for (int oc = 0; oc < g.out_c; ++oc) {
                    const bool keep = need_q ? (allq[(size_t)oc * row_len + gi] != 0)
                                             : (w[(size_t)oc * g.w_oc_stride + gi] != 0.0f);
                    if (keep) ++cur;
                }
            }
            off[groups] = cur;

            const uint32_t alloc_n = (total > 0) ? total : 1;
            uint8_t *occ = (uint8_t*)heap_caps_calloc(alloc_n, sizeof(uint8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            float *nzw = need_f ? (float*)heap_caps_calloc(alloc_n, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL) : nullptr;
            int8_t *nzwq = need_q ? (int8_t*)heap_caps_calloc(alloc_n, sizeof(int8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL) : nullptr;
            if (!occ || (need_f && !nzw) || (need_q && !nzwq)) {
                if (off) free(off);
                if (occ) free(occ);
                if (nzw) free(nzw);
                if (nzwq) free(nzwq);
                if (allq) free(allq);
                if (wsc) free(wsc);
                return false;
            }

            uint32_t n = 0;
            for (int gi = 0; gi < groups; ++gi) {
                for (int oc = 0; oc < g.out_c; ++oc) {
                    const float val = w[(size_t)oc * g.w_oc_stride + gi];
                    const bool keep = need_q ? (allq[(size_t)oc * row_len + gi] != 0) : (val != 0.0f);
                    if (keep) {
                        occ[n] = (uint8_t)oc;
                        if (need_f) nzw[n] = val;
                        if (need_q) nzwq[n] = allq[(size_t)oc * row_len + gi];
                        ++n;
                    }
                }
            }

            tb.off = off;
            tb.oc = occ;
            tb.wf = nzw;
            tb.wq = nzwq;
        }

        if (allq) {
            free(allq);
            allq = nullptr;
        }

        s_wsc_conv[l] = wsc;
        if (wsc) {
            float *wsc_in = (float*)heap_caps_calloc((size_t)g.out_c, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!wsc_in) return false;
            for (int oc = 0; oc < g.out_c; ++oc) wsc_in[oc] = wsc[oc] / 127.0f;
            s_wsc_conv_in[l] = wsc_in;
        }

        // Precompute the destination index for each input position and tap.
        {
            const snn_conv_layer_cfg_t &cl = g_snn_config.conv_layers[l];
            const size_t n_tabs = (size_t)g.in_h * g.in_w * g.k2;
            uint16_t *toff = (uint16_t*)heap_caps_calloc(n_tabs, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!toff) return false;
            for (int ih = 0; ih < g.in_h; ++ih) {
                for (int iw = 0; iw < g.in_w; ++iw) {
                    uint16_t *t = toff + ((size_t)ih * g.in_w + iw) * g.k2;
                    for (int kh = 0; kh < g.k_h; ++kh) {
                        const int oh_u = ih + cl.padding_h - kh;
                        const bool row_ok = (oh_u >= 0) && (oh_u < g.out_h * cl.stride) && ((oh_u % cl.stride) == 0);
                        const int oh = row_ok ? (oh_u / cl.stride) : -1;
                        for (int kw = 0; kw < g.k_w; ++kw) {
                            const int ow_u = iw + cl.padding_w - kw;
                            const bool ok = row_ok && (ow_u >= 0) && (ow_u < g.out_w * cl.stride) && ((ow_u % cl.stride) == 0);
                            if (!ok) { t[kh * g.k_w + kw] = 0xFFFFu; continue; }
                            const int ow = ow_u / cl.stride;
                            const int flat = oh * g.out_w + ow;
#if SNN_SEL_RUNTIME
                            const bool nhwc = (s_conv_variant & 1) != 0;
#else
                            const bool nhwc = SNN_CFG_NHWC;
#endif
                            const uint32_t v = nhwc ? (uint32_t)(flat * g.out_c) : (uint32_t)flat;
                            t[kh * g.k_w + kw] = (v > 0xFFFEu) ? 0xFFFFu : (uint16_t)v;
                        }
                    }
                }
            }
            s_tap_off[l] = toff;
        }

        // NHWC-Umindizierung: (oh, ow, oc) -> (oc, oh, ow)
#if SNN_SEL_RUNTIME
        if (s_conv_variant & 1) {
#else
        if (SNN_CFG_NHWC) {
#endif
            const size_t n_src = (size_t)(g.has_pool ? g.n_pool : g.n_neurons);
            uint16_t *map = (uint16_t*)heap_caps_calloc(n_src, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!map) return false;
            for (int oc = 0; oc < g.out_c; ++oc) {
                const int limit = g.has_pool ? g.pool_out_h * g.pool_out_w : g.oc_plane;
                for (int p = 0; p < limit; ++p) {
                    map[(size_t)p * g.out_c + oc] = (uint16_t)(oc * limit + p);
                }
            }
            s_nhwc2nchw[l] = map;
        }
    }
    return true;
}

// Build INT8 copies of FC weights (scaled per output row) and tables
// for the lazy decay closed-form update.
static bool build_quant_and_decay_tables(void) {
    const snn_config_t &cfg = g_snn_config;

    // Use the same condition as FC kernel selection so the INT8 path
    // niemals auf eine nicht allozierte Tabelle zugreift.
    if (FC_QUANT_ON) {
        const int num_fc = cfg.hidden_layers_count + 1;
        for (int l = 0; l < num_fc; ++l) {
            const int in_dim = cfg.fc_layer_sizes[l];
            const int out_dim = cfg.fc_layer_sizes[l + 1];
            const int n = in_dim * out_dim;
            int8_t *q = (int8_t*)heap_caps_calloc((size_t)n, sizeof(int8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            float *wsc = (float*)heap_caps_calloc((size_t)out_dim, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!q || !wsc) return false;
            for (int i = 0; i < out_dim; ++i) {
                (void)quantize_row(g_snn_fc_weights[l] + (size_t)i * in_dim, in_dim, q + (size_t)i * in_dim, &wsc[i]);
            }
            float *wsc_in = (float*)heap_caps_calloc((size_t)out_dim, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!wsc_in) return false;
            for (int i = 0; i < out_dim; ++i) wsc_in[i] = wsc[i] / 127.0f;
            s_fc_wq[l] = q;
            s_wsc_fc[l] = wsc;
            s_wsc_fc_in[l] = wsc_in;
        }
    }

    if (ALLOC_LAZY) {
        const int max_dt = cfg.num_steps + 1;
        s_pow_beta = (float*)heap_caps_calloc((size_t)max_dt, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        s_bias_sum = (float*)heap_caps_calloc((size_t)max_dt, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_pow_beta || !s_bias_sum) return false;
        const float beta = cfg.beta;
        for (int dt = 0; dt < max_dt; ++dt) {
            s_pow_beta[dt] = powf(beta, (float)dt);
            s_bias_sum[dt] = (beta == 1.0f) ? (float)dt : ((1.0f - s_pow_beta[dt]) / (1.0f - beta));
        }
    }
    return true;
}

static bool init_snn_buffers(void) {
    const snn_config_t &cfg = g_snn_config;
    init_geometry();

    ESP_LOGI(TAG, "SNN init: input=%d, conv_layers=%d, flat=%d, fc=%d->%d->%d, steps=%d",
             cfg.input_size, cfg.num_conv_layers, cfg.flat_size,
             cfg.fc_layer_sizes[0], cfg.fc_layer_sizes[1], cfg.fc_layer_sizes[2],
             cfg.num_steps);
    ESP_LOGI(TAG, "SNN init: free heap before allocation: %u Bytes (min: %u)",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));

#if SNN_SEL_RUNTIME
    s_conv_variant = s_opt_nhwc | (s_opt_convq << 1) | (s_opt_acc32 << 2) | (s_opt_fcq << 3);
    s_lazy_active = s_opt_lazy;
#else
    s_conv_variant = SNN_CFG_NHWC | (SNN_CFG_INT8 << 1) | (SNN_CFG_ACC32 << 2) | (SNN_CFG_INT8 << 3);
    s_lazy_active = SNN_CFG_LAZY;
#endif

    const int in_size = (cfg.input_size > 0) ? cfg.input_size : MNIST_IMAGE_SIZE;
    if (in_size > 65535) {
        ESP_LOGE(TAG, "input_size %d exceeds the 16-bit event format!", in_size);
        return false;
    }
    s_snn_in_events = (uint16_t*)heap_caps_calloc(in_size, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    if (!s_snn_in_events) { ESP_LOGE(TAG, "FAIL: s_snn_in_events (%d B)", in_size * 2); return false; }

    int max_neurons_even = 0, max_neurons_odd = 0;
    int max_pool = 0;
    for (int l = 0; l < cfg.num_conv_layers; ++l) {
        const snn_geom_t &g = s_geom[l];
        ESP_LOGI(TAG, "Conv%d: %dx%dx%d->%dx%dx%d, neurons=%d, pool=%d",
                 l, g.in_c, g.in_h, g.in_w, g.out_c, g.out_h, g.out_w, g.n_neurons, g.n_pool);
        if (g.n_neurons > 65535 || g.n_pool > 65535) {
            ESP_LOGE(TAG, "Conv layer %d exceeds the 16-bit event format!", l);
            return false;
        }
        s_snn_conv_mem[l] = (float*)heap_caps_calloc(g.n_neurons, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_snn_conv_mem[l]) { ESP_LOGE(TAG, "FAIL: conv_mem[%d] (%d B)", l, g.n_neurons * 4); return false; }

        if (need_stamp_tbl()) {
            s_last_t[l] = (uint8_t*)heap_caps_calloc(g.n_neurons, sizeof(uint8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!s_last_t[l]) { ESP_LOGE(TAG, "FAIL: last_t[%d] (%d B)", l, g.n_neurons); return false; }
        }
        if (ALLOC_ACC32) {
            s_conv_acc[l] = (int32_t*)heap_caps_calloc(g.n_neurons, sizeof(int32_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!s_conv_acc[l]) { ESP_LOGE(TAG, "FAIL: conv_acc[%d] (%d B)", l, g.n_neurons * 4); return false; }
        }

        if (l % 2 == 0) {
            if (g.n_neurons > max_neurons_even) max_neurons_even = g.n_neurons;
        } else {
            if (g.n_neurons > max_neurons_odd) max_neurons_odd = g.n_neurons;
        }

        if (g.has_pool && g.n_pool > max_pool) {
            max_pool = g.n_pool;
        }
        if (g.has_pool && g.pool_type == 1) {
            s_snn_conv_pool_scales[l] = (float*)heap_caps_calloc(g.n_pool, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!s_snn_conv_pool_scales[l]) { ESP_LOGE(TAG, "FAIL: pool_scales[%d] (%d B)", l, g.n_pool * 4); return false; }
        }
        const int in_size = g.in_c * g.in_h * g.in_w;
        if (in_size > s_max_in_ev) s_max_in_ev = in_size;
    }

    if (max_neurons_even > 0) {
        uint16_t *buf_even = (uint16_t*)heap_caps_calloc(max_neurons_even, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!buf_even) { ESP_LOGE(TAG, "FAIL: conv_events even (%d B)", max_neurons_even * 2); return false; }
        for (int l = 0; l < cfg.num_conv_layers; l += 2) {
            s_snn_conv_events[l] = buf_even;
        }
    }
    if (max_neurons_odd > 0) {
        uint16_t *buf_odd = (uint16_t*)heap_caps_calloc(max_neurons_odd, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!buf_odd) { ESP_LOGE(TAG, "FAIL: conv_events odd (%d B)", max_neurons_odd * 2); return false; }
        for (int l = 1; l < cfg.num_conv_layers; l += 2) {
            s_snn_conv_events[l] = buf_odd;
        }
    }

    uint16_t *pool_buf_0 = nullptr;
    uint16_t *pool_buf_1 = nullptr;
    if (max_pool > 0) {
        pool_buf_0 = (uint16_t*)heap_caps_calloc(max_pool, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        pool_buf_1 = (uint16_t*)heap_caps_calloc(max_pool, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!pool_buf_0 || !pool_buf_1) { ESP_LOGE(TAG, "FAIL: pool_buf (%d B)", max_pool * 2); return false; }
        for (int l = 0; l < cfg.num_conv_layers; ++l) {
            if (s_geom[l].has_pool) {
                s_snn_conv_pool_events[l] = (l % 2 == 0) ? pool_buf_0 : pool_buf_1;
            }
        }
    }

    const bool need_outst = (SNN_CFG_NHWC == 0) && (g_snn_config.beta == 1.0f);
#if SNN_SEL_RUNTIME
    const bool alloc_outst = true;
#else
    /* OUTST needs BETA1=true, which requires LAZY_ON && !NHWC && beta==1.
     * Without lazy decay, BETA1 is always false, so OUTST buffers are wasted. */
    const bool alloc_outst = need_outst && LAZY_ON;
#endif
    if (alloc_outst && s_max_in_ev > 0) {
        s_ev_bucket = (uint16_t*)heap_caps_calloc((size_t)s_max_in_ev, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        s_ev_row    = (uint16_t*)heap_caps_calloc((size_t)s_max_in_ev, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        s_ev_ofs    = (uint16_t*)heap_caps_calloc((size_t)s_max_in_ev, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        s_ev_start  = (uint32_t*)heap_caps_calloc(68, sizeof(uint32_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_ev_bucket || !s_ev_row || !s_ev_ofs || !s_ev_start) { ESP_LOGE(TAG, "FAIL: ev_bucket/row/ofs (%d B)", s_max_in_ev * 2); return false; }
    }
    if (max_pool > 0) {
        const size_t visited_size = (max_pool > 2048) ? (size_t)max_pool : 2048;
        s_snn_pool_visited = (uint8_t*)heap_caps_calloc(visited_size, sizeof(uint8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_snn_pool_visited) { ESP_LOGE(TAG, "FAIL: pool_visited (%d B)", (int)visited_size); return false; }
    }

    if (cfg.num_conv_layers > 0 && cfg.flat_size > 0 && cfg.flat_size <= 65535) {
        if (max_pool > 0 && cfg.flat_size <= max_pool) {
            const int last_pool_buf_idx = (cfg.num_conv_layers - 1) % 2;
            s_snn_flat_events = (last_pool_buf_idx == 0) ? pool_buf_1 : pool_buf_0;
        } else {
            s_snn_flat_events = (uint16_t*)heap_caps_calloc(cfg.flat_size, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!s_snn_flat_events) { ESP_LOGE(TAG, "FAIL: flat_events (%d B)", cfg.flat_size * 2); return false; }
        }
        const snn_geom_t &last = s_geom[cfg.num_conv_layers - 1];
        if (last.has_pool && last.pool_type == 1) {
            s_snn_flat_scales = (float*)heap_caps_calloc(cfg.flat_size, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!s_snn_flat_scales) { ESP_LOGE(TAG, "FAIL: flat_scales (%d B)", cfg.flat_size * 4); return false; }
        }
    }

    const int num_fc = cfg.hidden_layers_count + 1;
    for (int l = 0; l < num_fc; ++l) {
        const int out_dim = cfg.fc_layer_sizes[l + 1];
        if (out_dim > 65535) {
            ESP_LOGE(TAG, "FC layer %d exceeds the 16-bit event format!", l);
            return false;
        }
        s_snn_fc_mem[l] = (float*)heap_caps_calloc(out_dim, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        s_snn_fc_events[l] = (uint16_t*)heap_caps_calloc(out_dim, sizeof(uint16_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_snn_fc_mem[l] || !s_snn_fc_events[l]) return false;
        if (ALLOC_LAZY) {
            s_last_t_fc[l] = (uint8_t*)heap_caps_calloc(out_dim, sizeof(uint8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!s_last_t_fc[l]) return false;
        }
    }

    if (!build_sparse_tables()) return false;
    if (!build_quant_and_decay_tables()) return false;


#if SNN_PROFILE
    snn_prof_reset();
#endif
    ESP_LOGI(TAG, "Free heap after SNN init: %u Bytes",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
    ESP_LOGI(TAG, "Kernel: Layout=%s, Conv weights=%s, FC weights=%s, Conv accumulator=%s, decay=%s (%s)",
             (s_conv_variant & 1) ? "NHWC" : "NCHW",
             (s_conv_variant & 2) ? "INT8" : "FP32",
             FC_QUANT_ON ? "INT8" : "FP32",
             ((s_conv_variant & 2) && (s_conv_variant & 4)) ? "INT32" : "FP32",
             LAZY_ON ? "lazy" : "explizit",
             (cfg.beta == 1.0f) ? "beta=1, Decay ohne Wirkung" : "beta<1");
    return true;
}

// ---------------------------------------------------------------------
// Eingangs-Encoding
// ---------------------------------------------------------------------
static void encode_input_times(const float *img) {
    const snn_config_t &cfg = g_snn_config;
    if (cfg.encoding_type == ENCODING_TTFS) {
        const float last = (float)(cfg.num_steps - 1);
        for (int j = 0; j < cfg.input_size; ++j) {
            const float val = img[j];
            if (val > 0.0f) {
                const float clamped = (val > 1.0f) ? 1.0f : val;
                s_snn_in_tf[j] = (int16_t)lroundf(last * (1.0f - clamped));
            } else {
                s_snn_in_tf[j] = -1;
            }
        }
    } else if (cfg.encoding_type == ENCODING_PERIODIC) {
        for (int j = 0; j < cfg.input_size; ++j) {
            const float val = img[j];
            const float clamped = (val < 0.0f) ? 0.0f : ((val > 1.0f) ? 1.0f : val);
            s_snn_in_tf[j] = (int16_t)lroundf(clamped * (float)cfg.num_steps);
        }
    }
}

static int build_input_events(const float *img, int t) {
    const snn_config_t &cfg = g_snn_config;
    uint16_t *out = s_snn_in_events;
    int n = 0;

    switch (cfg.encoding_type) {
        case ENCODING_TTFS:
            for (int j = 0; j < cfg.input_size; ++j) {
                if (s_snn_in_tf[j] == t) out[n++] = (uint16_t)j;
            }
            break;
        case ENCODING_PERIODIC:
            for (int j = 0; j < cfg.input_size; ++j) {
                if (t < s_snn_in_tf[j]) out[n++] = (uint16_t)j;
            }
            break;
        case ENCODING_POISSON: {
            for (int j = 0; j < cfg.input_size; ++j) {
                const float p = img[j];
                if (p > 0.0f) {
                    const float r = (float)(esp_random() & 0xFFFF) / 65535.0f;
                    if (r < p) out[n++] = (uint16_t)j;
                }
            }
            break;
        }
        default: // ENCODING_DIRECT
            for (int j = 0; j < cfg.input_size; ++j) {
                // Direct temporal inputs may be signed analog currents.
                // Include both polarities in the sparse event list; fc_body
                // applies the original signed value when accumulating.
                if (cfg.temporal_input ? (img[j] != 0.0f) : (img[j] > 0.0f)) out[n++] = (uint16_t)j;
            }
            break;
    }
    return n;
}

// ---------------------------------------------------------------------
// Sparse Conv-Layer
//
//   1. Synaptic accumulation only for active input events and only
//      over weights that were not zeroed by pruning
//   2. Branchless threshold check and event generation
//
// Decay and bias are applied either explicitly in a separate pass
// (LAZY = false) or lazily in closed form on the first
// access to a neuron at time step t:
//     mem(dt) = beta^dt * mem + bias * (1 - beta^dt) / (1 - beta)
//
// Template-Parameter:
//   NHWC  false = Index oc*oc_plane + p (Pixel-minor, Ebene-major)
//          true  = Index p*out_c + oc (Pixel-major, oc minor) -> die
//                  Scatter-Schreibzugriffe liegen dann contiguierlich
//   QUANT false = FP32 weights during scatter
//          true  = INT8 weights, INT32 accumulation in a register (true) or
//                  in a separate INT32 accumulator (ACC32, 4 bytes/neuron)
//   ACC32 Use an INT32 accumulator per neuron instead of dequantizing
//          each weight
//   LAZY  Apply decay/bias lazily in closed form
// ---------------------------------------------------------------------
template <bool NHWC, bool QUANT, bool ACC32, bool LAZY, bool BETA1>
static int conv_body(int l, const uint16_t *ev, int n_ev,
                     const float *scale_src, int t, uint32_t *synops) {
    const snn_config_t &cfg = g_snn_config;
    const snn_geom_t &g = s_geom[l];
    const sparse_tbl_t &tb = s_tbl[l];
    float *mem = s_snn_conv_mem[l];
    const float *b = g_snn_conv_biases[l];
    const float beta = cfg.beta;
    const float thr = cfg.threshold;
    const int plane = g.oc_plane;
    const int out_c = g.out_c;
    const bool unit_scale = (scale_src == nullptr);

    // Per-output-channel scaling; for scaled input (DIRECT),
    // input is quantized by 1/127, hence the second table.
    const float *wsc = QUANT ? (unit_scale ? s_wsc_conv[l] : s_wsc_conv_in[l]) : nullptr;
    int32_t *acc = ACC32 ? s_conv_acc[l] : nullptr;

    const bool track = LAZY && !BETA1;
    const bool TRACK = track;
    uint8_t *last = (LAZY || BETA1) ? s_last_t[l] : nullptr;
    const float *pb = track ? s_pow_beta : nullptr;
    const float *bs = track ? s_bias_sum : nullptr;
    // With beta == 1, bias can be separated from the state:
    //   v = mem[i] + b*dt  und  mu[i] := mem[i] - b*last[i]  ergibt  v = mu[i] + b*stamp
    // mu remains unchanged between time steps and is updated only on a spike
    // to -b*stamp. This removes decay updates from the scatter.
    const bool MARK = BETA1 && g.has_neg_bias;
    const uint8_t stamp = (uint8_t)(t + 1);

    if (!LAZY) {
        for (int oc = 0; oc < out_c; ++oc) {
            const float bv = b ? b[oc] : 0.0f;
            if (NHWC) {
                for (int p = 0; p < plane; ++p) {
                    float &m = mem[p * out_c + oc];
                    m = beta * m + bv;
                }
            } else {
                float *m = mem + oc * plane;
                for (int i = 0; i < plane; ++i) {
                    m[i] = beta * m[i] + bv;
                }
            }
        }
        if (ACC32) {
            memset(acc, 0, (size_t)g.n_neurons * sizeof(int32_t));
        }
    }

    #if SNN_PROFILE
    PROF_T(t_scatter);
#endif
    // Output-stationary scatter (NCHW only, beta == 1, unscaled input):
    // Group events by input channel, then collect valid destination indices
    // for each (channel, tap) and iterate over output channels. The weight
    // stays in a register, removing bounds checks from the inner loop.
    const bool OUTST = BETA1 && !NHWC && (scale_src == nullptr) && (g.in_c <= 63);
    if (OUTST && n_ev > 0) {
        const uint32_t *off = tb.off;
        const uint8_t *occ = tb.oc;
        const float *nw = tb.wf;
        const int8_t *nq = tb.wq;
        const uint16_t *t0 = s_tap_off[l];
        const int in_w = g.in_w;
        const int k2 = g.k2;
        const int in_hw = g.in_hw;
        const int in_c = g.in_c;
        uint32_t local_synops = 0;

        // Events je Kanal zaehlen und buendeln (Reihenfolge bleibt aufsteigend)
        for (int c = 0; c <= in_c; ++c) s_ev_start[c] = 0;
        for (int e = 0; e < n_ev; ++e) s_ev_start[ev[e] / in_hw + 1]++;
        for (int c = 0; c < in_c; ++c) s_ev_start[c + 1] += s_ev_start[c];
        {
            uint32_t fill[64];
            for (int c = 0; c <= in_c && c < 64; ++c) fill[c] = s_ev_start[c];
            for (int e = 0; e < n_ev; ++e) {
                const uint16_t flat = ev[e];
                const uint32_t c = (uint32_t)(flat / in_hw);
                const uint32_t pos = fill[c < 64 ? c : 63]++;
                s_ev_bucket[pos] = flat;
                const uint32_t r0 = flat - c * in_hw;
                const uint32_t ih = r0 / in_w;
                s_ev_row[pos] = (uint16_t)((ih * in_w + (r0 - ih * in_w)) * k2);
            }
        }

        for (int ic = 0; ic < in_c; ++ic) {
            const uint32_t b0 = s_ev_start[ic];
            const uint32_t b1 = s_ev_start[ic + 1];
            if (b1 == b0) continue;
            const int nevic = (int)(b1 - b0);
            const uint16_t *rowp = s_ev_row + b0;

            for (int tap = 0; tap < k2; ++tap) {
                const uint32_t gi = (uint32_t)(ic * k2 + tap);

                // Collect valid destination indices for this channel's events
                int nv = 0;
                for (int j = 0; j < nevic; ++j) {
                    const uint16_t tof = t0[(size_t)rowp[j] + tap];
                    if (tof != 0xFFFFu) s_ev_ofs[nv++] = tof;
                }
                if (nv == 0) continue;

                if (tb.is_dense) {
                    for (int oc = 0; oc < out_c; ++oc) {
                        const uint32_t obase = (uint32_t)oc * (uint32_t)plane;
                        const float w_val = g_snn_conv_weights[l][(size_t)oc * g.w_oc_stride + gi];
                        if (w_val == 0.0f) continue;
                        for (int j = 0; j < nv; ++j) {
                            const uint32_t idx = obase + s_ev_ofs[j];
                            if (MARK) last[idx] = stamp;
                            mem[idx] += w_val;
                        }
                    }
                    local_synops += (uint32_t)out_c * (uint32_t)nv;
                } else {
                    const uint32_t s = off[gi];
                    const uint32_t e2 = off[gi + 1];
                    if (s == e2) continue;

                    for (uint32_t n = s; n < e2; ++n) {
                        const uint32_t obase = (uint32_t)occ[n] * (uint32_t)plane;
                        if (QUANT) {
                            const float wsc_oc = wsc[occ[n]];
                            const int32_t q = nq[n];
                            for (int j = 0; j < nv; ++j) {
                                const uint32_t idx = obase + s_ev_ofs[j];
                                if (MARK) last[idx] = stamp;
                                if (ACC32) acc[idx] += q;
                                else mem[idx] += (float)q * wsc_oc;
                            }
                        } else {
                            const float w = nw[n];
                            for (int j = 0; j < nv; ++j) {
                                const uint32_t idx = obase + s_ev_ofs[j];
                                if (MARK) last[idx] = stamp;
                                mem[idx] += w;
                            }
                        }
                    }
                    local_synops += (e2 - s) * (uint32_t)nv;
                }
            }
        }
        if (synops) *synops += local_synops;
#if SNN_PROFILE
        s_prof[4] += (uint64_t)n_ev;
#endif
    } else if (n_ev > 0) {
        const uint32_t *off = tb.off;
        const uint8_t *occ = tb.oc;
        const float *nw = tb.wf;
        const int8_t *nq = tb.wq;
        const int k2 = g.k2;
        const int in_w = g.in_w;
        const int in_hw = g.in_hw;
        (void)in_hw;
        uint32_t local_synops = 0;

        for (int e = 0; e < n_ev; ++e) {
#if SNN_PROFILE
            s_prof[4]++;   // Events
#endif
            const uint16_t flat = ev[e];
            int iscale = 1;
            if (!unit_scale) {
                const float fscale = scale_src[flat];
                if (fscale == 0.0f) continue;
                iscale = (int)lroundf(fscale * 127.0f);
                if (iscale == 0) continue;
            }

            int ic, ih, iw;
            if (NHWC) {
                ic = flat % g.in_c;
                const int r0 = flat / g.in_c;
                ih = r0 / in_w;
                iw = r0 - ih * in_w;
            } else {
                ic = flat / in_hw;
                const int r0 = flat - ic * in_hw;
                ih = r0 / in_w;
                iw = r0 - ih * in_w;
            }
            const int gbase = ic * k2;
            const uint16_t *t9 = s_tap_off[l] + ((size_t)ih * in_w + iw) * k2;

            for (int tap = 0; tap < k2; ++tap) {
                const uint32_t base = t9[tap];
                if (base == 0xFFFFu) continue;

                const uint32_t gi = gbase + tap;
                if (tb.is_dense) {
                    for (int oc = 0; oc < out_c; ++oc) {
                        const float w_val = g_snn_conv_weights[l][(size_t)oc * g.w_oc_stride + gi];
                        if (w_val == 0.0f) continue;
                        const uint32_t idx = base + (NHWC ? (uint32_t)oc : (uint32_t)oc * (uint32_t)plane);
                        if (MARK) last[idx] = stamp;
                        if (BETA1) {
                            mem[idx] += unit_scale ? w_val : w_val * scale_src[flat];
                            continue;
                        }
                        if (TRACK) {
                            const int dt = stamp - (int)last[idx];
                            if (dt) {
                                mem[idx] = mem[idx] * pb[dt] + b[oc] * bs[dt];
                                last[idx] = stamp;
                            }
                        }
                        mem[idx] += unit_scale ? w_val : w_val * scale_src[flat];
                    }
                    local_synops += (uint32_t)out_c;
                } else {
                    const uint32_t s = off[gi];
                    const uint32_t e2 = off[gi + 1];
                    for (uint32_t n = s; n < e2; ++n) {
                        const uint32_t idx = base + (NHWC ? (uint32_t)occ[n] : (uint32_t)occ[n] * (uint32_t)plane);
                        if (MARK) last[idx] = stamp;
                        if (BETA1) {
                            // Der Bias steckt bereits in mem (als mu), es gibt
                            // nichts nachzufuehren.
                            if (QUANT) {
                                const int32_t prod = (int32_t)nq[n] * (int32_t)iscale;
                                if (ACC32) {
                                    acc[idx] += prod;
                                } else {
                                    mem[idx] += (float)prod * wsc[occ[n]];
                                }
                            } else {
                                mem[idx] += unit_scale ? nw[n] : nw[n] * scale_src[flat];
                            }
                            continue;
                        }
                        const int oc = (QUANT || LAZY) ? occ[n] : 0;
                        if (TRACK) {
                            // dt == 0 is the normal case: the same neuron is
                            // in the timestep for multiple kernel positions
                            // reached, so decay must be applied only once.
                            const int dt = stamp - (int)last[idx];
                            if (dt) {
                                mem[idx] = mem[idx] * pb[dt] + b[oc] * bs[dt];
                                last[idx] = stamp;
                            }
                        }
                        if (QUANT) {
                            const int32_t prod = (int32_t)nq[n] * (int32_t)iscale;
                            if (ACC32) {
                                acc[idx] += prod;
                            } else {
                                mem[idx] += (float)prod * wsc[oc];
                            }
                        } else {
                            mem[idx] += unit_scale ? nw[n] : nw[n] * scale_src[flat];
                        }
                    }
                    local_synops += e2 - s;
                }
            }
        }
        if (synops) *synops += local_synops;
    }
#if SNN_PROFILE
    PROF_ADD(t_scatter, 0);
    PROF_T(t_eval);
#endif

    uint16_t *out_ev = s_snn_conv_events[l];
    int n_out = 0;

    // Evaluate one neuron. LAZY applies decay and bias in closed form
    // nach, ACC32 loest den INT32-Akkumulator auf.
    auto eval = [&](int i, int oc, float bv) {
        float v;
        if (LAZY) {
            const int dt = stamp - (int)last[i];
            v = mem[i] * pb[dt] + bv * bs[dt];
            last[i] = stamp;
        } else {
            v = mem[i];
        }
        if (ACC32) {
            v += (float)acc[i] * wsc[oc];
            acc[i] = 0;
        }
        const int sp = (v >= thr);
        out_ev[n_out] = (uint16_t)i;
        n_out += sp;
        mem[i] = sp ? 0.0f : v;
    };

    // Like eval, but for a neuron already touched by scatter: decay and bias
    // have already been applied, so only the threshold check remains.
    auto eval_touched = [&](int i, int oc) {
        float v = mem[i];
        if (ACC32) {
            v += (float)acc[i] * wsc[oc];
            acc[i] = 0;
        }
        const int sp = (v >= thr);
        out_ev[n_out] = (uint16_t)i;
        n_out += sp;
        mem[i] = sp ? 0.0f : v;
    };

    if (BETA1) {
        // v = mu[i] + b*stamp; mu bleibt ohne Spike unveraendert.
        for (int oc = 0; oc < out_c; ++oc) {
            const float bv = b ? b[oc] : 0.0f;
            const float bvs = bv * (float)stamp;
            const int base = oc * plane;
            if (bv < 0.0f) {
                // Ohne Spike kann die Membran den Wert mu unveraendert nur
                // reduce the scan: untouched neurons cannot spike.
                for (int p = 0; p < plane; ++p) {
                    const int i = base + p;
                    if (last[i] != stamp) continue;
                    last[i] = 0;
                    const float m0 = mem[i];
                    float nu = m0;
                    if (ACC32) {
                        nu = m0 + (float)acc[i] * wsc[oc];
                        acc[i] = 0;
                        mem[i] = nu;   // Fold the INT32 contribution into mu.
                    }
                    if (nu + bvs >= thr) {
                        mem[i] = -bvs;
                        out_ev[n_out++] = (uint16_t)i;
                    }
                }
            } else {
                for (int p = 0; p < plane; ++p) {
                    const int i = base + p;
                    const float m0 = mem[i];
                    float nu = m0;
                    if (ACC32) {
                        nu = m0 + (float)acc[i] * wsc[oc];
                        acc[i] = 0;
                        mem[i] = nu;
                    }
                    if (nu + bvs >= thr) {
                        mem[i] = -bvs;
                        out_ev[n_out++] = (uint16_t)i;
                    }
                }
            }
        }
#if SNN_PROFILE
        PROF_ADD(t_eval, 1);
#endif
        return n_out;
    }

    if (NHWC) {
        // Pixel-major: ascending index, sequential membrane accesses
        for (int p = 0; p < plane; ++p) {
            for (int oc = 0; oc < out_c; ++oc) {
                const int i = p * out_c + oc;
                const float bv = b ? b[oc] : 0.0f;
                if (LAZY && bv < 0.0f) {
                    if (last[i] == stamp) eval_touched(i, oc);
                } else {
                    eval(i, oc, bv);
                }
            }
        }
    } else {
        // Channel-major: contiguous range for each channel
        for (int oc = 0; oc < out_c; ++oc) {
            const float bv = b ? b[oc] : 0.0f;
            const int base = oc * plane;
            if (LAZY && bv < 0.0f) {
                for (int p = 0; p < plane; ++p) {
                    const int i = base + p;
                    if (last[i] == stamp) eval_touched(i, oc);
                }
            } else {
                for (int p = 0; p < plane; ++p) {
                    eval(base + p, oc, bv);
                }
            }
        }
    }
#if SNN_PROFILE
    PROF_ADD(t_eval, 1);
#endif
    return n_out;
}

template <bool NHWC, bool QUANT, bool ACC32>
static int conv_variant(int l, const uint16_t *ev, int n_ev,
                        const float *scale_src, int t, uint32_t *synops) {
    // beta == 1 is handled separately by BETA1: bias is already
    // included in the state, so scatter needs no decay update.
    // This requires lazy decay (otherwise the preparation pass computes
    // Bias schon in die Membran ein) und das NCHW-Layout (die Variante in
    // conv_body indiziert kanalweise).
    const bool b1 = LAZY_ON && !NHWC && (g_snn_config.beta == 1.0f);
    if (LAZY_ON) {
        if (b1) return conv_body<NHWC, QUANT, ACC32, true,  true >(l, ev, n_ev, scale_src, t, synops);
        return conv_body<NHWC, QUANT, ACC32, true,  false>(l, ev, n_ev, scale_src, t, synops);
    }
    if (b1) return conv_body<NHWC, QUANT, ACC32, false, true >(l, ev, n_ev, scale_src, t, synops);
    return conv_body<NHWC, QUANT, ACC32, false, false>(l, ev, n_ev, scale_src, t, synops);
}

#if SNN_SEL_RUNTIME
static int conv_dispatch(int l, const uint16_t *ev, int n_ev,
                         const float *scale_src, int t, uint32_t *synops) {
    switch (s_conv_variant & 7) {
        case 0: return conv_variant<false, false, false>(l, ev, n_ev, scale_src, t, synops);
        case 1: return conv_variant<true,  false, false>(l, ev, n_ev, scale_src, t, synops);
        case 2: return conv_variant<false, true,  false>(l, ev, n_ev, scale_src, t, synops);
        case 3: return conv_variant<true,  true,  false>(l, ev, n_ev, scale_src, t, synops);
        case 6: return conv_variant<false, true,  true >(l, ev, n_ev, scale_src, t, synops);
        case 7: return conv_variant<true,  true,  true >(l, ev, n_ev, scale_src, t, synops);
        default: return conv_variant<false, false, false>(l, ev, n_ev, scale_src, t, synops);
    }
}
#define CONV_FWD conv_dispatch
#else
#define CONV_FWD conv_variant<SNN_CFG_NHWC, SNN_CFG_INT8, (SNN_CFG_INT8 && SNN_CFG_ACC32)>
#endif

// Event-basiertes Max-Pooling: jede gepoolte Zelle meldet hoechstens ein Event
template <bool NHWC>
static int pool_body(int l, const uint16_t *ev, int n_ev) {
    const snn_geom_t &g = s_geom[l];
    uint16_t *out = s_snn_conv_pool_events[l];
    memset(s_snn_pool_visited, 0, (size_t)g.n_pool);

    const int out_w = g.out_w;
    const int oc_plane = g.oc_plane;
    const int out_c = g.out_c;
    const int pool_stride_h = g.pool_stride_h;
    const int pool_stride_w = g.pool_stride_w;
    const int pool_out_h = g.pool_out_h;
    const int pool_out_w = g.pool_out_w;
    int n = 0;

    for (int e = 0; e < n_ev; ++e) {
        const int idx = ev[e];
        int oc, oh, ow;
        if (NHWC) {
            oc = idx % out_c;
            const int rem = idx / out_c;
            oh = rem / out_w;
            ow = rem - oh * out_w;
        } else {
            oc = idx / oc_plane;
            const int rem = idx - oc * oc_plane;
            oh = rem / out_w;
            ow = rem - oh * out_w;
        }
        const int ph  = oh / pool_stride_h;
        const int pw  = ow / pool_stride_w;
        if (ph >= pool_out_h || pw >= pool_out_w) continue;
        const int p_idx = NHWC ? ((ph * pool_out_w + pw) * out_c + oc)
                               : ((oc * pool_out_h + ph) * pool_out_w + pw);
        if (s_snn_pool_visited[p_idx] == 0) out[n++] = (uint16_t)p_idx;
        if (s_snn_pool_visited[p_idx] < 255) s_snn_pool_visited[p_idx]++;
    }
    if (g.pool_type == 1) {
        float *scales = s_snn_conv_pool_scales[l];
        const float denom = (float)(g.pool_h * g.pool_w);
        for (int i = 0; i < n; ++i) {
            const int idx = out[i];
            scales[idx] = (float)s_snn_pool_visited[idx] / denom;
        }
    }
    return n;
}

template <bool NHWC>
static int pool_variant(int l, const uint16_t *ev, int n_ev) {
    return pool_body<NHWC>(l, ev, n_ev);
}

#if SNN_SEL_RUNTIME
static int pool_dispatch(int l, const uint16_t *ev, int n_ev) {
    return ((s_conv_variant & 1) != 0) ? pool_variant<true>(l, ev, n_ev)
                                       : pool_variant<false>(l, ev, n_ev);
}
#define POOL_FWD pool_dispatch
#else
#define POOL_FWD pool_variant<SNN_CFG_NHWC>
#endif

// ---------------------------------------------------------------------
// Sparse FC layer: accumulate over active input events with
// row-wise weight access (one mem access per output).
// Bei QUANT erfolgt die Akkumulation als INT8 x INT8 -> INT32 im Register.
// ---------------------------------------------------------------------
template <bool QUANT, bool LAZY>
static int fc_body(int l, const uint16_t *ev, int n_ev,
                   const float *scale_src, int t, uint32_t *synops) {
    const snn_config_t &cfg = g_snn_config;
    const int in_dim = cfg.fc_layer_sizes[l];
    const int out_dim = cfg.fc_layer_sizes[l + 1];
    const float *w = g_snn_fc_weights[l];
    const int8_t *wq = QUANT ? s_fc_wq[l] : nullptr;
    const float *wsc = QUANT ? (scale_src ? s_wsc_fc_in[l] : s_wsc_fc[l]) : nullptr;
    const float *b = g_snn_fc_biases[l];
    float *mem = s_snn_fc_mem[l];
    const float beta = cfg.beta;
    const float thr = cfg.threshold;
    const float input_scale = (cfg.input_scale > 0.0f) ? cfg.input_scale : 1.0f;
    const bool unit_scale = (scale_src == nullptr);
    uint8_t *last = LAZY ? s_last_t_fc[l] : nullptr;
    const float *pb = LAZY ? s_pow_beta : nullptr;
    const float *bs = LAZY ? s_bias_sum : nullptr;
    const uint8_t stamp = (uint8_t)(t + 1);
    uint16_t *out_ev = s_snn_fc_events[l];
    int n_out = 0;

    // Do not skip untouched neurons: the FC layer has no
    // scatter operation to set last[] for the current step. last[i]
    // bezoge sich dann auf den vorherigen Schritt.
    for (int i = 0; i < out_dim; ++i) {
        const float bv = b ? b[i] : 0.0f;
        float sum = 0.0f;
        if (QUANT) {
            const int8_t *row = wq + (size_t)i * in_dim;
            int32_t isum = 0;
            if (unit_scale) {
                for (int e = 0; e < n_ev; ++e) {
                    isum += (int32_t)row[ev[e]];
                }
            } else {
                for (int e = 0; e < n_ev; ++e) {
                    const int j = ev[e];
                    isum += (int32_t)row[j] * (int32_t)lroundf(scale_src[j] * input_scale * 127.0f);
                }
            }
            sum = (float)isum * wsc[i];
        } else {
            const float *row = w + (size_t)i * in_dim;
            if (unit_scale) {
                for (int e = 0; e < n_ev; ++e) {
                    sum += row[ev[e]];
                }
            } else {
                for (int e = 0; e < n_ev; ++e) {
                    const int j = ev[e];
                    sum += row[j] * scale_src[j] * input_scale;
                }
            }
        }

        if (cfg.analog_readout && l == cfg.hidden_layers_count) {
            s_snn_readout_sum[i] += bv + sum;
            continue;
        }

        float v;
        if (LAZY) {
            const int dt = stamp - (int)last[i];
            v = beta * pb[dt] * mem[i] + bv * bs[dt] + sum;
            last[i] = stamp;
        } else {
            v = beta * mem[i] + bv + sum;
        }
        const int sp = (v >= thr);
        out_ev[n_out] = (uint16_t)i;
        n_out += sp;
        mem[i] = sp ? 0.0f : v;
    }

    if (synops) *synops += (uint32_t)n_ev * (uint32_t)out_dim;
    return n_out;
}

template <bool QUANT>
static int fc_variant(int l, const uint16_t *ev, int n_ev,
                      const float *scale_src, int t, uint32_t *synops) {
    if (LAZY_ON) return fc_body<QUANT, true>(l, ev, n_ev, scale_src, t, synops);
    return fc_body<QUANT, false>(l, ev, n_ev, scale_src, t, synops);
}

#if SNN_SEL_RUNTIME
static int fc_dispatch(int l, const uint16_t *ev, int n_ev,
                       const float *scale_src, int t, uint32_t *synops) {
    return FC_QUANT_ON ? fc_variant<true>(l, ev, n_ev, scale_src, t, synops)
                       : fc_variant<false>(l, ev, n_ev, scale_src, t, synops);
}
#define FC_FWD fc_dispatch
#else
#define FC_FWD fc_variant<SNN_CFG_INT8_FC>
#endif

// Ein Simulationsschritt: Conv-Pipeline (optional) und FC-Pipeline
static void snn_step(const float *img, int t, uint32_t *total_spk,
                     uint32_t *total_synops) {
    const snn_config_t &cfg = g_snn_config;
    const int L = cfg.num_conv_layers;
    const int num_fc_layers = cfg.hidden_layers_count + 1;
    const bool direct = (cfg.encoding_type == ENCODING_DIRECT);

    int n_ev = build_input_events(img, t);
    const uint16_t *ev = s_snn_in_events;

    const float *scale_src = nullptr;
    for (int l = 0; l < L; ++l) {
        if (l == 0 && direct) scale_src = img;
        const int n_conv = CONV_FWD(l, ev, n_ev, scale_src, t, total_synops);
        if (total_spk) *total_spk += (uint32_t)n_conv;
        if (s_geom[l].has_pool) {
            PROF_T(t_pool);
            n_ev = POOL_FWD(l, s_snn_conv_events[l], n_conv);
            PROF_ADD(t_pool, 2);
            ev = s_snn_conv_pool_events[l];
            scale_src = (s_geom[l].pool_type == 1) ? s_snn_conv_pool_scales[l] : nullptr;
        } else {
            n_ev = n_conv;
            ev = s_snn_conv_events[l];
            scale_src = nullptr;
        }
    }

    if (L > 0) {
        const snn_geom_t &last = s_geom[L - 1];
        const int src_size = last.has_pool ? last.n_pool : last.n_neurons;
        const bool needs_map = NHWC_ON && s_snn_flat_events;
        if (cfg.flat_size != src_size || needs_map) {
            const uint16_t *map = needs_map ? s_nhwc2nchw[L - 1] : nullptr;
            int n_flat = 0;
            for (int e = 0; e < n_ev; ++e) {
                const uint16_t idx = ev[e];
                if (idx >= cfg.flat_size && map == nullptr) continue;
                const uint16_t dst_idx = map ? map[idx] : idx;
                s_snn_flat_events[n_flat++] = dst_idx;
                if (s_snn_flat_scales) s_snn_flat_scales[dst_idx] = last.pool_type == 1 ? s_snn_conv_pool_scales[L - 1][idx] : 1.0f;
            }
            ev = s_snn_flat_events;
            n_ev = n_flat;
        } else if (s_snn_flat_scales) {
            for (int e = 0; e < n_ev; ++e) s_snn_flat_scales[ev[e]] = s_snn_conv_pool_scales[L - 1][ev[e]];
        }
    }

    for (int l = 0; l < num_fc_layers; ++l) {
        const float *scale = (L == 0 && l == 0 && direct) ? img : (l == 0 && s_snn_flat_scales ? s_snn_flat_scales : nullptr);
        PROF_T(t_fc);
        const int n_out = FC_FWD(l, ev, n_ev, scale, t, total_synops);
        PROF_ADD(t_fc, 3);
        if (total_spk) *total_spk += (uint32_t)n_out;
        if (l == num_fc_layers - 1 && !cfg.analog_readout) {
            const uint16_t *out_ev = s_snn_fc_events[l];
            for (int e = 0; e < n_out; ++e) {
                s_snn_spk_sum[out_ev[e]]++;
            }
        }
        ev = s_snn_fc_events[l];
        n_ev = n_out;
    }
}

static void reset_snn_state(void) {
    const snn_config_t &cfg = g_snn_config;
    for (int l = 0; l < cfg.num_conv_layers; ++l) {
        memset(s_snn_conv_mem[l], 0, (size_t)s_geom[l].n_neurons * sizeof(float));
        if (need_stamp_tbl()) {
            // last_t = 0 corresponds to "virtual timestep -1", so that the
            // Neuron im ersten Schritt t = 0 den Bias genau einmal erhaelt.
            memset(s_last_t[l], 0, (size_t)s_geom[l].n_neurons * sizeof(uint8_t));
        }
        if (ALLOC_ACC32) {
            memset(s_conv_acc[l], 0, (size_t)s_geom[l].n_neurons * sizeof(int32_t));
        }
    }
    for (int l = 0; l <= cfg.hidden_layers_count; ++l) {
        memset(s_snn_fc_mem[l], 0, (size_t)cfg.fc_layer_sizes[l + 1] * sizeof(float));
        if (ALLOC_LAZY) {
            memset(s_last_t_fc[l], 0, (size_t)cfg.fc_layer_sizes[l + 1] * sizeof(uint8_t));
        }
    }
    memset(s_snn_spk_sum, 0, sizeof(s_snn_spk_sum));
    memset(s_snn_readout_sum, 0, sizeof(s_snn_readout_sum));
}

static int snn_argmax(void) {
    const snn_config_t &cfg = g_snn_config;
    int best_class = 0;
    float max_spk = cfg.analog_readout ? s_snn_readout_sum[0] : (float)s_snn_spk_sum[0];
    for (int i = 1; i < cfg.output_size && i < 128; ++i) {
        const float score = cfg.analog_readout ? s_snn_readout_sum[i] : (float)s_snn_spk_sum[i];
        if (score > max_spk) {
            max_spk = score;
            best_class = i;
        }
    }
    return best_class;
}

#if SNN_SEL_DENSE_REF || SNN_SEL_RUNTIME
// ---------------------------------------------------------------------
// Dense reference path: visits every weight in a layer and uses
// keine Sparse-Tabellen und keine Ereignislisten. Dient nur dem Abgleich
// the same arithmetic as the optimized paths (NHWC/NCHW indexing, pruning skips,
// Quantisierung, Lazy Decay).
// ---------------------------------------------------------------------
static float *s_ref_mem[SNN_MAX_CONV_LAYERS] = {nullptr};
static float *s_ref_mem_fc[SNN_MAX_FC_LAYERS] = {nullptr};
static uint8_t *s_ref_spike[SNN_MAX_CONV_LAYERS] = {nullptr};
static uint8_t *s_ref_pool[SNN_MAX_CONV_LAYERS] = {nullptr};
static float *s_ref_flat = nullptr;   // Dense vector for the next layer.
static float *s_ref_in = nullptr;

static bool init_dense_reference(void) {
    const snn_config_t &cfg = g_snn_config;
    int flat_max = cfg.flat_size;
    for (int l = 0; l < cfg.num_conv_layers; ++l) {
        const snn_geom_t &g = s_geom[l];
        s_ref_mem[l] = (float*)heap_caps_calloc(g.n_neurons, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        s_ref_spike[l] = (uint8_t*)heap_caps_calloc(g.n_neurons, sizeof(uint8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_ref_mem[l] || !s_ref_spike[l]) return false;
        if (g.has_pool) {
            s_ref_pool[l] = (uint8_t*)heap_caps_calloc(g.n_pool, sizeof(uint8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!s_ref_pool[l]) return false;
            if (g.n_pool > flat_max) flat_max = g.n_pool;
        } else if (g.n_neurons > flat_max) {
            flat_max = g.n_neurons;
        }
    }
    const int num_fc = cfg.hidden_layers_count + 1;
    for (int l = 0; l < num_fc; ++l) {
        const int out_dim = cfg.fc_layer_sizes[l + 1];
        if (out_dim > flat_max) flat_max = out_dim;
        s_ref_mem_fc[l] = (float*)heap_caps_calloc(out_dim, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_ref_mem_fc[l]) return false;
    }
    s_ref_flat = (float*)heap_caps_calloc((size_t)flat_max, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    s_ref_in = (float*)heap_caps_calloc((size_t)cfg.input_size, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    return (s_ref_flat != nullptr) && (s_ref_in != nullptr);
}

// Dichtes Conv ohne Sparse-Tabelle: NCHW, alle Gewichtsslotpositionen
static void conv_dense_ref(int l, const float *src) {
    const snn_config_t &cfg = g_snn_config;
    const snn_geom_t &g = s_geom[l];
    const snn_conv_layer_cfg_t &cl = cfg.conv_layers[l];
    const float *w = g_snn_conv_weights[l];
    const float *b = g_snn_conv_biases[l];
    float *mem = s_ref_mem[l];
    uint8_t *spk = s_ref_spike[l];

    for (int oc = 0; oc < g.out_c; ++oc) {
        const float bv = b ? b[oc] : 0.0f;
        const float *wrow = w + (size_t)oc * g.w_oc_stride;
        for (int oh = 0; oh < g.out_h; ++oh) {
            for (int ow = 0; ow < g.out_w; ++ow) {
                float acc = 0.0f;
                for (int ic = 0; ic < g.in_c; ++ic) {
                    for (int kh = 0; kh < g.k_h; ++kh) {
                        const int ih = oh * cl.stride + kh - cl.padding_h;
                        if (ih < 0 || ih >= g.in_h) continue;
                        for (int kw = 0; kw < g.k_w; ++kw) {
                            const int iw = ow * cl.stride + kw - cl.padding_w;
                            if (iw < 0 || iw >= g.in_w) continue;
                            const float x = src[(ic * g.in_h + ih) * g.in_w + iw];
                            if (x != 0.0f) acc += wrow[(ic * g.k_h + kh) * g.k_w + kw] * x;
                        }
                    }
                }
                const int idx = oc * g.oc_plane + oh * g.out_w + ow;
                const float v = cfg.beta * mem[idx] + bv + acc;
                const int sp = (v >= cfg.threshold);
                spk[idx] = (uint8_t)sp;
                mem[idx] = sp ? 0.0f : v;
            }
        }
    }
}

static void pool_dense_ref(int l, float *dst) {
    const snn_geom_t &g = s_geom[l];
    const uint8_t *spk = s_ref_spike[l];
    for (int oc = 0; oc < g.out_c; ++oc) {
        for (int ph = 0; ph < g.pool_out_h; ++ph) {
            for (int pw = 0; pw < g.pool_out_w; ++pw) {
                int count = 0;
                for (int dh = 0; dh < g.pool_h; ++dh) {
                    const int oh = ph * g.pool_stride_h + dh;
                    for (int dw = 0; dw < g.pool_w; ++dw) {
                        const int ow = pw * g.pool_stride_w + dw;
                        if (oh < g.out_h && ow < g.out_w) count += spk[oc * g.oc_plane + oh * g.out_w + ow];
                    }
                }
                const float pooled = g.pool_type == 1
                    ? (float)count / (float)(g.pool_h * g.pool_w)
                    : (count > 0 ? 1.0f : 0.0f);
                dst[oc * g.pool_out_h * g.pool_out_w + ph * g.pool_out_w + pw] = pooled;
            }
        }
    }
}

static int run_snn_inference_dense_ref(const float *img, uint32_t *out_spikes) {
    const snn_config_t &cfg = g_snn_config;
    const int L = cfg.num_conv_layers;
    const int num_fc_layers = cfg.hidden_layers_count + 1;
    const bool direct = (cfg.encoding_type == ENCODING_DIRECT);

    for (int l = 0; l < L; ++l) {
        memset(s_ref_mem[l], 0, (size_t)s_geom[l].n_neurons * sizeof(float));
    }
    for (int l = 0; l < num_fc_layers; ++l) {
        memset(s_ref_mem_fc[l], 0, (size_t)cfg.fc_layer_sizes[l + 1] * sizeof(float));
    }
    int spk_sum[128];
    float readout_sum[128];
    memset(spk_sum, 0, sizeof(spk_sum));
    memset(readout_sum, 0, sizeof(readout_sum));
    if (!cfg.temporal_input) encode_input_times(img);

    uint32_t spikes = 0;
    for (int t = 0; t < cfg.num_steps; ++t) {
        const float *step_img = cfg.temporal_input ? img + (size_t)t * cfg.input_size : img;
        for (int j = 0; j < cfg.input_size; ++j) {
            switch (cfg.encoding_type) {
                case ENCODING_TTFS:     s_ref_in[j] = (s_snn_in_tf[j] == t) ? 1.0f : 0.0f; break;
                case ENCODING_PERIODIC: s_ref_in[j] = (t < s_snn_in_tf[j]) ? 1.0f : 0.0f; break;
                case ENCODING_POISSON: {
                    const float p = step_img[j];
                    if (p > 0.0f) {
                        const float r = (float)(esp_random() & 0xFFFF) / 65535.0f;
                        s_ref_in[j] = (r < p) ? 1.0f : 0.0f;
                    } else {
                        s_ref_in[j] = 0.0f;
                    }
                    break;
                }
                default:
                    s_ref_in[j] = cfg.temporal_input ? step_img[j] : ((step_img[j] > 0.0f) ? step_img[j] : 0.0f);
                    break;
            }
        }

        const float *cur = s_ref_in;
        for (int l = 0; l < L; ++l) {
            conv_dense_ref(l, cur);
            const snn_geom_t &g = s_geom[l];
            int n = 0;
            for (int i = 0; i < g.n_neurons; ++i) n += s_ref_spike[l][i];
            spikes += (uint32_t)n;
            if (g.has_pool) {
                pool_dense_ref(l, s_ref_flat);
                cur = s_ref_flat;
            } else {
                for (int i = 0; i < g.n_neurons; ++i) s_ref_flat[i] = (float)s_ref_spike[l][i];
                cur = s_ref_flat;
            }
        }

        const float *in_fc = cur;
        for (int l = 0; l < num_fc_layers; ++l) {
            const int in_dim = cfg.fc_layer_sizes[l];
            const int out_dim = cfg.fc_layer_sizes[l + 1];
            const float *w = g_snn_fc_weights[l];
            const float *b = g_snn_fc_biases[l];
            float *next = s_ref_flat;
            for (int i = 0; i < out_dim; ++i) {
                const float *row = w + (size_t)i * in_dim;
                float acc = 0.0f;
                if (l == 0 && direct) {
                    for (int j = 0; j < in_dim; ++j) {
                        const float input_scale = (cfg.input_scale > 0.0f) ? cfg.input_scale : 1.0f;
                        const float x = step_img[j] * input_scale;
                        if (x != 0.0f) acc += row[j] * x;
                    }
                } else {
                    for (int j = 0; j < in_dim; ++j) acc += row[j] * in_fc[j];
                }
                if (cfg.analog_readout && l == num_fc_layers - 1) {
                    readout_sum[i] += (b ? b[i] : 0.0f) + acc;
                    next[i] = 0.0f;
                    continue;
                }
                const float v = cfg.beta * s_ref_mem_fc[l][i] + (b ? b[i] : 0.0f) + acc;
                const int sp = (v >= cfg.threshold);
                next[i] = sp ? 1.0f : 0.0f;
                s_ref_mem_fc[l][i] = sp ? 0.0f : v;
                spikes += (uint32_t)sp;
                if (l == num_fc_layers - 1 && sp) spk_sum[i]++;
            }
            in_fc = next;
        }
    }

    if (out_spikes) *out_spikes = spikes;
    int best = 0;
    for (int i = 1; i < cfg.output_size && i < 128; ++i) {
        const float candidate = cfg.analog_readout ? readout_sum[i] : (float)spk_sum[i];
        const float current = cfg.analog_readout ? readout_sum[best] : (float)spk_sum[best];
        if (candidate > current) best = i;
    }
    for (int l = 0; l < L; ++l) {
        memset(s_ref_mem[l], 0, (size_t)s_geom[l].n_neurons * sizeof(float));
    }
    for (int l = 0; l < num_fc_layers; ++l) {
        memset(s_ref_mem_fc[l], 0, (size_t)cfg.fc_layer_sizes[l + 1] * sizeof(float));
    }
    return best;
}
#endif

static int run_snn_inference_dense(const float *img, uint32_t *out_spikes, uint32_t *out_synops) {
    const snn_config_t &cfg = g_snn_config;
    uint32_t total_spk = 0;
    uint32_t total_synops = 0;

    reset_snn_state();
    if (!cfg.temporal_input) encode_input_times(img);

    for (int t = 0; t < cfg.num_steps; ++t) {
        const float *step_img = cfg.temporal_input ? img + (size_t)t * cfg.input_size : img;
        snn_step(step_img, t, &total_spk, &total_synops);
    }

    if (out_spikes) *out_spikes = total_spk;
    if (out_synops) *out_synops = total_synops;
    const int prediction = snn_argmax();
    reset_snn_state();
    return prediction;
}

static int run_snn_inference_event_based(const float *img, uint32_t *out_spikes, uint32_t *out_synops) {
    const snn_config_t &cfg = g_snn_config;
    uint32_t total_spk = 0;
    uint32_t total_synops = 0;

    reset_snn_state();
    if (!cfg.temporal_input) encode_input_times(img);

    for (int t = 0; t < cfg.num_steps; ++t) {
        const float *step_img = cfg.temporal_input ? img + (size_t)t * cfg.input_size : img;
        snn_step(step_img, t, &total_spk, &total_synops);
    }

    if (out_spikes) *out_spikes = total_spk;
    if (out_synops) *out_synops = total_synops;
    const int prediction = snn_argmax();
    reset_snn_state();
    return prediction;
}

static int run_snn_inference(const float *img, uint32_t *out_spikes, uint32_t *out_synops = nullptr) {
    if (g_snn_config.execution_mode == EXECUTION_MODE_EVENT_BASED) {
        return run_snn_inference_event_based(img, out_spikes, out_synops);
    } else {
        return run_snn_inference_dense(img, out_spikes, out_synops);
    }
}

#if SNN_SEL_RUNTIME
extern "C" int snn_host_init(void) { return init_snn_buffers() ? 1 : 0; }
extern "C" int snn_host_run(const float *img, uint32_t *spk, uint32_t *syn) {
    return run_snn_inference(img, spk, syn);
}
extern "C" void snn_select_variants(int nhwc, int convq, int acc32, int lazy, int fcq) {
    s_opt_nhwc = nhwc;
    s_opt_convq = convq;
    s_opt_acc32 = acc32;
    s_opt_lazy = lazy;
    s_opt_fcq = fcq;
}
extern "C" int snn_dense_ref_init(void) { return init_dense_reference() ? 1 : 0; }
extern "C" int snn_dense_ref_predict(const float *img, uint32_t *spk) {
    return run_snn_inference_dense_ref(img, spk);
}
#endif

#ifndef HOST_TEST
extern "C" void app_main(void)
{
#ifndef DATASET_NAME
#define DATASET_NAME "mnist"
#endif
    const char *model_type_name = g_model_type == MODEL_TYPE_SNN ? "SNN" :
                                  (g_model_type == MODEL_TYPE_TRANSFORMER ?
                                   (g_transformer_config.spiking ? "Spiking-Transformer" : "Transformer-ANN") : "ANN");
    ESP_LOGI(TAG, "=== Initializing ESP32 inference: %s (%s, dataset: %s) ===",
             g_model_name, model_type_name, DATASET_NAME);

    // 1. Trigger-GPIO konfigurieren
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << TRIGGER_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    gpio_set_level(TRIGGER_GPIO, 0);

    // 2. TensorFlow Lite Micro initialisieren (nur bei ANN)
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;
    int input_elements = 0;

    if (g_model_type == MODEL_TYPE_ANN) {
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!tensor_arena) {
            ESP_LOGE(TAG, "Failed to allocate tensor arena (%d bytes)!", (int)kTensorArenaSize);
            return;
        }

        const tflite::Model *model = tflite::GetModel(g_model);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            ESP_LOGE(TAG, "Model schema version %d does not match TFLM version %d!",
                     model->version(), TFLITE_SCHEMA_VERSION);
            return;
        }
        ESP_LOGI(TAG, "TensorFlow Lite Micro model loaded (size: %d bytes, type: ANN).", g_model_len);

        static tflite::MicroMutableOpResolver<8> resolver;
        resolver.AddFullyConnected();
        resolver.AddRelu();
        resolver.AddConv2D();
        resolver.AddMaxPool2D();
        resolver.AddAveragePool2D();
        resolver.AddReshape();
        resolver.AddSoftmax();

        static tflite::MicroInterpreter interp(model, resolver, tensor_arena, kTensorArenaSize);
        if (interp.AllocateTensors() != kTfLiteOk) {
            ESP_LOGE(TAG, "Tensor allocation failed! Increase the arena size?");
            return;
        }
        interpreter = &interp;
        input = interpreter->input(0);
        output = interpreter->output(0);
        input_elements = 1;
        for (int d = 0; d < input->dims->size; ++d) {
            input_elements *= input->dims->data[d];
        }
        if (input_elements > MNIST_IMAGE_SIZE) input_elements = MNIST_IMAGE_SIZE;
        ESP_LOGI(TAG, "TFLM engine initialized. Input dimensions: [%d, %d, %d, %d]",
                 input->dims->data[0],
                 (input->dims->size > 1) ? input->dims->data[1] : 1,
                 (input->dims->size > 2) ? input->dims->data[2] : 1,
                 (input->dims->size > 3) ? input->dims->data[3] : 1);
    } else if (g_model_type == MODEL_TYPE_SNN) {
        if (!init_snn_buffers()) {
            ESP_LOGE(TAG, "Failed to allocate SNN buffers!");
            return;
        }

        const char *ntype_str = (g_snn_config.neuron_type == NEURON_TYPE_IF) ? "IF" : "LIF";
        const char *enc_str = "DIRECT";
        if (g_snn_config.encoding_type == ENCODING_TTFS) enc_str = "TTFS";
        else if (g_snn_config.encoding_type == ENCODING_POISSON) enc_str = "POISSON";
        else if (g_snn_config.encoding_type == ENCODING_PERIODIC) enc_str = "PERIODIC";

        const char *exec_str = (g_snn_config.execution_mode == EXECUTION_MODE_EVENT_BASED) ? "EVENT-BASED" : "DENSE";

        ESP_LOGI(TAG, "SNN parameters: Neuron=%s, Encoding=%s, Mode=%s, Conv=%d, FC=%d, %d Steps, Beta=%.2f, Thresh=%.2f",
                 ntype_str, enc_str, exec_str, g_snn_config.num_conv_layers, g_snn_config.hidden_layers_count + 1,
                 g_snn_config.num_steps, g_snn_config.beta, g_snn_config.threshold);
    } else if (g_model_type == MODEL_TYPE_TRANSFORMER) {
        if (g_transformer_config.tokens != KWS_TOKENS ||
            g_transformer_config.input_features != KWS_FEATURES ||
            g_transformer_config.embed_dim != KWS_EMBED ||
            g_transformer_config.num_heads != KWS_HEADS ||
            g_transformer_config.num_layers != KWS_LAYERS ||
            g_transformer_config.feedforward_dim != KWS_FFN ||
            g_transformer_config.output_size != KWS_CLASSES ||
            DATASET_INPUT_SIZE != KWS_TOKENS * KWS_FEATURES ||
            DATASET_NUM_CLASSES != KWS_CLASSES) {
            ESP_LOGE(TAG, "Transformer configuration does not match the ESP32 kernel.");
            return;
        }
        if (g_transformer_config.spiking &&
            (g_transformer_config.num_steps < 1 || g_transformer_config.num_steps > 32 ||
             g_transformer_config.beta < 0.0f || g_transformer_config.beta >= 1.0f ||
             g_transformer_config.threshold <= 0.0f)) {
            ESP_LOGE(TAG, "Spiking Transformer neuron parameters are invalid.");
            return;
        }
        input_elements = DATASET_INPUT_SIZE;
        ESP_LOGI(TAG, "Compact speech Transformer initialized: %d tokens, D=%d, heads=%d, layers=%d, classes=%d",
                 g_transformer_config.tokens, g_transformer_config.embed_dim,
                 g_transformer_config.num_heads, g_transformer_config.num_layers,
                 g_transformer_config.output_size);
    }

#if HAS_SPEECH_AUDIO
    if (!g_speech_mel_filterbank) {
        ESP_LOGE(TAG, "Speech Commands Mel filter bank is missing from the model export.");
        return;
    }
    init_speech_frontend();
    ESP_LOGI(TAG, "Speech Commands frontend initialized (16 kHz, FFT=%d, Mel=%d, Features=%dx%d).",
             SPEECH_NFFT, SPEECH_MELS, KWS_FEATURES, KWS_TOKENS);
#endif

    #if SNN_SEL_DENSE_REF > 0
        if (init_dense_reference()) {
            const int n_check = (SNN_SEL_DENSE_REF < MNIST_TEST_SUBSET) ? SNN_SEL_DENSE_REF : MNIST_TEST_SUBSET;
            int mism_pred = 0, mism_spk = 0, max_delta = 0;
            for (int i = 0; i < n_check; ++i) {
                uint32_t s1 = 0, s2 = 0;
                const int p_fast = run_snn_inference(mnist_test_images[i], &s1, nullptr);
                const int p_ref = run_snn_inference_dense_ref(mnist_test_images[i], &s2);
                if (p_fast != p_ref) ++mism_pred;
                // Die Spike-Zahlen duerfen um wenige Stellen abweichen: der
                // Referenzpfad summiert in anderer Reihenfolge, bei Grenzwerten
                // am Schwellwert kippt dadurch einzelne Spikes.
                if (s1 != s2) {
                    ++mism_spk;
                    const int d = (int)s1 - (int)s2;
                    if (d < 0 ? -d : d > max_delta) max_delta = d < 0 ? -d : d;
                }
            }
            ESP_LOGI(TAG, "Dense reference path self-test (%d samples): %d predictions differ, "
                         "%d spike counts differ (max %d spikes)",
                     n_check, mism_pred, mism_spk, max_delta);
        } else {
            ESP_LOGE(TAG, "Failed to allocate dense reference path!");
        }
#endif

    ESP_LOGI(TAG, "Number of test samples: %d", MNIST_TEST_SUBSET);
    ESP_LOGI(TAG, "Trigger GPIO %d (connected to Joulescope IN1) is ready.\n", TRIGGER_GPIO);

    while (1) {
        ESP_LOGI(TAG, "Waiting 2 seconds before starting the inference measurement cycle...");
        vTaskDelay(pdMS_TO_TICKS(2000));

        // GPIO start pattern: 3x HIGH (200 ms), separated by LOW (100 ms) -> measurement (HIGH)
        // Three pulses clearly distinguish it from the 2-pulse end pattern and noise.
        gpio_set_level(TRIGGER_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(200));
        gpio_set_level(TRIGGER_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(100));
        gpio_set_level(TRIGGER_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(200));
        gpio_set_level(TRIGGER_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(100));
        gpio_set_level(TRIGGER_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(200));
        gpio_set_level(TRIGGER_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(100));
        ESP_LOGI(TAG, "Start pattern (3 pulses) sent; waiting 1 second before inference...");
        vTaskDelay(pdMS_TO_TICKS(1000));

        // Start measurement by setting the trigger pin HIGH
        ESP_LOGI(TAG, ">>> START: Set GPIO %d HIGH (inference running) <<<", TRIGGER_GPIO);
        gpio_set_level(TRIGGER_GPIO, 1);

        int64_t start_time = esp_timer_get_time();
        const uint32_t cycle_start = esp_cpu_get_cycle_count();
        int correct_count = 0;
        uint64_t total_all_spikes = 0;
        uint64_t total_all_synops = 0;
        #if SNN_PROFILE
        snn_prof_reset();
        #endif

        for (int i = 0; i < MNIST_TEST_SUBSET; ++i) {
#if HAS_SPEECH_AUDIO
            // PCM remains in flash; the Mel spectrogram is computed for each sample
            // on the ESP32 during the trigger window and included in the measurement.
            speech_audio_to_mel(speech_test_audio[i], s_speech_features);
            const float *model_input = s_speech_features;
#else
            const float *img_data = mnist_test_images[i];
            const float *model_input = img_data;
#endif
            int predicted = -1;

            if (g_model_type == MODEL_TYPE_ANN && interpreter) {
                memcpy(input->data.f, model_input, (size_t)input_elements * sizeof(float));
                if (interpreter->Invoke() == kTfLiteOk) {
                    predicted = argmax_float(output->data.f, MNIST_NUM_OUTPUTS);
                }
            } else if (g_model_type == MODEL_TYPE_TRANSFORMER) {
                predicted = run_transformer_inference(model_input);
                if (g_transformer_config.spiking) {
                    total_all_spikes += s_transformer_spikes;
                    total_all_synops += s_transformer_synops;
                }
            } else {
                uint32_t sample_spikes = 0;
                uint32_t sample_synops = 0;
                predicted = run_snn_inference(model_input, &sample_spikes, &sample_synops);
                total_all_spikes += sample_spikes;
                total_all_synops += sample_synops;
            }

            if (predicted == mnist_test_labels[i]) {
                correct_count++;
            }
            // Yield CPU to prevent watchdog timeout on long inferences
            vTaskDelay(0);
        }

        int64_t end_time = esp_timer_get_time();

        // End pattern: LOW (100 ms) -> HIGH (200 ms) -> LOW (100 ms) -> HIGH (200 ms) -> LOW (500 ms)
        // Two pulses clearly distinguish it from the 3-pulse start pattern.
        gpio_set_level(TRIGGER_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(100));
        gpio_set_level(TRIGGER_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(200));
        gpio_set_level(TRIGGER_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(100));
        gpio_set_level(TRIGGER_GPIO, 1);
        vTaskDelay(pdMS_TO_TICKS(200));
        gpio_set_level(TRIGGER_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(500));
        ESP_LOGI(TAG, ">>> END: End pattern (2 pulses) sent (inference complete) <<<", TRIGGER_GPIO);

        int64_t elapsed_us = end_time - start_time;
#if SNN_PROFILE
        const uint32_t cycle_end = esp_cpu_get_cycle_count();
        s_cycle_delta = (uint32_t)(cycle_end - cycle_start);
#else
        (void)cycle_start;
#endif
        float elapsed_ms = (float)elapsed_us / 1000.0f;
        float time_per_inf_ms = elapsed_ms / (float)MNIST_TEST_SUBSET;
        float accuracy_pct = (100.0f * (float)correct_count) / (float)MNIST_TEST_SUBSET;

        ESP_LOGI(TAG, "Inference measurement complete:");
        ESP_LOGI(TAG, "  - Model:              %s (%s)", g_model_name, model_type_name);
        ESP_LOGI(TAG, "  - Samples evaluated: %d", MNIST_TEST_SUBSET);
        ESP_LOGI(TAG, "  - Correct samples:   %d / %d", correct_count, MNIST_TEST_SUBSET);
        ESP_LOGI(TAG, "  - Test accuracy:     %.2f %%", accuracy_pct);
        ESP_LOGI(TAG, "  - Total duration:    %.3f ms", elapsed_ms);
        ESP_LOGI(TAG, "  - Time per inference:%.3f ms (%.1f Infs/s)",
                 time_per_inf_ms, (time_per_inf_ms > 0) ? (1000.0f / time_per_inf_ms) : 0.0f);

        if (g_model_type == MODEL_TYPE_SNN ||
            (g_model_type == MODEL_TYPE_TRANSFORMER && g_transformer_config.spiking)) {
            float avg_spk_per_inf = (float)total_all_spikes / (float)MNIST_TEST_SUBSET;
            float avg_synops_per_inf = (float)total_all_synops / (float)MNIST_TEST_SUBSET;
            ESP_LOGI(TAG, "  - Total spikes:      %" PRIu64 " (avg %.1f spikes/inference)",
                     total_all_spikes, avg_spk_per_inf);
            ESP_LOGI(TAG, "  - Total SynOps:      %" PRIu64 " (avg %.1f SynOps/inference)",
                     total_all_synops, avg_synops_per_inf);
    #if SNN_PROFILE
        {
            uint64_t c = 0, p = 0, f = 0, cs = 0, ce = 0;
            snn_prof_get(&c, &p, &f);
            snn_prof_get_split(&cs, &ce);
            ESP_LOGI(TAG, "  - Conv total:        %.3f ms (%.1f %%)", (double)c / 1000.0, 100.0 * (double)c / (double)elapsed_us);
            ESP_LOGI(TAG, "    Scatter:           %.3f ms (%.1f %%)", (double)cs / 1000.0, 100.0 * (double)cs / (double)elapsed_us);
            ESP_LOGI(TAG, "    Thresholding:      %.3f ms (%.1f %%)", (double)ce / 1000.0, 100.0 * (double)ce / (double)elapsed_us);
            ESP_LOGI(TAG, "  - Pool time:         %.3f ms (%.1f %%)", (double)p / 1000.0, 100.0 * (double)p / (double)elapsed_us);
            ESP_LOGI(TAG, "  - FC time:           %.3f ms (%.1f %%)", (double)f / 1000.0, 100.0 * (double)f / (double)elapsed_us);
            uint64_t nev = 0, nmac = 0;
            snn_prof_get_counts(&nev, &nmac);
            ESP_LOGI(TAG, "  - Events/MACs per inf: %.1f / %.0f", (double)nev / MNIST_TEST_SUBSET, (double)nmac / MNIST_TEST_SUBSET);
            if (elapsed_us > 0) {
                ESP_LOGI(TAG, "  - Clock (mcycle):    %.1f MHz",
                         (double)s_cycle_delta * 1e6 / (double)elapsed_us / 1e6);
            }
        }
#endif
        printf("ESP32_METRICS: ACC=%.2f;CORRECT=%d;TOTAL=%d;TIME_US=%" PRId64 ";SPIKES=%" PRIu64 ";SYNOPS=%" PRIu64 "\n",
                   accuracy_pct, correct_count, MNIST_TEST_SUBSET, elapsed_us, total_all_spikes, total_all_synops);
        } else {
            printf("ESP32_METRICS: ACC=%.2f;CORRECT=%d;TOTAL=%d;TIME_US=%" PRId64 "\n",
                   accuracy_pct, correct_count, MNIST_TEST_SUBSET, elapsed_us);
        }

        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
#endif
