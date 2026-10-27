#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MODEL_TYPE_ANN 0
#define MODEL_TYPE_SNN 1
#define MODEL_TYPE_TRANSFORMER 2

#define NEURON_TYPE_LIF 0
#define NEURON_TYPE_IF  1

#define ENCODING_DIRECT    0
#define ENCODING_TTFS      1
#define ENCODING_POISSON   2
#define ENCODING_PERIODIC  3

#define EXECUTION_MODE_DENSE       0
#define EXECUTION_MODE_EVENT_BASED 1

#define SNN_MAX_FC_LAYERS   8
#define SNN_MAX_CONV_LAYERS 4

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_h, kernel_w;
    int stride;
    int padding_h, padding_w;
    int pool_h, pool_w;
    int pool_stride_h, pool_stride_w;
    int pool_type;  // 0=max, 1=average
    int in_h, in_w;
    int out_h, out_w;
    int pool_out_h, pool_out_w;
} snn_conv_layer_cfg_t;

typedef struct {
    int input_size;
    int num_conv_layers;
    snn_conv_layer_cfg_t conv_layers[SNN_MAX_CONV_LAYERS];
    int flat_size;
    int hidden_layers_count;
    int fc_layer_sizes[SNN_MAX_FC_LAYERS + 1];
    int num_steps;
    float beta;
    float threshold;
    int output_size;
    int neuron_type;
    int encoding_type;
    int execution_mode;
    int temporal_input;
    float input_scale;
    int analog_readout;
} snn_config_t;

typedef struct {
    const float *qkv_w, *qkv_b, *attn_out_w, *attn_out_b;
    const float *norm1_w, *norm1_b, *ff1_w, *ff1_b;
    const float *ff2_w, *ff2_b, *norm2_w, *norm2_b;
} transformer_layer_weights_t;

typedef struct {
    int tokens, input_features, embed_dim, num_heads, num_layers, feedforward_dim, output_size;
    int spiking, num_steps;
    float beta, threshold, input_scale;
} transformer_config_t;

extern const int g_model_type;
extern const char* const g_model_name;

// ANN-Spezifisch (TensorFlow Lite Micro FlatBuffer)
extern const unsigned char g_model[];
extern const int g_model_len;

// SNN-Spezifisch (LIF/IF Neuronen Parameter & Gewichte)
extern const snn_config_t g_snn_config;
extern const float* const g_snn_conv_weights[];
extern const float* const g_snn_conv_biases[];
extern const float* const g_snn_fc_weights[];
extern const float* const g_snn_fc_biases[];
extern const transformer_config_t g_transformer_config;
extern const float* const g_transformer_input_w;
extern const float* const g_transformer_input_b;
extern const float* const g_transformer_position;
extern const transformer_layer_weights_t g_transformer_layers[];
extern const float* const g_transformer_norm_w;
extern const float* const g_transformer_norm_b;
extern const float* const g_transformer_classifier_w;
extern const float* const g_transformer_classifier_b;
extern const float* const g_speech_mel_filterbank;

#ifdef __cplusplus
}
#endif
