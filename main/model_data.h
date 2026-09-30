#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MODEL_TYPE_ANN 0
#define MODEL_TYPE_SNN 1

#define NEURON_TYPE_LIF 0
#define NEURON_TYPE_IF  1

#define SNN_MAX_LAYERS 8

typedef struct {
    int input_size;
    int hidden_layers_count;
    int layer_sizes[SNN_MAX_LAYERS + 1];
    int num_steps;
    float beta;
    float threshold;
    int output_size;
    int neuron_type;
} snn_config_t;

extern const int g_model_type;
extern const char* const g_model_name;

// ANN-Spezifisch (TensorFlow Lite Micro FlatBuffer)
extern const unsigned char g_model[];
extern const int g_model_len;

// SNN-Spezifisch (LIF Neuronen Parameter & Gewichte)
extern const snn_config_t g_snn_config;
extern const float* const g_snn_weights[];
extern const float* const g_snn_biases[];

#ifdef __cplusplus
}
#endif
