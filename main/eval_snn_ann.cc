#include <stdio.h>
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
#include "esp_heap_caps.h"

// TensorFlow Lite Micro (für ANN)
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "mnist_test_data.h"

#define TRIGGER_GPIO GPIO_NUM_4

static const char *TAG = "EVAL_SNN_ANN";

// Tensor Arena für TensorFlow Lite Micro (nur bei ANN)
constexpr size_t kTensorArenaSize = 128 * 1024;
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

// SNN Dynamische Puffer
static float *s_snn_conv_mem[SNN_MAX_CONV_LAYERS] = {nullptr};
static uint8_t *s_snn_conv_spk[SNN_MAX_CONV_LAYERS] = {nullptr};
static uint8_t *s_snn_conv_pool_spk[SNN_MAX_CONV_LAYERS] = {nullptr};

static float *s_snn_fc_mem[SNN_MAX_FC_LAYERS] = {nullptr};
static uint8_t *s_snn_fc_spk[SNN_MAX_FC_LAYERS] = {nullptr};
static float *s_snn_fc_cur[SNN_MAX_FC_LAYERS] = {nullptr};
static int s_snn_spk_sum[128] = {0};
static int16_t s_snn_in_tf[MNIST_IMAGE_SIZE];
static uint8_t s_snn_in_spk[MNIST_IMAGE_SIZE];
static uint8_t *s_snn_flat_spk = nullptr;

static bool init_snn_buffers(void) {
    const snn_config_t &cfg = g_snn_config;
    for (int l = 0; l < cfg.num_conv_layers; ++l) {
        int n_neurons = cfg.conv_layers[l].out_channels * cfg.conv_layers[l].out_h * cfg.conv_layers[l].out_w;
        s_snn_conv_mem[l] = (float*)heap_caps_calloc(n_neurons, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        s_snn_conv_spk[l] = (uint8_t*)heap_caps_calloc(n_neurons, sizeof(uint8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_snn_conv_mem[l] || !s_snn_conv_spk[l]) return false;

        if (cfg.conv_layers[l].pool_size > 1) {
            int n_pool = cfg.conv_layers[l].out_channels * cfg.conv_layers[l].pool_out_h * cfg.conv_layers[l].pool_out_w;
            s_snn_conv_pool_spk[l] = (uint8_t*)heap_caps_calloc(n_pool, sizeof(uint8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
            if (!s_snn_conv_pool_spk[l]) return false;
        }
    }

    if (cfg.num_conv_layers > 0 && cfg.flat_size > 0) {
        s_snn_flat_spk = (uint8_t*)heap_caps_calloc(cfg.flat_size, sizeof(uint8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_snn_flat_spk) return false;
    }

    int num_fc = cfg.hidden_layers_count + 1;
    for (int l = 0; l < num_fc; ++l) {
        int out_dim = cfg.fc_layer_sizes[l + 1];
        s_snn_fc_mem[l] = (float*)heap_caps_calloc(out_dim, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        s_snn_fc_spk[l] = (uint8_t*)heap_caps_calloc(out_dim, sizeof(uint8_t), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        s_snn_fc_cur[l] = (float*)heap_caps_calloc(out_dim, sizeof(float), MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!s_snn_fc_mem[l] || !s_snn_fc_spk[l] || !s_snn_fc_cur[l]) return false;
    }
    return true;
}

static int run_snn_inference(const float *img, uint32_t *out_spikes) {
    const snn_config_t &cfg = g_snn_config;
    const int num_fc_layers = cfg.hidden_layers_count + 1;
    uint32_t total_spk = 0;

    // 1. Membranpotenziale & Spikes zurücksetzen
    for (int l = 0; l < cfg.num_conv_layers; ++l) {
        int n_neurons = cfg.conv_layers[l].out_channels * cfg.conv_layers[l].out_h * cfg.conv_layers[l].out_w;
        for (int i = 0; i < n_neurons; ++i) {
            s_snn_conv_mem[l][i] = 0.0f;
            s_snn_conv_spk[l][i] = 0;
        }
        if (cfg.conv_layers[l].pool_size > 1) {
            int n_pool = cfg.conv_layers[l].out_channels * cfg.conv_layers[l].pool_out_h * cfg.conv_layers[l].pool_out_w;
            for (int i = 0; i < n_pool; ++i) {
                s_snn_conv_pool_spk[l][i] = 0;
            }
        }
    }
    for (int l = 0; l < num_fc_layers; ++l) {
        int out_dim = cfg.fc_layer_sizes[l + 1];
        for (int i = 0; i < out_dim; ++i) {
            s_snn_fc_mem[l][i] = 0.0f;
            s_snn_fc_spk[l][i] = 0;
            s_snn_fc_cur[l][i] = 0.0f;
        }
    }
    for (int i = 0; i < cfg.output_size && i < 128; ++i) {
        s_snn_spk_sum[i] = 0;
    }

    // 2. Eingangs-Encoding vorverarbeiten
    if (cfg.encoding_type == ENCODING_TTFS) {
        for (int j = 0; j < cfg.input_size; ++j) {
            float val = img[j];
            if (val > 0.0f) {
                float clamped = (val > 1.0f) ? 1.0f : val;
                s_snn_in_tf[j] = (int16_t)roundf((float)(cfg.num_steps - 1) * (1.0f - clamped));
            } else {
                s_snn_in_tf[j] = -1;
            }
        }
    } else if (cfg.encoding_type == ENCODING_PERIODIC) {
        for (int j = 0; j < cfg.input_size; ++j) {
            float val = img[j];
            float clamped = (val < 0.0f) ? 0.0f : ((val > 1.0f) ? 1.0f : val);
            s_snn_in_tf[j] = (int16_t)roundf(clamped * (float)cfg.num_steps);
        }
    }

    // Bei reinem FC-Netzwerk mit Direct Coding: Schicht 0 Eingangsstrom vorberechnen
    if (cfg.num_conv_layers == 0 && cfg.encoding_type == ENCODING_DIRECT) {
        int in_dim0 = cfg.fc_layer_sizes[0];
        int out_dim0 = cfg.fc_layer_sizes[1];
        const float *w0 = g_snn_fc_weights[0];
        const float *b0 = g_snn_fc_biases[0];
        for (int i = 0; i < out_dim0; ++i) {
            float sum = b0[i];
            const float *row = &w0[i * in_dim0];
            for (int j = 0; j < in_dim0; ++j) {
                sum += row[j] * img[j];
            }
            s_snn_fc_cur[0][i] = sum;
        }
    }

    // 3. Zeitschleife über num_steps
    for (int t = 0; t < cfg.num_steps; ++t) {
        if (cfg.num_conv_layers > 0) {
            // Convolutional SNN Pipeline
            for (int l = 0; l < cfg.num_conv_layers; ++l) {
                const snn_conv_layer_cfg_t &cl = cfg.conv_layers[l];
                const float *w = g_snn_conv_weights[l];
                const float *b = g_snn_conv_biases[l];

                for (int oc = 0; oc < cl.out_channels; ++oc) {
                    for (int oh = 0; oh < cl.out_h; ++oh) {
                        for (int ow = 0; ow < cl.out_w; ++ow) {
                            float sum = b[oc];
                            for (int ic = 0; ic < cl.in_channels; ++ic) {
                                for (int kh = 0; kh < cl.kernel_size; ++kh) {
                                    int ih = oh * cl.stride - cl.padding + kh;
                                    if (ih < 0 || ih >= cl.in_h) continue;
                                    for (int kw = 0; kw < cl.kernel_size; ++kw) {
                                        int iw = ow * cl.stride - cl.padding + kw;
                                        if (iw < 0 || iw >= cl.in_w) continue;

                                        float in_val = 0.0f;
                                        if (l == 0) {
                                            int in_idx = ih * cl.in_w + iw;
                                            if (cfg.encoding_type == ENCODING_DIRECT) {
                                                in_val = img[in_idx];
                                            } else if (cfg.encoding_type == ENCODING_TTFS) {
                                                in_val = (s_snn_in_tf[in_idx] == t) ? 1.0f : 0.0f;
                                            } else if (cfg.encoding_type == ENCODING_PERIODIC) {
                                                in_val = (t < s_snn_in_tf[in_idx]) ? 1.0f : 0.0f;
                                            } else if (cfg.encoding_type == ENCODING_POISSON) {
                                                float p = img[in_idx];
                                                if (p > 0.0f) {
                                                    float r = (float)(esp_random() & 0xFFFF) / 65535.0f;
                                                    in_val = (r < p) ? 1.0f : 0.0f;
                                                }
                                            }
                                        } else {
                                            const snn_conv_layer_cfg_t &prev_cl = cfg.conv_layers[l - 1];
                                            if (prev_cl.pool_size > 1) {
                                                int prev_idx = (ic * prev_cl.pool_out_h + ih) * prev_cl.pool_out_w + iw;
                                                in_val = (float)s_snn_conv_pool_spk[l - 1][prev_idx];
                                            } else {
                                                int prev_idx = (ic * prev_cl.out_h + ih) * prev_cl.out_w + iw;
                                                in_val = (float)s_snn_conv_spk[l - 1][prev_idx];
                                            }
                                        }

                                        if (in_val != 0.0f) {
                                            int w_idx = ((oc * cl.in_channels + ic) * cl.kernel_size + kh) * cl.kernel_size + kw;
                                            sum += w[w_idx] * in_val;
                                        }
                                    }
                                }
                            }

                            int out_idx = (oc * cl.out_h + oh) * cl.out_w + ow;
                            float v_new = cfg.beta * s_snn_conv_mem[l][out_idx] + sum;
                            uint8_t s = (v_new >= cfg.threshold) ? 1 : 0;
                            s_snn_conv_mem[l][out_idx] = s ? 0.0f : v_new;
                            s_snn_conv_spk[l][out_idx] = s;
                            if (s) {
                                total_spk++;
                            }
                        }
                    }
                }

                // Pooling
                if (cl.pool_size > 1) {
                    for (int oc = 0; oc < cl.out_channels; ++oc) {
                        for (int ph = 0; ph < cl.pool_out_h; ++ph) {
                            for (int pw = 0; pw < cl.pool_out_w; ++pw) {
                                uint8_t max_spk = 0;
                                for (int pkh = 0; pkh < cl.pool_size; ++pkh) {
                                    int oh = ph * cl.pool_stride + pkh;
                                    if (oh >= cl.out_h) continue;
                                    for (int pkw = 0; pkw < cl.pool_size; ++pkw) {
                                        int ow = pw * cl.pool_stride + pkw;
                                        if (ow >= cl.out_w) continue;
                                        int idx = (oc * cl.out_h + oh) * cl.out_w + ow;
                                        if (s_snn_conv_spk[l][idx]) {
                                            max_spk = 1;
                                            break;
                                        }
                                    }
                                    if (max_spk) break;
                                }
                                int p_idx = (oc * cl.pool_out_h + ph) * cl.pool_out_w + pw;
                                s_snn_conv_pool_spk[l][p_idx] = max_spk;
                            }
                        }
                    }
                }
            }

            // Flatten
            const snn_conv_layer_cfg_t &last_cl = cfg.conv_layers[cfg.num_conv_layers - 1];
            if (last_cl.pool_size > 1) {
                int total_flat = last_cl.out_channels * last_cl.pool_out_h * last_cl.pool_out_w;
                for (int i = 0; i < total_flat; ++i) {
                    s_snn_flat_spk[i] = s_snn_conv_pool_spk[cfg.num_conv_layers - 1][i];
                }
            } else {
                int total_flat = last_cl.out_channels * last_cl.out_h * last_cl.out_w;
                for (int i = 0; i < total_flat; ++i) {
                    s_snn_flat_spk[i] = s_snn_conv_spk[cfg.num_conv_layers - 1][i];
                }
            }

            // FC Layers
            const uint8_t *in_spikes = s_snn_flat_spk;
            for (int l = 0; l < num_fc_layers; ++l) {
                int in_dim = cfg.fc_layer_sizes[l];
                int out_dim = cfg.fc_layer_sizes[l + 1];
                const float *wl = g_snn_fc_weights[l];
                const float *bl = g_snn_fc_biases[l];

                for (int i = 0; i < out_dim; ++i) {
                    float sum = bl[i];
                    const float *row = &wl[i * in_dim];
                    for (int j = 0; j < in_dim; ++j) {
                        if (in_spikes[j]) {
                            sum += row[j];
                        }
                    }
                    float v_new = cfg.beta * s_snn_fc_mem[l][i] + sum;
                    uint8_t s = (v_new >= cfg.threshold) ? 1 : 0;
                    s_snn_fc_mem[l][i] = s ? 0.0f : v_new;
                    s_snn_fc_spk[l][i] = s;
                    if (s) {
                        total_spk++;
                        if (l == num_fc_layers - 1) {
                            s_snn_spk_sum[i]++;
                        }
                    }
                }
                in_spikes = s_snn_fc_spk[l];
            }

        } else {
            // Pure Fully-Connected SNN
            int in_dim0 = cfg.fc_layer_sizes[0];
            int out_dim0 = cfg.fc_layer_sizes[1];
            const float *w0 = g_snn_fc_weights[0];
            const float *b0 = g_snn_fc_biases[0];

            if (cfg.encoding_type == ENCODING_TTFS) {
                for (int j = 0; j < in_dim0; ++j) {
                    s_snn_in_spk[j] = (s_snn_in_tf[j] == t) ? 1 : 0;
                }
                for (int i = 0; i < out_dim0; ++i) {
                    float sum = b0[i];
                    const float *row = &w0[i * in_dim0];
                    for (int j = 0; j < in_dim0; ++j) {
                        if (s_snn_in_spk[j]) sum += row[j];
                    }
                    s_snn_fc_cur[0][i] = sum;
                }
            } else if (cfg.encoding_type == ENCODING_PERIODIC) {
                for (int j = 0; j < in_dim0; ++j) {
                    s_snn_in_spk[j] = (t < s_snn_in_tf[j]) ? 1 : 0;
                }
                for (int i = 0; i < out_dim0; ++i) {
                    float sum = b0[i];
                    const float *row = &w0[i * in_dim0];
                    for (int j = 0; j < in_dim0; ++j) {
                        if (s_snn_in_spk[j]) sum += row[j];
                    }
                    s_snn_fc_cur[0][i] = sum;
                }
            } else if (cfg.encoding_type == ENCODING_POISSON) {
                for (int j = 0; j < in_dim0; ++j) {
                    float r = (float)(esp_random() & 0xFFFF) / 65535.0f;
                    float clamped = (img[j] < 0.0f) ? 0.0f : ((img[j] > 1.0f) ? 1.0f : img[j]);
                    s_snn_in_spk[j] = (r < clamped) ? 1 : 0;
                }
                for (int i = 0; i < out_dim0; ++i) {
                    float sum = b0[i];
                    const float *row = &w0[i * in_dim0];
                    for (int j = 0; j < in_dim0; ++j) {
                        if (s_snn_in_spk[j]) sum += row[j];
                    }
                    s_snn_fc_cur[0][i] = sum;
                }
            }

            // Schicht 0 LIF/IF
            for (int i = 0; i < out_dim0; ++i) {
                float v_new = cfg.beta * s_snn_fc_mem[0][i] + s_snn_fc_cur[0][i];
                uint8_t s = (v_new >= cfg.threshold) ? 1 : 0;
                s_snn_fc_mem[0][i] = s ? 0.0f : v_new;
                s_snn_fc_spk[0][i] = s;
                if (s) {
                    total_spk++;
                    if (num_fc_layers == 1) {
                        s_snn_spk_sum[i]++;
                    }
                }
            }

            // Nachfolgende Schichten
            for (int l = 1; l < num_fc_layers; ++l) {
                int in_dim = cfg.fc_layer_sizes[l];
                int out_dim = cfg.fc_layer_sizes[l + 1];
                const float *wl = g_snn_fc_weights[l];
                const float *bl = g_snn_fc_biases[l];
                const uint8_t *prev_spk = s_snn_fc_spk[l - 1];

                for (int i = 0; i < out_dim; ++i) {
                    float sum = bl[i];
                    const float *row = &wl[i * in_dim];
                    for (int j = 0; j < in_dim; ++j) {
                        if (prev_spk[j]) sum += row[j];
                    }
                    float v_new = cfg.beta * s_snn_fc_mem[l][i] + sum;
                    uint8_t s = (v_new >= cfg.threshold) ? 1 : 0;
                    s_snn_fc_mem[l][i] = s ? 0.0f : v_new;
                    s_snn_fc_spk[l][i] = s;
                    if (s) {
                        total_spk++;
                        if (l == num_fc_layers - 1) {
                            s_snn_spk_sum[i]++;
                        }
                    }
                }
            }
        }
    }

    // 4. Argmax
    int best_class = 0;
    int max_spk = s_snn_spk_sum[0];
    for (int i = 1; i < cfg.output_size && i < 128; ++i) {
        if (s_snn_spk_sum[i] > max_spk) {
            max_spk = s_snn_spk_sum[i];
            best_class = i;
        }
    }

    if (out_spikes) {
        *out_spikes = total_spk;
    }
    return best_class;
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "=== Initialisiere ESP32 MNIST Inferenz: %s (%s) ===",
             g_model_name, (g_model_type == MODEL_TYPE_SNN) ? "SNN" : "ANN");

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

    if (g_model_type == MODEL_TYPE_ANN) {
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (!tensor_arena) {
            ESP_LOGE(TAG, "Allokation der Tensor-Arena (%d Bytes) fehlgeschlagen!", (int)kTensorArenaSize);
            return;
        }

        const tflite::Model *model = tflite::GetModel(g_model);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            ESP_LOGE(TAG, "Modell Schema-Version %d stimmt nicht mit TFLM-Version %d ueberein!",
                     model->version(), TFLITE_SCHEMA_VERSION);
            return;
        }
        ESP_LOGI(TAG, "TensorFlow Lite Micro Modell erfolgreich geladen (Groesse: %d Bytes, Typ: ANN).", g_model_len);

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
            ESP_LOGE(TAG, "Tensor-Allokation fehlgeschlagen! Arena-Groesse vergroessern?");
            return;
        }
        interpreter = &interp;
        input = interpreter->input(0);
        output = interpreter->output(0);
        ESP_LOGI(TAG, "TFLM Engine initialisiert. Input Dims: [%d, %d, %d, %d]",
                 input->dims->data[0],
                 (input->dims->size > 1) ? input->dims->data[1] : 1,
                 (input->dims->size > 2) ? input->dims->data[2] : 1,
                 (input->dims->size > 3) ? input->dims->data[3] : 1);
    } else {
        if (!init_snn_buffers()) {
            ESP_LOGE(TAG, "Allokation der SNN-Puffer fehlgeschlagen!");
            return;
        }

        const char *ntype_str = (g_snn_config.neuron_type == NEURON_TYPE_IF) ? "IF" : "LIF";
        const char *enc_str = "DIRECT";
        if (g_snn_config.encoding_type == ENCODING_TTFS) enc_str = "TTFS";
        else if (g_snn_config.encoding_type == ENCODING_POISSON) enc_str = "POISSON";
        else if (g_snn_config.encoding_type == ENCODING_PERIODIC) enc_str = "PERIODIC";

        ESP_LOGI(TAG, "SNN SpikingJelly Parameter: Neuron=%s, Encoding=%s, Conv=%d, FC=%d, %d Steps, Beta=%.2f, Thresh=%.2f",
                 ntype_str, enc_str, g_snn_config.num_conv_layers, g_snn_config.hidden_layers_count + 1,
                 g_snn_config.num_steps, g_snn_config.beta, g_snn_config.threshold);
    }

    ESP_LOGI(TAG, "Anzahl Test-Samples: %d", MNIST_TEST_SUBSET);
    ESP_LOGI(TAG, "Trigger-Pin GPIO %d (mit Joulescope IN1 verbunden) bereit.\n", TRIGGER_GPIO);

    while (1) {
        ESP_LOGI(TAG, "Warte 3 Sekunden vor Start des Inferenz-Messzyklus...");
        vTaskDelay(pdMS_TO_TICKS(3000));

        // Start der Messung: Trigger-Pin auf HIGH
        ESP_LOGI(TAG, ">>> START: Setze GPIO %d auf HIGH (Inferenz laeuft) <<<", TRIGGER_GPIO);
        gpio_set_level(TRIGGER_GPIO, 1);

        int64_t start_time = esp_timer_get_time();
        int correct_count = 0;
        uint64_t total_all_spikes = 0;

        for (int i = 0; i < MNIST_TEST_SUBSET; ++i) {
            const float *img_data = mnist_test_images[i];
            int predicted = -1;

            if (g_model_type == MODEL_TYPE_ANN && interpreter) {
                // ANN Inferenz
                int num_elements = 1;
                for (int d = 0; d < input->dims->size; ++d) {
                    num_elements *= input->dims->data[d];
                }
                for (int j = 0; j < num_elements && j < MNIST_IMAGE_SIZE; ++j) {
                    input->data.f[j] = img_data[j];
                }
                if (interpreter->Invoke() == kTfLiteOk) {
                    predicted = argmax_float(output->data.f, MNIST_NUM_OUTPUTS);
                }
            } else {
                // SNN Inferenz
                uint32_t sample_spikes = 0;
                predicted = run_snn_inference(img_data, &sample_spikes);
                total_all_spikes += sample_spikes;
            }

            if (predicted == mnist_test_labels[i]) {
                correct_count++;
            }
        }

        int64_t end_time = esp_timer_get_time();

        // Ende der Messung: Trigger-Pin auf LOW
        gpio_set_level(TRIGGER_GPIO, 0);
        ESP_LOGI(TAG, ">>> ENDE: Setze GPIO %d auf LOW (Inferenz beendet) <<<", TRIGGER_GPIO);

        int64_t elapsed_us = end_time - start_time;
        float elapsed_ms = (float)elapsed_us / 1000.0f;
        float time_per_inf_ms = elapsed_ms / (float)MNIST_TEST_SUBSET;
        float accuracy_pct = (100.0f * (float)correct_count) / (float)MNIST_TEST_SUBSET;

        ESP_LOGI(TAG, "Inferenz-Messung abgeschlossen:");
        ESP_LOGI(TAG, "  - Modell:             %s (%s)", g_model_name, (g_model_type == MODEL_TYPE_SNN) ? "SNN" : "ANN");
        ESP_LOGI(TAG, "  - Samples evaluiert:  %d", MNIST_TEST_SUBSET);
        ESP_LOGI(TAG, "  - Korrekte Samples:   %d / %d", correct_count, MNIST_TEST_SUBSET);
        ESP_LOGI(TAG, "  - Test-Genauigkeit:   %.2f %%", accuracy_pct);
        ESP_LOGI(TAG, "  - Gesamtdauer:        %.3f ms", elapsed_ms);
        ESP_LOGI(TAG, "  - Zeit pro Inferenz:  %.3f ms (%.1f Infs/s)",
                 time_per_inf_ms, (time_per_inf_ms > 0) ? (1000.0f / time_per_inf_ms) : 0.0f);

        if (g_model_type == MODEL_TYPE_SNN) {
            float avg_spk_per_inf = (float)total_all_spikes / (float)MNIST_TEST_SUBSET;
            ESP_LOGI(TAG, "  - Spikes gesamt:      %" PRIu64 " (Ø %.1f Spikes/Inferenz)",
                     total_all_spikes, avg_spk_per_inf);
            printf("ESP32_METRICS: ACC=%.2f;CORRECT=%d;TOTAL=%d;TIME_US=%" PRId64 ";SPIKES=%" PRIu64 "\n",
                   accuracy_pct, correct_count, MNIST_TEST_SUBSET, elapsed_us, total_all_spikes);
        } else {
            printf("ESP32_METRICS: ACC=%.2f;CORRECT=%d;TOTAL=%d;TIME_US=%" PRId64 "\n",
                   accuracy_pct, correct_count, MNIST_TEST_SUBSET, elapsed_us);
        }

        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
