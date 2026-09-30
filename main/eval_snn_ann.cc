#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"

// TensorFlow Lite Micro (für ANN)
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "mnist_test_data.h"

#define TRIGGER_GPIO GPIO_NUM_4

static const char *TAG = "EVAL_SNN_ANN";

// Tensor Arena für TensorFlow Lite Micro (nur bei ANN)
constexpr size_t kTensorArenaSize = 64 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

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

// SNN Puffer (statisch alloziert)
#define SNN_MAX_NEURONS_PER_LAYER 1024
static float s_snn_mem[SNN_MAX_LAYERS][SNN_MAX_NEURONS_PER_LAYER];
static uint8_t s_snn_spk[SNN_MAX_LAYERS][SNN_MAX_NEURONS_PER_LAYER];
static float s_snn_cur[SNN_MAX_LAYERS][SNN_MAX_NEURONS_PER_LAYER];
static int s_snn_spk_sum[SNN_MAX_NEURONS_PER_LAYER];

static int run_snn_inference(const float *img, uint32_t *out_spikes) {
    const snn_config_t &cfg = g_snn_config;
    const int num_layers = cfg.hidden_layers_count + 1;

    // 1. Membranpotenziale & Spikes zurücksetzen
    for (int l = 0; l < num_layers; ++l) {
        int out_dim = cfg.layer_sizes[l + 1];
        for (int i = 0; i < out_dim; ++i) {
            s_snn_mem[l][i] = 0.0f;
            s_snn_spk[l][i] = 0;
        }
    }
    for (int i = 0; i < cfg.output_size; ++i) {
        s_snn_spk_sum[i] = 0;
    }

    uint32_t total_spk = 0;

    // 2. Eingangsstrom für Schicht 0 vorberechnen (statisches Bild: cur0 = W0 * img + b0)
    int in_dim0 = cfg.layer_sizes[0];
    int out_dim0 = cfg.layer_sizes[1];
    const float *w0 = g_snn_weights[0];
    const float *b0 = g_snn_biases[0];

    for (int i = 0; i < out_dim0; ++i) {
        float sum = b0[i];
        const float *row = &w0[i * in_dim0];
        for (int j = 0; j < in_dim0; ++j) {
            sum += row[j] * img[j];
        }
        s_snn_cur[0][i] = sum;
    }

    // 3. Zeitschleife über num_steps
    for (int t = 0; t < cfg.num_steps; ++t) {
        // Schicht 0 LIF-Schritt (SpikingJelly LIFNode hard-reset: v = beta * v_prev + cur; spike = (v >= th); v = spike ? 0 : v)
        for (int i = 0; i < out_dim0; ++i) {
            float v_new = cfg.beta * s_snn_mem[0][i] + s_snn_cur[0][i];
            uint8_t s = (v_new >= cfg.threshold) ? 1 : 0;
            s_snn_mem[0][i] = s ? 0.0f : v_new;
            s_snn_spk[0][i] = s;
            if (s) {
                total_spk++;
                if (num_layers == 1) {
                    s_snn_spk_sum[i]++;
                }
            }
        }

        // Nachfolgende Schichten l = 1 .. num_layers - 1
        for (int l = 1; l < num_layers; ++l) {
            int in_dim = cfg.layer_sizes[l];
            int out_dim = cfg.layer_sizes[l + 1];
            const float *wl = g_snn_weights[l];
            const float *bl = g_snn_biases[l];
            const uint8_t *prev_spk = s_snn_spk[l - 1];

            // Event-driven Akkumulation (nur wenn vorheriges Neuron gefeuert hat!)
            for (int i = 0; i < out_dim; ++i) {
                float sum = bl[i];
                const float *row = &wl[i * in_dim];
                for (int j = 0; j < in_dim; ++j) {
                    if (prev_spk[j]) {
                        sum += row[j];
                    }
                }
                s_snn_cur[l][i] = sum;
            }

            // LIF-Schritt für Schicht l
            for (int i = 0; i < out_dim; ++i) {
                float v_new = cfg.beta * s_snn_mem[l][i] + s_snn_cur[l][i];
                uint8_t s = (v_new >= cfg.threshold) ? 1 : 0;
                s_snn_mem[l][i] = s ? 0.0f : v_new;
                s_snn_spk[l][i] = s;
                if (s) {
                    total_spk++;
                    if (l == num_layers - 1) {
                        s_snn_spk_sum[i]++;
                    }
                }
            }
        }
    }

    // 4. Argmax der akkumulierten Spikes
    int best_class = 0;
    int max_spk = s_snn_spk_sum[0];
    for (int i = 1; i < cfg.output_size; ++i) {
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

    // 2. TensorFlow Lite Micro Modell initialisieren (fuer ANN und SNN)
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;

    const tflite::Model *model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Modell Schema-Version %d stimmt nicht mit TFLM-Version %d ueberein!",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    ESP_LOGI(TAG, "TensorFlow Lite Micro Modell erfolgreich geladen (Groesse: %d Bytes, Typ: %s).",
             g_model_len, (g_model_type == MODEL_TYPE_SNN) ? "SNN" : "ANN");

    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();

    static tflite::MicroInterpreter interp(model, resolver, tensor_arena, kTensorArenaSize);
    if (interp.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Tensor-Allokation fehlgeschlagen! Arena-Groesse vergroessern?");
        return;
    }
    interpreter = &interp;
    input = interpreter->input(0);
    output = interpreter->output(0);
    ESP_LOGI(TAG, "TFLM Engine initialisiert. Input Dim=%d, Output Dim=%d", input->dims->data[1], output->dims->data[1]);

    if (g_model_type == MODEL_TYPE_SNN) {
        const char *ntype_str = (g_snn_config.neuron_type == NEURON_TYPE_IF) ? "IF" : "LIF";
        ESP_LOGI(TAG, "SNN SpikingJelly Parameter: Neuron=%s, %d Schichten, %d Steps, Beta=%.2f, Thresh=%.2f",
                 ntype_str, g_snn_config.hidden_layers_count + 1, g_snn_config.num_steps,
                 g_snn_config.beta, g_snn_config.threshold);
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
                for (int j = 0; j < MNIST_IMAGE_SIZE; ++j) {
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

            int expected = (int)mnist_test_labels[i];
            if (predicted == expected) {
                correct_count++;
            }
        }

        int64_t end_time = esp_timer_get_time();

        // Ende der Messung: Trigger-Pin auf LOW
        gpio_set_level(TRIGGER_GPIO, 0);
        ESP_LOGI(TAG, ">>> STOPP: Setze GPIO %d auf LOW (Inferenz beendet) <<<", TRIGGER_GPIO);

        int64_t total_time_us = end_time - start_time;
        double avg_time_ms = (double)total_time_us / (1000.0 * MNIST_TEST_SUBSET);
        double accuracy = (100.0 * correct_count) / MNIST_TEST_SUBSET;

        ESP_LOGI(TAG, "================ ESP32 TEST ACCURACY ================");
        ESP_LOGI(TAG, "Modell:               %s (%s)", g_model_name, (g_model_type == MODEL_TYPE_SNN) ? "SNN" : "ANN");
        ESP_LOGI(TAG, "Gesamtdauer:          %" PRId64 " us (%.2f ms)", total_time_us, (double)total_time_us / 1000.0);
        ESP_LOGI(TAG, "Dauer pro Bild:       %.3f ms", avg_time_ms);
        ESP_LOGI(TAG, "Klassifikationen:     %d / %d korrekt (Genauigkeit: %.2f %%)", correct_count, MNIST_TEST_SUBSET, accuracy);
        ESP_LOGI(TAG, "ESP32 Test Accuracy:  %.2f %%", accuracy);
        ESP_LOGI(TAG, "ESP32_METRICS: ACC=%.2f;CORRECT=%d;TOTAL=%d;TIME_US=%" PRId64, accuracy, correct_count, MNIST_TEST_SUBSET, total_time_us);
        if (g_model_type == MODEL_TYPE_SNN) {
            double avg_spikes = (double)total_all_spikes / (double)MNIST_TEST_SUBSET;
            ESP_LOGI(TAG, "Gesamt-Spikes:        %" PRIu64 " (durchschnittlich %.1f Spikes/Inferenz)", total_all_spikes, avg_spikes);
        }
        ESP_LOGI(TAG, "=====================================================\n");
        ESP_LOGI(TAG, "Messzyklus abgeschlossen. Warte 5 Sekunden bis zum naechsten Zyklus...\n");

        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
