idf_component_register(SRCS "eval_snn_ann.cc" "model_data.cc"
                    INCLUDE_DIRS "."
                    REQUIRES esp-tflite-micro driver esp_timer)
