idf_component_register(SRCS "eval_snn_ann.cc" "model_data.cc"
                    INCLUDE_DIRS "."
                    REQUIRES esp-tflite-micro driver esp_timer)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3)

# Auswahl der SNN-Kernelvarianten, ueberschreibbar mit -DSNN_SEL_...=<0|1>.
# Gemessen auf dem ESP32-C6 (500 MNIST-Bilder, TTFS, dense, jeweils Mittel aus
# drei Messzyklen, identische Vorhersagen und Spike-Zahlen in allen Varianten):
#
#   Variante                          ms/Inferenz
#   FP32-FC, expliziter Decay             342,2   (Ausgangsstand)
#   + INT8-FC + Lazy Decay                297,4
#   + beta==1-Faltung                     258,1
#   + Tap-Index-Tabelle                   226,1
#   + Sprung statt branch-los im Pass     204,4
#   + output-stationaere Streuung         196,0   (Vorgabe)
#
#   SNN_SEL_LAYOUT_NHWC  1 = NHWC-Layout (oc minor). Auf dem C6 durchgaengig
#                            langsamer als NCHW (345 statt 342 ms): 1 = aus
#   SNN_SEL_INT8         INT8-Conv-Gewichte. Ohne SNN_SEL_CONV_ACC32 muss pro
#                            Gewicht dequantisiert werden; auf einem Kern ohne
#                            Vektorinstruktionen ist das fcvt.w.s der
#                            Flaschenhals. Mit ACC32 nur 297,9 -> 292,9 ms bei
#                            69 KB weniger freiem Heap: 0 = aus
#   SNN_SEL_INT8_FC      INT8-FC-Gewichte. Der Gewinn kommt nicht aus der
#                            Rechenoperation, sondern daraus, dass die Gewichte
#                            in SRAM statt im Flash-Cache liegen: FC-Zeit
#                            1882 ms -> 171 ms: 1 = an
#   SNN_SEL_CONV_ACC32   eigener INT32-Akkumulator je Conv-Neuron, kostet
#                            4 Byte je Neuron (75 KB bei diesem Modell):
#                            0 = aus
#   SNN_SEL_LAZY_DECAY   Decay/Bias lazy in geschlossener Form statt als
#                            eigener Durchlauf: 350 ms -> 293 ms: 1 = an
#   SNN_SEL_DENSE_REF    0 = aus, sonst Anzahl Test-Samples fuer den Selbsttest
#                            gegen den dichten Referenzpfad (nur Verifikation,
#                            um ein Vielfaches langsamer): 0 = aus
#   SNN_PROFILE          Zeitaufschluesselung Conv/Scatter/Schwellenpass/Pool/FC
#                            (kostet esp_timer-Aufrufe je Layer und Schritt):
#                            0 = aus
#
# Ohne SNN_SEL_* wird zusaetzlich ein beta==1-Sonderpfad aktiv: bei beta = 1 ist
# der Decay ein No-op, der Bias laesst sich aus dem Zustand herausziehen
# (mu = mem - b*last, v = mu + b*stamp). Damit entfaellt im Scatter jede
# Decay-Nachfuehrung und der Schwellenpass kommt pro Neuron mit einem Zugriff
# aus. Die uebrigen Schalter bleiben fuer den Vergleich erhalten.
set(SNN_SEL_LAYOUT_NHWC 0 CACHE STRING "NHWC-Layout")
set(SNN_SEL_INT8        0 CACHE STRING "INT8-Conv-Gewichte")
set(SNN_SEL_INT8_FC     1 CACHE STRING "INT8-FC-Gewichte")
set(SNN_SEL_CONV_ACC32  0 CACHE STRING "INT32-Akkumulator je Conv-Neuron")
set(SNN_SEL_LAZY_DECAY  1 CACHE STRING "Lazy Decay")
set(SNN_SEL_DENSE_REF   0 CACHE STRING "Samples im Selbsttest dichter Referenzpfad")
set(SNN_PROFILE         0 CACHE STRING "Zeitaufschluesselung")

target_compile_definitions(${COMPONENT_LIB} PRIVATE
    SNN_SEL_LAYOUT_NHWC=${SNN_SEL_LAYOUT_NHWC}
    SNN_SEL_INT8=${SNN_SEL_INT8}
    SNN_SEL_INT8_FC=${SNN_SEL_INT8_FC}
    SNN_SEL_CONV_ACC32=${SNN_SEL_CONV_ACC32}
    SNN_SEL_LAZY_DECAY=${SNN_SEL_LAZY_DECAY}
    SNN_SEL_DENSE_REF=${SNN_SEL_DENSE_REF}
    SNN_PROFILE=${SNN_PROFILE})