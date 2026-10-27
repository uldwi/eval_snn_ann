# ANN/SNN Evaluation on ESP32

Train ANN and SNN models in PyTorch, evaluate them on datasets, and measure supported models on an ESP32 with a Joulescope.

## Project layout

```text
eval_snn_ann/             Python package
  config.py               configuration loading and architecture parsing
  data/                   dataset registry and preprocessing
  models/                 ANN, SNN, SpikCommander, and model factory
  optimization.py         evaluation, pruning, quantization, clustering
  training.py             training loops, checkpoints, and cache signatures
  exporting.py            TFLite and ESP-IDF C data export
  workflow.py             configured multi-model training workflow
  measurement/power.py    ESP32/Joulescope measurement workflow
main/                     ESP-IDF application and generated model data
tests/                    Python tests
config.yaml               model and board configuration
```

`train_mnist.py`, `datasets_hub.py`, and `measure_power.py` remain as compatibility entry points. New Python code should import from `eval_snn_ann`.

## Common commands

```bash
pip install -r requirements.txt
python train_mnist.py --config config.yaml --list-datasets
python train_mnist.py --config config.yaml --device cuda --skip-evaluation
python measure_power.py --config config.yaml
python -m unittest discover -s tests
```

The ESP-IDF project stays at the repository root because `main/` and the root `CMakeLists.txt` are the firmware build entry points. Datasets, checkpoints, and ESP-IDF build outputs are local generated data and are not committed.
