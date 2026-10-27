# ANN/SNN Evaluation on ESP32

This project trains artificial neural networks (ANNs) and spiking neural networks (SNNs) with PyTorch, exports supported models for ESP32, and measures inference time and energy on the device. Model and board selection are configured in `config.yaml`.

## Quick start

1. Clone the repository and create a virtual environment.
2. Install a PyTorch build that matches your Python version and hardware.
3. Install the project dependencies and download the datasets you need.
4. Review `config.yaml` and train the configured models.
5. Set up ESP-IDF, an ESP32 board, and a Joulescope before taking measurements.

### Installation

PyTorch, TorchVision, and, for Speech Commands, TorchAudio must use compatible versions. Install them together from the same CPU or CUDA build. For a CPU-only environment, for example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA, use the selector at [pytorch.org](https://pytorch.org/get-started/locally/) and install all three packages from the matching build. TorchAudio is required for the real Speech Commands dataset. MNE and scikit-learn, both listed in `requirements.txt`, are used to prepare the PhysioNet EEGMI data.

### Datasets and training

List supported datasets and aliases:

```bash
python train_mnist.py --list-datasets
```

Train the models in the configuration:

```bash
python train_mnist.py --config config.yaml
```

Useful options:

```bash
python train_mnist.py --help
python train_mnist.py --config config.yaml --device cuda
python train_mnist.py --config config.yaml --dataset speech_commands --epochs 80
python train_mnist.py --config config.yaml --force-train
```

`--dataset` filters the configured models by dataset. Without a filter, all entries under `models` are processed in order. `--force-train` bypasses matching checkpoint caches and retrains the models. Per-model training parameters, architectures, optimization, and early stopping are configured in `config.yaml`. Before a run, review fields such as `epochs`, `batch_size`, `learning_rate`, `dataset`, `architecture`, and `optimization`.

Training checkpoints are stored in `saved_models/` by default. Datasets are stored in `data/` by default. These paths contain local generated data and should not be committed to Git.

### ESP32 power measurements

`measure_power.py` runs the configured training/export workflow, builds and flashes the ESP-IDF firmware, and measures it with a Joulescope. Before running it, activate ESP-IDF, select the board in `config.yaml`, connect the ESP32, and make the Joulescope available.

```bash
source "$HOME/esp/esp-idf/export.sh"
python measure_power.py --config config.yaml --list-datasets
python measure_power.py --config config.yaml --no-wait
```

After flashing, the script waits for input by default so you can disconnect the USB cable. `--no-wait` skips that pause. `--no-flash` disables automatic flashing. `--no-train` skips training and export and requires compatible exported model data to already exist. See `python measure_power.py --help` for all options.

Compare measurements only when input data, preprocessing, board, clock, build configuration, and measurement conditions match. The Speech Commands model computes its Mel features on the ESP32 so that preprocessing cost is included.

### Switching ESP-IDF targets

Supported targets are `esp32s3` and `esp32c6`:

```bash
./set_target.sh esp32s3
./set_target.sh esp32c6
```

The script updates `microcontroller.target`, selects the ESP-IDF target, and rebuilds the firmware. It deletes the `build/` directory as part of the target change.

## Configuration

`config.yaml` contains the model list, dataset settings, measurement parameters, and board configuration. Models are processed in the order they appear under `models`. Each entry starts with a unique `name` and defines at least a model type, dataset, and architecture.

Example model entry:

```yaml
models:
  - name: ANN_MNIST_Example
    type: ann
    architecture: standard
    dataset: mnist
    epochs: 20
    batch_size: 64
    learning_rate: 0.001
```

SNN-specific fields such as `neuron_type`, `encoding_type`, `num_steps`, `threshold`, and `output_readout` depend on the model. Use `--list-datasets` for valid dataset names and `--help` for supported CLI overrides.

## Project structure

```text
eval_snn_ann/
  config.py                 Configuration loading and architecture parameters
  data/                     Dataset registry, loaders, and preprocessing
  models/                   ANN, SNN, Transformer models, and model factory
  optimization.py           Evaluation, pruning, quantization, and clustering
  training.py               Training loops, checkpoints, and cache signatures
  exporting.py              TFLite and C export
  workflow.py               Multi-model training workflow
  measurement/power.py      ESP32 and Joulescope workflow
tests/                      Python tests
main/                       ESP-IDF application
config.yaml                 Models, board, and measurement configuration
train_mnist.py              Training entry point (compatibility wrapper)
measure_power.py            Measurement entry point (compatibility wrapper)
datasets_hub.py             Dataset entry point (compatibility wrapper)
```

New Python modules should import from `eval_snn_ann`. The root scripts remain available for existing commands and integrations. The ESP-IDF project stays at the repository root because ESP-IDF expects `CMakeLists.txt` and `main/` there.

## DACHS

The DACHS workflow has a login-node setup step for dataset downloads and a SLURM training script:

```bash
./dachs_setup.sh
sbatch dachs_train.sh
```

See [DACHS_TRAINING.md](DACHS_TRAINING.md) for prerequisites and details. Adjust the partition, runtime, and resources in the SLURM directives to match current cluster access.

## Developer notes

Tests are located in `tests/`. Run them with:

```bash
python -m unittest discover -s tests
```

Package metadata is in `pyproject.toml`. An editable install is optional:

```bash
python -m pip install -e .
```

Export and device inference do not support every PyTorch architecture or operator. Before measuring a model, confirm that it exports successfully and runs on the target firmware.
