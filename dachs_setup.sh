#!/bin/bash
# dachs_setup.sh - Run on a DACHS login node (not as a batch job).
# Download datasets to persistent $HOME storage
# so the SLURM job can copy them to $TMPDIR on a compute node.

set -e

echo "=== DACHS setup: downloading datasets (login node) ==="

# Change to the project directory
PROJECT_DIR="$HOME/code/eval_snn_ann"
cd "$PROJECT_DIR"

# Fail before creating environments or downloading data if the checkout is
# missing files introduced by the package-layout refactor.
if [ ! -f "eval_snn_ann/__init__.py" ] || [ ! -f "eval_snn_ann/data/__init__.py" ] || [ ! -f "eval_snn_ann/data/hub.py" ]; then
    echo "ERROR: The project checkout at '$PROJECT_DIR' is incomplete or outdated."
    echo "The current code requires eval_snn_ann/data/{__init__.py,hub.py}."
    echo "Update the entire checkout (for example, run 'git pull' from the repository)"
    echo "or copy the complete project tree from your updated working copy, then rerun this script."
    exit 1
fi

# Create a Python virtual environment on the login node (recommended for consistency)
if [ ! -d "dachs_env" ]; then
    python3 -m venv dachs_env
fi
source dachs_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -r requirements.txt

# Load each dataset once (downloads it if it is not already available)
echo "Downloading MNIST..."
python3 -c "from datasets_hub import get_dataset; get_dataset('mnist', data_dir='./data', download=True); print('MNIST OK')"

echo "Downloading Fashion-MNIST..."
python3 -c "from datasets_hub import get_dataset; get_dataset('fashion_mnist', data_dir='./data', download=True); print('Fashion-MNIST OK')"

echo "Downloading CIFAR-10..."
python3 -c "from datasets_hub import get_dataset; get_dataset('cifar10', data_dir='./data', download=True); print('CIFAR-10 OK')"

echo "Downloading Speech Commands (KWS)..."
python3 -c "from datasets_hub import get_dataset; get_dataset('speech_commands', data_dir='./data', download=True); print('Speech Commands OK')"

echo "Loading HAR time series..."
python3 -c "from datasets_hub import get_dataset; get_dataset('har_time_series', data_dir='./data', download=True); print('HAR OK')"

echo "Loading synthetic time series..."
python3 -c "from datasets_hub import get_dataset; get_dataset('synthetic_time_series', data_dir='./data', download=True); print('Synthetic Time Series OK')"

echo "Loading EEG..."
python3 -c "from datasets_hub import get_dataset; get_dataset('eeg_data', data_dir='./data', download=True); print('EEG OK')"

echo "Loading EEG Eye State..."
python3 -c "from datasets_hub import get_dataset; get_dataset('eeg_eye_state', data_dir='./data', download=True); print('EEG Eye State OK')"

echo "Downloading PhysioNet EEG Motor Imagery (109 subjects, runs 4/8/12)..."
python3 -c "from datasets_hub import get_dataset; train, test, info = get_dataset('physionet_eeg_mi', data_dir='./data', download=True); print(f'PhysioNet EEGMI OK: {len(train)} Train, {len(info.validation_dataset)} Val, {len(test)} Test')"

echo "Loading ECG..."
python3 -c "from datasets_hub import get_dataset; get_dataset('ecg_data', data_dir='./data', download=True); print('ECG OK')"

echo ""
echo "=== Dataset preparation complete. Data stored in $HOME/code/eval_snn_ann/data/ ==="
echo "Submit the SLURM job with: sbatch dachs_train.sh"
deactivate
