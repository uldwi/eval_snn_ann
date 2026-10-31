#!/bin/bash
#SBATCH --job-name=dl_mnist_ann
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=48
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=hansert.jonas@gmail.com

set -euo pipefail

PROJECT_DIR="$HOME/code/eval_snn_ann"
PHYSIONET_SOURCE="$PROJECT_DIR/eval_snn_ann/data/physionet.py"
TRAIN_SCRIPT="$PROJECT_DIR/dachs_train.sh"

if ! grep -Eq '^[[:space:]]*rank="full",[[:space:]]*(#.*)?$' "$PHYSIONET_SOURCE"; then
    echo "ERROR: DACHS checkout is missing the PhysioNet CSP rank fix."
    echo "Sync the updated eval_snn_ann/data/physionet.py from your working copy, then resubmit."
    exit 1
fi

if grep -Eq '^[[:space:]]+--force-train([[:space:]\\]|$)' "$TRAIN_SCRIPT"; then
    echo "ERROR: dachs_train.sh still forces all models to retrain."
    echo "Sync the updated dachs_train.sh from your working copy, then resubmit."
    exit 1
fi

module load devel/cuda/12.4

cd "$TMPDIR"
python3 -m venv dachs_env
. dachs_env/bin/activate
python3 -m pip install --upgrade pip

# PyTorch mit CUDA 12.4-Support (kompatibel zum DACHS CUDA-Modul)
python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

python3 -m pip install -r "$PROJECT_DIR/requirements.txt"

# Copy the dataset directory from $HOME to $TMPDIR.
# IMPORTANT: Run dachs_setup.sh on the login node first
# (or run the dataset loader manually) so data is present in $HOME.

rsync -avz "$PROJECT_DIR/data/" "$TMPDIR/data/"

# Train from the configuration and skip evaluation (no Joulescope/ESP32 on the cluster).
python3 "$PROJECT_DIR/train_mnist.py" \
    --config "$PROJECT_DIR/config.yaml" \
    --save-dir "$PROJECT_DIR/saved_models" \
    --device cuda \
    --skip-evaluation

time deactivate
