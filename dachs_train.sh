#!/bin/bash
#SBATCH --job-name=dl_mnist_ann
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=48
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=hansert.jonas@gmail.com

set -euo pipefail

module load devel/cuda/12.4

cd "$TMPDIR"
python3 -m venv dachs_env
. dachs_env/bin/activate
python3 -m pip install --upgrade pip

# PyTorch mit CUDA 12.4-Support (kompatibel zum DACHS CUDA-Modul)
python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

python3 -m pip install -r $HOME/code/eval_snn_ann/requirements.txt

# Copy the dataset directory from $HOME to $TMPDIR.
# IMPORTANT: Run dachs_setup.sh on the login node first
# (or run the dataset loader manually) so data is present in $HOME.

rsync -avz "$HOME/code/eval_snn_ann/data/" "$TMPDIR/data/"

# Train from the configuration and skip evaluation (no Joulescope/ESP32 on the cluster).
python3 $HOME/code/eval_snn_ann/train_mnist.py \
    --config "$HOME/code/eval_snn_ann/config.yaml" \
    --save-dir "$HOME/code/eval_snn_ann/saved_models" \
    --device cuda \
    --skip-evaluation

time deactivate
