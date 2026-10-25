#!/bin/bash
#SBATCH --job-name=dl_mnist_ann
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=48
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=hansert.jonas@gmail.com

module load devel/cuda/12.4

cd $TMPDIR
python3 -m venv dachs_env
. dachs_env/bin/activate
python3 -m pip install --upgrade pip

# PyTorch mit CUDA 12.4-Support (kompatibel zum DACHS CUDA-Modul)
python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

python3 -m pip install -r $HOME/code/eval_snn_ann/requirements.txt

# Datenverzeichnis von $HOME nach $TMPDIR rsyncen.
# WICHTIG: Vorher sollte auf dem Login-Node einmal
#   sbatch dachs_setup.sh (oder manuell python run) ausgeführt werden,
#   damit die Daten in $HOME/code/eval_snn_ann/data/ liegen.
rsync -avz $HOME/code/eval_snn_ann/data/ $TMPDIR/data/

# Training: Konfiguration nutzen, Evaluation überspringen (kein Joulescope/ESP32)
python3 $HOME/code/eval_snn_ann/train_mnist.py \
    --config $HOME/code/eval_snn_ann/config.yaml \
    --save-dir $HOME/code/eval_snn_ann/saved_models \
    --device cuda \
    --skip-evaluation \
    --force-train

time deactivate
