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
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python3 -m pip install -r $HOME/code/eval_snn_ann/requirements.txt

rsync -avz $HOME/code/eval_snn_ann/data/ $TMPDIR/data/

# Training; Checkpoints werden ins persistente Home-Verzeichnis geschrieben
# (nicht in $TMPDIR, das nach dem Slurm-Job gelöscht wird)
python3 $HOME/code/eval_snn_ann/train_mnist.py \
    --config $HOME/code/eval_snn_ann/config.yaml \
    --save-dir $HOME/code/eval_snn_ann/saved_models \
    --device cuda \
    --skip-evaluation \
    --force-train

# Exporte (model_data.h/cc) landen ebenfalls im Home-Verzeichnis
mkdir -p $HOME/code/eval_snn_ann/main

time deactivate
