#!/bin/bash
#SBATCH --job-name=dl_mnist_ann
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=48
#SBATCH --time=24:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=jonas@hs-esslingen.de

module load devel/cuda/12.4

cd $TMPDIR
python3 -m venv dachs_env
. dachs_env/bin/activate
python3 -m pip install --upgrade pip

# PyTorch mit CUDA 12.4-Support (kompatibel zum DACHS CUDA-Modul)
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python3 -m pip install -r $HOME/code/eval_snn_ann/requirements.txt

rsync -avz $HOME/code/eval_snn_ann/data/ $TMPDIR/data/

python3 $HOME/code/eval_snn_ann/train_mnist.py \
    --device cuda \
    --dataset mnist \
    --epochs 5 \
    --batch-size 64 \
    --force-train

time deactivate
