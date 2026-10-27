# Training on DACHS

This guide follows `dachs_setup.sh` and `dachs_train.sh`: prepare datasets on a login node, then run training as a SLURM batch job on a compute node.

## Prerequisites

- A DACHS/bwHPC account and access to the project directory
- A checkout at `$HOME/code/eval_snn_ann` (or updated paths in the scripts)
- Enough storage for datasets in `data/`, the Python environment, and model checkpoints
- Access to an NVIDIA GPU partition for the training job

## 1. Prepare datasets on a login node

Connect to a login node and run the setup script in an interactive shell:

```bash
ssh <username>@<dachs-login-host>
cd "$HOME/code/eval_snn_ann"
./dachs_setup.sh
```

The script creates `dachs_env`, installs the CUDA 12.4 PyTorch/TorchVision/TorchAudio builds and packages from `requirements.txt`, then downloads the datasets listed in `dachs_setup.sh` to `data/`. This may take a while, especially for PhysioNet EEGMI.

Run setup directly on the login node; do not submit it with `sbatch`. The training job expects dataset files to be present because compute-node downloads cannot be assumed to work.

If the cluster does not allow package installation or downloads from login nodes, use an approved environment or transfer workflow that follows DACHS policies. `dachs_setup.sh` currently pins CUDA 12.4 wheels and is tailored to the cluster's Python/CUDA setup.

## 2. Submit the SLURM training job

After setup and downloads finish:

```bash
cd "$HOME/code/eval_snn_ann"
sbatch dachs_train.sh
```

The batch script loads CUDA 12.4, creates an environment under `$TMPDIR`, installs PyTorch and project dependencies, copies `data/` into the temporary job directory, and runs:

```bash
python train_mnist.py \
  --config "$HOME/code/eval_snn_ann/config.yaml" \
  --save-dir "$HOME/code/eval_snn_ann/saved_models" \
  --device cuda \
  --skip-evaluation \
  --force-train
```

`--skip-evaluation` skips ESP32/Joulescope evaluation. `--force-train` retrains even when matching checkpoints exist. Checkpoints are saved under `saved_models/` in persistent home storage; temporary files under `$TMPDIR` are removed after the job ends.

## Adjust SLURM resources

The current resources are set by the `#SBATCH` directives at the top of `dachs_train.sh`. Update the partition, GPU count, and runtime to match the resources currently available to your account. Partition names and GPU types can change, so check the current DACHS documentation instead of relying on old examples.

To enable email notifications, set `--mail-type` and `--mail-user` in `dachs_train.sh` to your preferred values.

## Configuration and targeted runs

`config.yaml` defines which models and datasets are processed. Models are run in the order listed under `models`. Edit the configuration before submitting the job.

For a targeted run, adjust the command in `dachs_train.sh`, for example:

```bash
python train_mnist.py \
  --config "$HOME/code/eval_snn_ann/config.yaml" \
  --dataset speech_commands \
  --epochs 80 \
  --device cuda \
  --skip-evaluation
```

List dataset names and command-line options with:

```bash
python train_mnist.py --list-datasets
python train_mnist.py --help
```

## Troubleshooting

- **Dataset missing in the job:** Check that `dachs_setup.sh` completed successfully on the login node and that `data/` is in the expected project directory.
- **CUDA/PyTorch error:** Confirm that the PyTorch wheels, loaded CUDA module, and GPU driver are compatible. The selected versions are listed in `dachs_setup.sh` and `dachs_train.sh`.
- **Download timeout:** Retry setup; many loaders use local caches. Also check available storage and network connectivity.
- **Job terminated:** Check the SLURM job status, standard output, and standard error. For a trial run, reduce the model list or epoch count in a separate configuration.
- **Checkpoints missing after the job:** Ensure `--save-dir` points to `$HOME`, not `$TMPDIR`.

See [README.md](README.md) for general installation and usage instructions.
