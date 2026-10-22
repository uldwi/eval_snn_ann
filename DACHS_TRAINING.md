# Training auf DACHS (bwHPC Datenanalyse Cluster)

## Voraussetzungen

- DACHS Zugang (Login) über SSH
- Registriert auf dem bwHPC Cluster

## Aufbau / Setup

1. **Anmelden:**

```bash
ssh <username>@login.dachs.bwhpc.de
```

2. **CUDA-Modul laden (für NVIDIA GPU Teil):**

```bash
module load devel/cuda/12.4
```

3. **Virtuelle Umgebung mit PyTorch (CUDA 12.4) erstellen:**

```bash
python3 -m venv dachs_env
source dachs_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -r /home/<username>/code/eval_snn_ann/requirements.txt
```

## GPU Trainingslauf einreichen

Das bereitgestellte Skript `dachs_train.sh` nutzt die Konfigurationsdatei `config.yaml`:

```bash
sbatch dachs_train.sh
```

Alternativ manuell (Batch Job):

```bash
sbatch --job-name=dl_mnist_ann --partition=gpu1 --gres=gpu:1 \
       --ntasks-per-gpu=48 --time=24:00:00 \
       --mail-type=all --mail-user=hansert.jonas@gmail.com \
       dachs_train.sh
```

## Wichtige Parameter

| Parameter | Wert | Beschreibung |
|---|---|---|
| Partition | `gpu1` | NVIDIA L40S (45 Nodes, 48 GB VRAM) |
| Partition | `gpu8` | NVIDIA H100 (8 GPUs) |
| Partition | `gpu4` | AMD MI300A APU (ROCm) |
| `--gres=gpu:1` | 1 GPU | Anzahl der GPUs |
| `--time` | `24:00:00` | Maximaler Walltime |
| `--mail-user` | `hansert.jonas@gmail.com` | Benachrichtigung |

## GPU Batch-Skript

- `dachs_train.sh` – L40S GPU, PyTorch CUDA 12.4, `config.yaml`, `--device cuda`, `--skip-evaluation`

## Wichtige Hinweise

- **Nicht vergessen:** `--gres=gpu:1` muss im Batch Job gesetzt sein, sonst wird kein
  GPU zugewiesen und Training läuft auf CPU.
- Das CUDA Modul muss vor dem Start der Python-Prozesse geladen sein
  (`module load devel/cuda/12.4`).
- Das Skript nutzt `$TMPDIR` als Scratch-Verzeichnis und führt nach dem Training
  den Skript-Ende aus, um Ressourcen freizugeben.
- Die Konfiguration wird aus `config.yaml` gelesen (`--config`). Keine CLI-Überschreibungen
  nötig außer `--device cuda` und `--skip-evaluation`.

## GPU Trainingsaufruf (manuell)

```bash
python3 train_mnist.py --config config.yaml --device cuda --skip-evaluation --force-train
```

Für eine andere GPU-Partition (z. B. H100) im Skript ändern:

```bash
#SBATCH --partition=gpu8
#SBATCH --gres=gpu:1
```

## Hinweis zur GPU-Auslastung

DACHS bietet folgende GPU-Partitionen:

- `gpu1`: NVIDIA L40S – 45 Nodes, 1x GPU pro Node
- `gpu8`: NVIDIA H100 – 8x GPU auf einem Node
- `gpu4`: AMD MI300A – 1 Node, 4x AMD MI300A APU

Für PyTorch/CUDA (NVIDIA) sind `gpu1` und `gpu8` relevant. Für AMD sind
ROCm/HIP-Einstellungen nötig.
