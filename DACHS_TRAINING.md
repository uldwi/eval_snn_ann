# Training auf DACHS (bwHPC Datenanalyse Cluster)

## Voraussetzungen

- DACHS Zugang (Login) über SSH
- Registriert auf dem bwHPC Cluster

## Aufbau / Setup (zwei Schritte)

### Schritt 1: Daten herunterladen (Login-Node – **hat Internet**)

Da `data/` im `.gitignore` ist, liegen die Datensätze **nicht** im Git-Repo. Auf den Compute-Nodes gibt es **kein Internet**, daher muss der Download **einmalig auf dem Login-Node** erfolgen.

```bash
ssh <username>@login.dachs.bwhpc.de
cd $HOME/code/eval_snn_ann
./dachs_setup.sh
```

Das Skript `dachs_setup.sh`:
- Erstellt eine `dachs_env` Python-Umgebung auf dem Login-Node
- Installiert PyTorch (CUDA 12.4) + Dependencies
- Lädt **alle** Datensätze (MNIST, Fashion-MNIST, CIFAR-10, Speech Commands, HAR, EEG, ECG, Synthetic) in `$HOME/code/eval_snn_ann/data/`
- Nach Abschluss sind die Daten lokal persistenter Speicher verfügbar

### Schritt 2: Training starten (Compute-Node – **kein Internet**)

```bash
sbatch dachs_train.sh
```

Das Skript `dachs_train.sh`:
- Erstellt auf dem Compute-Node (`$TMPDIR`) eine neue `dachs_env` Umgebung
- Installiert PyTorch (CUDA 12.4) + Dependencies
- Rsync't die **bereits heruntergeladenen** Daten aus `$HOME/code/eval_snn_ann/data/` nach `$TMPDIR/data/`
- Startet das Training mit `--device cuda --skip-evaluation --force-train` und `config.yaml`
- Checkpoints (`.pt`, `.json`) landen in `$HOME/code/eval_snn_ann/saved_models/`

## Alternativ: Manuell (Batch Job)

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

## Wichtige Hinweise

- **Nicht vergessen:** `--gres=gpu:1` muss im Batch Job gesetzt sein, sonst wird kein
  GPU zugewiesen und Training läuft auf CPU.
- Das CUDA Modul muss vor dem Start der Python-Prozesse geladen sein
  (`module load devel/cuda/12.4`).
- **Schritt 1 (`dachs_setup.sh`) ist Pflicht**, damit CIFAR-10 & Co. auf den Compute-Nodes
  verfügbar sind. Ohne Internetzugang dort schlägt der Download fehl.
- Das Skript nutzt `$TMPDIR` als Scratch-Verzeichnis. Nach Job-Ende wird es gelöscht –
  nur `$HOME/code/eval_snn_ann/saved_models/` bleibt persistent.
- `train_mnist.py` CLI: `--device cuda` (GPU), `--skip-evaluation` (kein ESP32/Joulescope),
  `--config config.yaml` (alle Modell-Einstellungen aus YAML).

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