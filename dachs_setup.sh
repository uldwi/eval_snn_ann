#!/bin/bash
# dachs_setup.sh – Läuft auf dem DACHS **Login-Node** (nicht im Batch!)
# Lädt alle Datensätze herunter, damit sie im persistenten $HOME liegen
# und vom SLURM-Job (Compute-Node) via rsync nach $TMPDIR kopiert werden können.

set -e

echo "=== DACHS Setup: Lade Datensätze herunter (Login-Node) ==="

# Ins Projektverzeichnis wechseln
cd $HOME/code/eval_snn_ann

# Python venv auf Login-Node (optional, aber empfohlen für Konsistenz)
if [ ! -d "dachs_env" ]; then
    python3 -m venv dachs_env
fi
source dachs_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -r requirements.txt

# Alle Datensätze einmal laden (triggert Download, wenn nicht vorhanden)
echo "Lade MNIST..."
python3 -c "from datasets_hub import get_dataset; get_dataset('mnist', data_dir='./data', download=True); print('MNIST OK')"

echo "Lade Fashion-MNIST..."
python3 -c "from datasets_hub import get_dataset; get_dataset('fashion_mnist', data_dir='./data', download=True); print('Fashion-MNIST OK')"

echo "Lade CIFAR-10..."
python3 -c "from datasets_hub import get_dataset; get_dataset('cifar10', data_dir='./data', download=True); print('CIFAR-10 OK')"

echo "Lade Speech Commands (KWS)..."
python3 -c "from datasets_hub import get_dataset; get_dataset('speech_commands', data_dir='./data', download=True); print('Speech Commands OK')"

echo "Lade HAR Time Series..."
python3 -c "from datasets_hub import get_dataset; get_dataset('har_time_series', data_dir='./data', download=True); print('HAR OK')"

echo "Lade Synthetic Time Series..."
python3 -c "from datasets_hub import get_dataset; get_dataset('synthetic_time_series', data_dir='./data', download=True); print('Synthetic Time Series OK')"

echo "Lade EEG..."
python3 -c "from datasets_hub import get_dataset; get_dataset('eeg_data', data_dir='./data', download=True); print('EEG OK')"

echo "Lade EEG Eye State..."
python3 -c "from datasets_hub import get_dataset; get_dataset('eeg_eye_state', data_dir='./data', download=True); print('EEG Eye State OK')"

echo "Lade ECG..."
python3 -c "from datasets_hub import get_dataset; get_dataset('ecg_data', data_dir='./data', download=True); print('ECG OK')"

echo ""
echo "=== Alle Datensätze heruntergeladen und in $HOME/code/eval_snn_ann/data/ gespeichert ==="
echo "Jetzt SLURM-Job einreichen:  sbatch dachs_train.sh"
deactivate