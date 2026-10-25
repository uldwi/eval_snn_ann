#!/usr/bin/env python3
"""
Datensatz-Hub für neuronale Netze (ANN & SNN) auf eingebetteten Systemen (ESP32).

Unterstützte Datensätze und Domänen:
1. Computer Vision / Bildverarbeitung:
   - "mnist": Handgeschriebene Ziffern (0-9), 1x28x28 = 784 Features, 10 Klassen
   - "fashion_mnist": Kleidungsstücke (Zalando), 1x28x28 = 784 Features, 10 Klassen
   - "cifar10": Objektkategorien (herunterskaliert auf 1x16x16=256 oder 1x32x32=1024), 10 Klassen

2. Audio / Keyword Recognition (Keyword Spotting - KWS):
   - "speech_commands" / "kws": Echte Sprachaufnahmen aus Speech Commands v0.02,
     alle 35 gesprochenen Wörter, als 1x16x16 Mel-Spektrogramm mit 35 Klassen

3. Sensordaten / Time Series (Inertialmessdaten / Human Activity Recognition):
   - "har_time_series" / "uci_har": Beschleunigungs- & Gyroskopsensoren (6 Achsen x 20 Zeitschritte = 120 Features),
     6 Klassen: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying
   - "synthetic_time_series": Musterklassifikation von Zeitreihen (Sinus, Stufenimpuls, Trend, Chirp), 1x128 Features, 4 Klassen

4. Biomedizinische Signale / Gehirnwellen (EEG):
   - "eeg_data" / "eeg_bands": Elektroenzephalogramm (Delta, Theta, Alpha, Beta Gehirnwellen),
     8 Elektroden x 16 Frequenz/Zeitbins = 128 Features, 4 Klassen (Deep Sleep, Drowsy, Relaxed, Cognitive Focus)
   - "eeg_eye_state": EEG-Zustandserkennung (Augen geöffnet vs. geschlossen), 1x64 Features, 2 Klassen

5. Biomedizinische Signale / Kardiologie (ECG / EKG):
   - "ecg_data" / "ecg_heartbeat": EKG-Herzschlag-Klassifikation (MIT-BIH Arrhythmie Standard),
     128 Zeitschritte = 128 Features, 5 Klassen (Normal, Supraventrikulär, Ventrikulär, Fusion, Unbekannt)

6. Reale EEG-Motor-Imagery:
   - "physionet_eeg_mi" / "eeg_mi": PhysioNet EEGMMIDB, vorgestellte linke vs. rechte Faustbewegung
     aus den offiziellen Läufen 4, 8 und 12, 64x16 spektrale Merkmale, 2 Klassen
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import welch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms


@dataclass
class DatasetInfo:
    """Metadaten und Spezifikationen eines Datensatzes."""
    name: str
    domain: str
    description: str
    input_size: int
    in_channels: int
    in_height: int
    in_width: int
    num_classes: int
    class_names: List[str] = field(default_factory=list)
    validation_dataset: Optional[Dataset] = field(default=None, repr=False)


AVAILABLE_DATASETS: Dict[str, Dict[str, Union[str, int, List[str]]]] = {
    # 1. Bildverarbeitung
    "mnist": {
        "domain": "Computer Vision (Bildverarbeitung)",
        "description": "Klassischer MNIST Datensatz handgeschriebener Ziffern (0-9)",
        "input_size": 784,
        "in_channels": 1,
        "in_height": 28,
        "in_width": 28,
        "num_classes": 10,
        "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    },
    "fashion_mnist": {
        "domain": "Computer Vision (Bildverarbeitung)",
        "description": "Fashion-MNIST Datensatz für Kleidungsstücke von Zalando",
        "input_size": 784,
        "in_channels": 1,
        "in_height": 28,
        "in_width": 28,
        "num_classes": 10,
        "class_names": ["T-Shirt", "Hose", "Pullover", "Kleid", "Mantel", "Sandale", "Hemd", "Sneaker", "Tasche", "Stiefelette"],
    },
    "cifar10": {
        "domain": "Computer Vision (Bildverarbeitung)",
        "description": "CIFAR-10 Objektkategorien (Graustufen 16x16 für ressourcenschonende ESP32-Inferenz)",
        "input_size": 256,
        "in_channels": 1,
        "in_height": 16,
        "in_width": 16,
        "num_classes": 10,
        "class_names": ["Flugzeug", "Auto", "Vogel", "Katze", "Hirsch", "Hund", "Frosch", "Pferd", "Schiff", "LKW"],
    },

    # 2. Audio & Keyword Recognition (Sprachbefehle)
    "speech_commands": {
        "domain": "Audio & Keyword Spotting (Sprachbefehlserkennung)",
        "description": "Echte Sprachaufnahmen aus Speech Commands v0.02, alle 35 gesprochenen Wörter als Mel-Spektrogramm",
        "input_size": 256,
        "in_channels": 1,
        "in_height": 16,
        "in_width": 16,
        "num_classes": 35,
        "class_names": ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"],
    },
    "kws": {
        "domain": "Audio & Keyword Spotting (Sprachbefehlserkennung)",
        "description": "Alias für Speech Commands v0.02 mit allen 35 gesprochenen Wörtern",
        "input_size": 256,
        "in_channels": 1,
        "in_height": 16,
        "in_width": 16,
        "num_classes": 35,
        "class_names": ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"],
    },

    # 3. Sensordaten / Zeitreihen (Human Activity Recognition & Time Series)
    "har_time_series": {
        "domain": "Sensordaten & Zeitreihen (Human Activity Recognition)",
        "description": "6-Kanal IMU-Sensordaten (Beschleunigung & Gyroskop) zur Erkennung menschlicher Aktivitäten",
        "input_size": 120,
        "in_channels": 1,
        "in_height": 6,
        "in_width": 20,
        "num_classes": 6,
        "class_names": ["Gehen", "Treppe auf", "Treppe ab", "Sitzen", "Stehen", "Liegen"],
    },
    "uci_har": {
        "domain": "Sensordaten & Zeitreihen (Human Activity Recognition)",
        "description": "Alias für har_time_series (6 Aktivitäten, 6x20 IMU Features)",
        "input_size": 120,
        "in_channels": 1,
        "in_height": 6,
        "in_width": 20,
        "num_classes": 6,
        "class_names": ["Gehen", "Treppe auf", "Treppe ab", "Sitzen", "Stehen", "Liegen"],
    },
    "synthetic_time_series": {
        "domain": "Sensordaten & Zeitreihen (Mustererkennung)",
        "description": "Synthetische Zeitreihen mit 4 Signalmustern (Sinus-Oszillation, Impuls, Trend, Chirp-Frequenz)",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 1,
        "in_width": 128,
        "num_classes": 4,
        "class_names": ["Sinus-Oszillation", "Impuls/Stufe", "Linearer Trend", "Chirp-Modulation"],
    },

    # 4. Biomedizinische Signale / Gehirnwellen (EEG)
    "eeg_data": {
        "domain": "Biomedizinische Signale (EEG Gehirnwellen)",
        "description": "Elektroenzephalogramm (EEG) Signale aufgeteilt in 4 mentale Zustände (Delta, Theta, Alpha, Beta)",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 8,
        "in_width": 16,
        "num_classes": 4,
        "class_names": ["Tiefschlaf (Delta)", "Schlaefrig (Theta)", "Entspannt (Alpha)", "Konzentriert (Beta)"],
    },
    "eeg_bands": {
        "domain": "Biomedizinische Signale (EEG Gehirnwellen)",
        "description": "Alias für eeg_data (4 mentale Gehirnwellen-Bänder)",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 8,
        "in_width": 16,
        "num_classes": 4,
        "class_names": ["Tiefschlaf (Delta)", "Schlaefrig (Theta)", "Entspannt (Alpha)", "Konzentriert (Beta)"],
    },
    "eeg_eye_state": {
        "domain": "Biomedizinische Signale (EEG Gehirnwellen)",
        "description": "EEG-Zustandserkennung: Augen geöffnet vs. Augen geschlossen",
        "input_size": 64,
        "in_channels": 1,
        "in_height": 1,
        "in_width": 64,
        "num_classes": 2,
        "class_names": ["Augen geschlossen", "Augen geoeffnet"],
    },
    "physionet_eeg_mi": {
        "domain": "Biomedizinische Signale (EEG Motor Imagery)",
        "description": "Echte PhysioNet EEGMMIDB-Aufnahmen: vorgestellte linke vs. rechte Faustbewegung, personengetrennte Splits",
        "input_size": 1024,
        "in_channels": 1,
        "in_height": 64,
        "in_width": 16,
        "num_classes": 2,
        "class_names": ["Linke Faust (Motor Imagery)", "Rechte Faust (Motor Imagery)"],
    },

    # 5. Biomedizinische Signale / Kardiologie (ECG / EKG)
    "ecg_data": {
        "domain": "Biomedizinische Signale (EKG Herzrhythmus)",
        "description": "Elektrokardiogramm (EKG) Herzschlag-Klassifikation nach MIT-BIH Arrhythmie Standard (5 Klassen)",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 1,
        "in_width": 128,
        "num_classes": 5,
        "class_names": ["Normaler Schlag (N)", "Supraventrikulaer (S)", "Ventrikulaer (V)", "Fusionsschlag (F)", "Unbekannt/Schrittmacher (Q)"],
    },
    "ecg_heartbeat": {
        "domain": "Biomedizinische Signale (EKG Herzrhythmus)",
        "description": "Alias für ecg_data (5 Herzrhythmus-Klassen)",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 1,
        "in_width": 128,
        "num_classes": 5,
        "class_names": ["Normaler Schlag (N)", "Supraventrikulaer (S)", "Ventrikulaer (V)", "Fusionsschlag (F)", "Unbekannt/Schrittmacher (Q)"],
    },
}


DATASET_ALIASES: Dict[str, str] = {
    "kws": "speech_commands",
    "uci_har": "har_time_series",
    "eeg_bands": "eeg_data",
    "eeg_mi": "physionet_eeg_mi",
    "ecg_heartbeat": "ecg_data",
}


def normalize_dataset_name(name: str) -> str:
    """Normalisiert Alias-Namen auf den kanonischen Datensatznamen."""
    key = str(name).strip().lower()
    return DATASET_ALIASES.get(key, key)


def list_available_datasets() -> List[DatasetInfo]:
    """Gibt eine Liste aller verfügbaren Datensätze und deren Metadaten zurück."""
    unique_names = ["mnist", "fashion_mnist", "cifar10", "speech_commands", "har_time_series", "synthetic_time_series", "eeg_data", "eeg_eye_state", "physionet_eeg_mi", "ecg_data"]
    results = []
    for name in unique_names:
        if name in AVAILABLE_DATASETS:
            cfg = AVAILABLE_DATASETS[name]
            results.append(DatasetInfo(
                name=name,
                domain=str(cfg["domain"]),
                description=str(cfg["description"]),
                input_size=int(cfg["input_size"]),
                in_channels=int(cfg["in_channels"]),
                in_height=int(cfg["in_height"]),
                in_width=int(cfg["in_width"]),
                num_classes=int(cfg["num_classes"]),
                class_names=list(cfg["class_names"]),
            ))
    return results


def print_available_datasets_table():
    """Gibt eine übersichtliche Tabelle aller Datensätze auf der Konsole aus."""
    datasets_list = list_available_datasets()
    print("\n" + "=" * 90)
    print("VERFÜGBARE DATENSÄTZE NACH ANWENDUNGSBEREICHEN (DOMÄNEN)")
    print("=" * 90)
    print(f"{'Name':<22} | {'Domäne':<32} | {'Eingangsform':<14} | {'Klassen':<8}")
    print("-" * 90)
    for ds in datasets_list:
        shape_str = f"{ds.in_channels}x{ds.in_height}x{ds.in_width} ({ds.input_size})"
        print(f"{ds.name:<22} | {ds.domain:<32} | {shape_str:<14} | {ds.num_classes:<8}")
    print("=" * 90 + "\n")


# ---------------------------------------------------------------------------
# Deterministische Generatoren für synthetische Sensor- und Biomedizin-Datensätze
# ---------------------------------------------------------------------------

SPEECH_COMMANDS = (
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
)


class SpeechCommandsDataset(Dataset):
    """Alle 35 Speech-Commands-Wörter als kompakte Mel-Spektrogramme.

    Nichtsprachliche Hintergrundgeräusche sind keine Wortklasse und werden
    ausgelassen.
    """

    def __init__(self, source: Dataset, torchaudio_module):
        self.source = source
        self.keyword_to_index = {word: i for i, word in enumerate(SPEECH_COMMANDS)}
        self.samples = []
        for i in range(len(source)):
            relative_path, sample_rate, label, _, _ = source.get_metadata(i)
            if label in self.keyword_to_index:
                # get_metadata() returns paths relative to the archive root (_archive),
                # while _path already includes the speech_commands_v0.02 subdirectory.
                self.samples.append((os.path.join(source._archive, relative_path), sample_rate, label))

        self.mel_transform = torchaudio_module.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=40,
            f_min=20.0,
            f_max=7600.0,
            power=2.0,
        )
        self.feature_cache = [None] * len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cached = self.feature_cache[index]
        if cached is not None:
            return cached

        wav_path, sample_rate, label = self.samples[index]
        sample_rate, audio = wavfile.read(wav_path)
        if sample_rate != 16000:
            raise ValueError(f"Speech Commands erwartet 16 kHz, erhalten: {sample_rate} Hz")

        if np.issubdtype(audio.dtype, np.integer):
            dtype_info = np.iinfo(audio.dtype)
            if dtype_info.min < 0:
                audio = audio.astype(np.float32) / float(max(abs(dtype_info.min), dtype_info.max + 1))
            else:
                midpoint = (dtype_info.max + 1) / 2.0
                audio = (audio.astype(np.float32) - midpoint) / midpoint
        else:
            audio = audio.astype(np.float32)
        waveform = torch.from_numpy(audio.copy())
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)
        waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.size(1) < sample_rate:
            waveform = torch.nn.functional.pad(waveform, (0, sample_rate - waveform.size(1)))
        else:
            waveform = waveform[:, :sample_rate]

        mel = self.mel_transform(waveform)
        mel_db = 10.0 * torch.log10(mel.clamp_min(1e-10))
        mel_db = (mel_db - mel_db.max()).clamp(min=-80.0, max=0.0)
        mel = (mel_db + 80.0) / 80.0
        features = torch.nn.functional.adaptive_avg_pool2d(mel.unsqueeze(0), (16, 16)).squeeze(0)
        result = (features.to(torch.float32), self.keyword_to_index[label])
        self.feature_cache[index] = result
        return result


def _generate_synthetic_har(
    num_train: int = 4500,
    num_test: int = 900,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Erzeugt 6-Kanal-IMU-Sensordaten (3x Beschleunigung, 3x Gyroskop, je 20 Abtastpunkte -> 6x20=120 Features)
    für 6 Aktivitätsklassen (Gehen, Treppe auf, Treppe ab, Sitzen, Stehen, Liegen).
    """
    rng = np.random.RandomState(seed)
    num_classes = 6
    n_channels, n_steps = 6, 20

    def make_samples(n_samples: int):
        x_data = np.zeros((n_samples, 1, n_channels, n_steps), dtype=np.float32)
        y_data = np.zeros((n_samples,), dtype=np.int64)

        for i in range(n_samples):
            cls = i % num_classes
            y_data[i] = cls
            t = np.linspace(0, 2 * np.pi, n_steps)
            imu = np.zeros((n_channels, n_steps), dtype=np.float32)

            # Phasen- und Frequenzvariation
            phase = rng.uniform(0, 2 * np.pi)
            freq_var = rng.uniform(0.9, 1.1)

            if cls == 0:  # Gehen: rhythmische Schwingung in X, Z, Gyro Y
                imu[0] = 0.5 + 0.3 * np.sin(2 * t * freq_var + phase) # Acc X
                imu[1] = 0.1 * np.cos(t * freq_var + phase)           # Acc Y
                imu[2] = 0.9 + 0.4 * np.sin(2 * t * freq_var + phase) # Acc Z
                imu[3] = 0.2 * np.sin(t * freq_var)                   # Gyro X
                imu[4] = 0.6 * np.cos(2 * t * freq_var + phase)       # Gyro Y
                imu[5] = 0.1 * np.sin(t * freq_var)                   # Gyro Z
            elif cls == 1:  # Treppe auf: stärkere vertikale Impulse
                imu[0] = 0.4 + 0.4 * np.sin(2.5 * t * freq_var + phase)
                imu[1] = 0.2 * np.sin(t + phase)
                imu[2] = 1.2 + 0.6 * np.maximum(0, np.sin(2.5 * t * freq_var + phase))
                imu[3] = 0.3 * np.cos(2.5 * t)
                imu[4] = 0.5 * np.sin(2.5 * t * freq_var + phase)
                imu[5] = 0.2 * np.cos(t)
            elif cls == 2:  # Treppe ab: negative Stoßwellen & höhere Frequenz
                imu[0] = 0.6 + 0.5 * np.sin(3.0 * t * freq_var + phase)
                imu[1] = 0.25 * np.cos(1.5 * t)
                imu[2] = 0.7 - 0.5 * np.maximum(0, np.sin(3.0 * t * freq_var + phase))
                imu[3] = 0.4 * np.sin(3.0 * t)
                imu[4] = 0.7 * np.cos(3.0 * t * freq_var + phase)
                imu[5] = 0.3 * np.sin(2.0 * t)
            elif cls == 3:  # Sitzen: statische Lage (Schwerkraft auf Z/X), kaum Bewegung
                imu[0] = 0.2 + rng.normal(0, 0.02, n_steps)
                imu[1] = 0.0 + rng.normal(0, 0.02, n_steps)
                imu[2] = 0.95 + rng.normal(0, 0.02, n_steps)
                imu[3:] = rng.normal(0, 0.01, size=(3, n_steps))
            elif cls == 4:  # Stehen: vertikale Schwerkraft, leichtes Körperschwanken
                imu[0] = 0.05 + rng.normal(0, 0.03, n_steps)
                imu[1] = 0.02 + rng.normal(0, 0.03, n_steps)
                imu[2] = 1.0 + rng.normal(0, 0.03, n_steps)
                imu[3:] = rng.normal(0, 0.02, size=(3, n_steps))
            elif cls == 5:  # Liegen: Schwerkraft auf X oder Y statt Z
                imu[0] = 0.95 + rng.normal(0, 0.02, n_steps)
                imu[1] = 0.1 + rng.normal(0, 0.02, n_steps)
                imu[2] = 0.05 + rng.normal(0, 0.02, n_steps)
                imu[3:] = rng.normal(0, 0.01, size=(3, n_steps))

            # Rauschen und Skalierung
            noise = rng.normal(0, 0.03, size=(n_channels, n_steps))
            sample = np.clip((imu + noise + 2.0) / 4.0, 0.0, 1.0)
            x_data[i, 0] = sample.astype(np.float32)

        return x_data, y_data

    x_tr, y_tr = make_samples(num_train)
    x_te, y_te = make_samples(num_test)
    return TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)), TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))


def _generate_synthetic_eeg(
    num_train: int = 4000,
    num_test: int = 800,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Erzeugt 8-Kanal EEG-Gehirnwellensignale mit spektraler Zerlegung
    in 4 mentale Zustände (Delta: Tiefschlaf, Theta: Schläfrig, Alpha: Entspannt, Beta: Konzentriert).
    Form: 1x8x16 = 128 Features.
    """
    rng = np.random.RandomState(seed)
    num_classes = 4
    n_channels, n_bins = 8, 16

    # Dominante Frequenzbänder pro Zustand:
    # 0: Delta (0.5 - 4 Hz) -> niedrige Bins (0..3)
    # 1: Theta (4 - 8 Hz)   -> untere Bins (3..6)
    # 2: Alpha (8 - 13 Hz)  -> mittlere Bins (6..10)
    # 3: Beta  (13 - 30 Hz) -> hohe Bins (10..15)

    def make_samples(n_samples: int):
        x_data = np.zeros((n_samples, 1, n_channels, n_bins), dtype=np.float32)
        y_data = np.zeros((n_samples,), dtype=np.int64)

        for i in range(n_samples):
            cls = i % num_classes
            y_data[i] = cls
            eeg = np.zeros((n_channels, n_bins), dtype=np.float32)

            center_bin = [1.5, 4.5, 8.0, 12.5][cls]
            bandwidth = [1.2, 1.5, 1.8, 2.5][cls]

            for ch in range(n_channels):
                # Kanal-spezifische Variation
                ch_offset = rng.uniform(-0.5, 0.5)
                amp = rng.uniform(0.7, 1.0)
                for b in range(n_bins):
                    dist = (b - (center_bin + ch_offset)) ** 2
                    power = amp * np.exp(-dist / (2.0 * bandwidth ** 2))
                    # 1/f Hintergrundrauschen des Gehirns
                    bg_noise = 0.2 / (1.0 + 0.2 * b) + rng.uniform(0.0, 0.1)
                    eeg[ch, b] = power + bg_noise

            eeg = np.clip(eeg / eeg.max(), 0.0, 1.0)
            x_data[i, 0] = eeg.astype(np.float32)

        return x_data, y_data

    x_tr, y_tr = make_samples(num_train)
    x_te, y_te = make_samples(num_test)
    return TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)), TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))


def _generate_synthetic_ecg(
    num_train: int = 4000,
    num_test: int = 800,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Erzeugt 1-Kanal EKG-Herzschlagkurven (128 Abtastpunkte) nach dem MIT-BIH Arrhythmie-Standard
    für 5 Klassen:
      0: Normaler Herzschlag (N)
      1: Supraventrikuläre Extrasystole (S)
      2: Ventrikuläre Extrasystole (V)
      3: Fusionsschlag (F)
      4: Unbekannter / Schrittmacher-Schlag (Q)
    """
    rng = np.random.RandomState(seed)
    num_classes = 5
    n_samples_len = 128

    def make_samples(n_samples: int):
        x_data = np.zeros((n_samples, 1, 1, n_samples_len), dtype=np.float32)
        y_data = np.zeros((n_samples,), dtype=np.int64)

        for i in range(n_samples):
            cls = i % num_classes
            y_data[i] = cls
            t = np.linspace(0, 1, n_samples_len)
            signal = np.zeros(n_samples_len, dtype=np.float32)

            # Grundlinie
            baseline = rng.normal(0.0, 0.02, n_samples_len)

            if cls == 0:  # Normaler Herzschlag: P-Welle (0.2), QRS-Komplex (0.5), T-Welle (0.75)
                p_wave = 0.15 * np.exp(-((t - 0.22) ** 2) / 0.001)
                q_wave = -0.15 * np.exp(-((t - 0.46) ** 2) / 0.0002)
                r_wave = 1.00 * np.exp(-((t - 0.50) ** 2) / 0.0003)
                s_wave = -0.25 * np.exp(-((t - 0.54) ** 2) / 0.0002)
                t_wave = 0.35 * np.exp(-((t - 0.72) ** 2) / 0.003)
                signal = p_wave + q_wave + r_wave + s_wave + t_wave
            elif cls == 1:  # Supraventrikulär: deformierte/verfrühte P-Welle, normaler QRS
                p_wave = 0.25 * np.exp(-((t - 0.16) ** 2) / 0.0005)
                q_wave = -0.12 * np.exp(-((t - 0.44) ** 2) / 0.0002)
                r_wave = 0.85 * np.exp(-((t - 0.48) ** 2) / 0.0003)
                s_wave = -0.20 * np.exp(-((t - 0.52) ** 2) / 0.0002)
                t_wave = 0.30 * np.exp(-((t - 0.68) ** 2) / 0.003)
                signal = p_wave + q_wave + r_wave + s_wave + t_wave
            elif cls == 2:  # Ventrikulär (PVC): breiter, invertierter/monströser QRS-Komplex, keine P-Welle
                qrs_wide = -0.9 * np.exp(-((t - 0.48) ** 2) / 0.002) + 0.3 * np.exp(-((t - 0.38) ** 2) / 0.0015)
                t_inv = 0.5 * np.exp(-((t - 0.65) ** 2) / 0.004)
                signal = qrs_wide + t_inv
            elif cls == 3:  # Fusionsschlag: Mischform aus Normal und Ventrikulär
                p_wave = 0.10 * np.exp(-((t - 0.24) ** 2) / 0.001)
                qrs_fusion = 0.65 * np.exp(-((t - 0.48) ** 2) / 0.001) - 0.4 * np.exp(-((t - 0.55) ** 2) / 0.0015)
                t_wave = 0.30 * np.exp(-((t - 0.75) ** 2) / 0.003)
                signal = p_wave + qrs_fusion + t_wave
            elif cls == 4:  # Schrittmacher / Unbekannt: scharfer Stimulations-Spike + verzögerter Ausschlag
                spike = 1.0 * np.exp(-((t - 0.35) ** 2) / 0.00005)
                pacing_wave = 0.7 * np.exp(-((t - 0.45) ** 2) / 0.003) - 0.3 * np.exp(-((t - 0.60) ** 2) / 0.002)
                signal = spike + pacing_wave

            # Normalisierung auf [0, 1]
            raw = signal + baseline
            norm = (raw - (-0.5)) / (1.2 - (-0.5))
            x_data[i, 0, 0] = np.clip(norm, 0.0, 1.0).astype(np.float32)

        return x_data, y_data

    x_tr, y_tr = make_samples(num_train)
    x_te, y_te = make_samples(num_test)
    return TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)), TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))


def _load_physionet_eeg_mi(data_dir: str, seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """Lädt PhysioNet EEGMMIDB mit personengetrenntem Train/Val/Test-Split."""
    try:
        import mne
        from mne.datasets import eegbci
    except ImportError as exc:
        raise ImportError(
            "Der echte PhysioNet EEG-Motor-Imagery-Datensatz benötigt MNE. "
            "Installiere die Projektabhängigkeiten mit `pip install -r requirements.txt`."
        ) from exc

    dataset_root = os.path.join(data_dir, "physionet_eeg_mi")
    raw_root = os.path.join(dataset_root, "raw")
    cache_root = os.path.join(dataset_root, "processed_v1")
    os.makedirs(cache_root, exist_ok=True)

    subject_ids = np.arange(1, 110)
    rng = np.random.RandomState(seed)
    shuffled_subjects = rng.permutation(subject_ids)
    train_subjects = set(shuffled_subjects[:76].tolist())
    validation_subjects = set(shuffled_subjects[76:92].tolist())
    test_subjects = set(shuffled_subjects[92:].tolist())
    split_features = {"train": [], "validation": [], "test": []}
    split_labels = {"train": [], "validation": [], "test": []}

    def subject_split(subject_id: int) -> str:
        if subject_id in train_subjects:
            return "train"
        if subject_id in validation_subjects:
            return "validation"
        return "test"

    for subject_id in subject_ids:
        cache_path = os.path.join(cache_root, f"subject_{subject_id:03d}.npz")
        if os.path.isfile(cache_path):
            with np.load(cache_path) as cache:
                features = cache["features"]
                labels = cache["labels"]
        else:
            try:
                raw_paths = eegbci.load_data(
                    int(subject_id), [4, 8, 12], path=raw_root, update_path=False, verbose="ERROR"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"PhysioNet EEGMMIDB konnte für Person {subject_id} nicht geladen werden. "
                    "Prüfe die Internetverbindung und den freien Speicherplatz."
                ) from exc

            subject_features = []
            subject_labels = []
            for raw_path in raw_paths:
                raw = mne.io.read_raw_edf(raw_path, preload=True, verbose="ERROR")
                eegbci.standardize(raw)
                raw.filter(8.0, 30.0, picks="eeg", method="iir", verbose="ERROR")
                sample_rate = float(raw.info["sfreq"])
                signals = raw.get_data(picks="eeg")
                events, event_codes = mne.events_from_annotations(raw, verbose="ERROR")

                for event_name, class_id in (("T1", 0), ("T2", 1)):
                    if event_name not in event_codes:
                        continue
                    for event in events[events[:, 2] == event_codes[event_name]]:
                        start = int(event[0] - raw.first_samp + round(0.5 * sample_rate))
                        stop = start + int(round(2.0 * sample_rate))
                        if start < 0 or stop > signals.shape[1]:
                            continue
                        segment = signals[:, start:stop]
                        freqs, power = welch(
                            segment,
                            fs=sample_rate,
                            nperseg=min(256, segment.shape[1]),
                            axis=1,
                        )
                        frequency_edges = np.linspace(8.0, 30.0, 17)
                        bands = []
                        for low, high in zip(frequency_edges[:-1], frequency_edges[1:]):
                            mask = (freqs >= low) & (freqs < high)
                            if not np.any(mask):
                                nearest = int(np.argmin(np.abs(freqs - (low + high) / 2.0)))
                                band = power[:, nearest]
                            else:
                                band = power[:, mask].mean(axis=1)
                            bands.append(band)
                        log_power = 10.0 * np.log10(np.maximum(np.stack(bands, axis=1), 1e-30))
                        mean = float(log_power.mean())
                        std = float(log_power.std())
                        normalized = np.clip((log_power - mean) / max(std, 1e-6), -3.0, 3.0)
                        subject_features.append(((normalized + 3.0) / 6.0).astype(np.float32))
                        subject_labels.append(class_id)
                raw.close()

            if not subject_features:
                raise RuntimeError(f"Keine gültigen EEG-Motor-Imagery-Segmente für Person {subject_id} gefunden.")
            features = np.stack(subject_features)
            labels = np.asarray(subject_labels, dtype=np.int64)
            np.savez_compressed(cache_path, features=features, labels=labels)

        split = subject_split(int(subject_id))
        split_features[split].append(features)
        split_labels[split].append(labels)
        if int(subject_id) % 10 == 0:
            print(f"[PhysioNet EEGMI] Vorverarbeitung: {subject_id}/109 Personen")

    datasets_by_split = {}
    for split in ("train", "validation", "test"):
        x = np.concatenate(split_features[split], axis=0)[:, None, :, :]
        y = np.concatenate(split_labels[split], axis=0)
        datasets_by_split[split] = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

    return datasets_by_split["train"], datasets_by_split["validation"], datasets_by_split["test"]


def _generate_synthetic_general_time_series(
    num_train: int = 4000,
    num_test: int = 800,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Erzeugt 4-Klassen Zeitreihendatensatz (Länge 128) mit verschiedenen Dynamiken:
    0: Sinus/Periodisch, 1: Stufen/Impuls, 2: Linearer Trend, 3: Frequenz-modulierter Chirp.
    """
    rng = np.random.RandomState(seed)
    num_classes = 4
    n_len = 128

    def make_samples(n_samples: int):
        x_data = np.zeros((n_samples, 1, 1, n_len), dtype=np.float32)
        y_data = np.zeros((n_samples,), dtype=np.int64)

        for i in range(n_samples):
            cls = i % num_classes
            y_data[i] = cls
            t = np.linspace(0, 1, n_len)
            noise = rng.normal(0, 0.05, n_len)

            if cls == 0:  # Sinus
                freq = rng.uniform(3, 6)
                phase = rng.uniform(0, 2 * np.pi)
                s = 0.5 + 0.4 * np.sin(2 * np.pi * freq * t + phase)
            elif cls == 1:  # Stufenfunktion / Impuls
                step_pos = rng.uniform(0.3, 0.7)
                s = np.where(t < step_pos, 0.2, 0.8)
            elif cls == 2:  # Trend
                slope = rng.choice([-0.7, 0.7])
                base = 0.8 if slope < 0 else 0.2
                s = base + slope * t
            elif cls == 3:  # Chirp
                s = 0.5 + 0.4 * np.sin(2 * np.pi * (1 + 8 * t) * t)

            sig = np.clip(s + noise, 0.0, 1.0)
            x_data[i, 0, 0] = sig.astype(np.float32)

        return x_data, y_data

    x_tr, y_tr = make_samples(num_train)
    x_te, y_te = make_samples(num_test)
    return TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)), TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))


# ---------------------------------------------------------------------------
# Hauptfunktion zum Laden & Bereitstellen von Datensätzen
# ---------------------------------------------------------------------------

def get_dataset(
    name: str = "mnist",
    data_dir: str = "./data",
    download: bool = True,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, DatasetInfo]:
    """
    Lädt einen Datensatz anhand des Namens oder Alias.

    Gibt (train_dataset, test_dataset, dataset_info) zurück.
    """
    key = normalize_dataset_name(name)
    if key not in AVAILABLE_DATASETS:
        supported = ", ".join(f"'{k}'" for k in AVAILABLE_DATASETS.keys())
        raise ValueError(
            f"Unbekannter Datensatz '{name}'! Unterstützte Datensätze sind: {supported}.\n"
            f"Für eine Liste aller Datensätze führe 'python train_mnist.py --list-datasets' aus."
        )

    cfg = AVAILABLE_DATASETS[key]
    info = DatasetInfo(
        name=key,
        domain=str(cfg["domain"]),
        description=str(cfg["description"]),
        input_size=int(cfg["input_size"]),
        in_channels=int(cfg["in_channels"]),
        in_height=int(cfg["in_height"]),
        in_width=int(cfg["in_width"]),
        num_classes=int(cfg["num_classes"]),
        class_names=list(cfg["class_names"]),
    )

    os.makedirs(data_dir, exist_ok=True)

    if key in ("physionet_eeg_mi", "eeg_mi"):
        train_ds, validation_ds, test_ds = _load_physionet_eeg_mi(data_dir=data_dir, seed=seed)
        info.validation_dataset = validation_ds
        return train_ds, test_ds, info

    # Falls Daten lokal bereits vorhanden, Download überspringen (z. B. ohne Internetzugang auf HPC)
    if download:
        if key == "mnist":
            download = not os.path.isfile(os.path.join(data_dir, "MNIST", "raw", "train-images-idx3-ubyte"))
        elif key == "fashion_mnist":
            download = not os.path.isfile(os.path.join(data_dir, "FashionMNIST", "raw", "train-images-idx3-ubyte"))
        elif key == "cifar10":
            download = not os.path.isdir(os.path.join(data_dir, "cifar-10-batches-py")) and not os.path.isdir(os.path.join(data_dir, "cifar10"))

    # 1. Vision: MNIST
    if key == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ])
        train_ds = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
        test_ds = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
        return train_ds, test_ds, info

    # 2. Vision: Fashion-MNIST
    if key == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ])
        train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=download, transform=transform)
        test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=download, transform=transform)
        return train_ds, test_ds, info

    # 3. Vision: CIFAR-10 (herunterskaliert auf 1x16x16 Graustufen für Microcontroller)
    if key == "cifar10":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ])
        train_ds = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=transform)
        test_ds = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=transform)
        return train_ds, test_ds, info

    # 4. Audio: reales Speech Commands v0.02 (Keyword Spotting)
    if key in ("speech_commands", "kws"):
        try:
            import torchaudio
        except (ImportError, OSError) as exc:
            raise ImportError(
                "Speech Commands benötigt ein lauffähiges torchaudio. Installiere torch, torchvision und "
                "torchaudio gemeinsam aus demselben Build. CPU-only: "
                "pip install --force-reinstall torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cpu. "
                "Für CUDA muss das TorchAudio-Paket zur installierten CUDA-Version passen."
            ) from exc

        train_source = torchaudio.datasets.SPEECHCOMMANDS(
            root=data_dir, download=download, subset="training"
        )
        validation_source = torchaudio.datasets.SPEECHCOMMANDS(
            root=data_dir, download=download, subset="validation"
        )
        test_source = torchaudio.datasets.SPEECHCOMMANDS(
            root=data_dir, download=download, subset="testing"
        )
        info.validation_dataset = SpeechCommandsDataset(validation_source, torchaudio)
        return (
            SpeechCommandsDataset(train_source, torchaudio),
            SpeechCommandsDataset(test_source, torchaudio),
            info,
        )

    # 5. Sensordaten / HAR (Human Activity Recognition)
    if key in ("har_time_series", "uci_har"):
        train_ds, test_ds = _generate_synthetic_har(num_train=4500, num_test=900, seed=seed)
        return train_ds, test_ds, info

    # 6. Zeitreihen (Synthetisch / Allgemeine Muster)
    if key == "synthetic_time_series":
        train_ds, test_ds = _generate_synthetic_general_time_series(num_train=4000, num_test=800, seed=seed)
        return train_ds, test_ds, info

    # 7. Biomedizin / EEG Gehirnwellen
    if key in ("eeg_data", "eeg_bands"):
        train_ds, test_ds = _generate_synthetic_eeg(num_train=4000, num_test=800, seed=seed)
        return train_ds, test_ds, info

    if key == "eeg_eye_state":
        # 2-Klassen EEG Zustandserkennung (64 Merkmale)
        rng = np.random.RandomState(seed)
        def make_eye_data(n):
            x = np.zeros((n, 1, 1, 64), dtype=np.float32)
            y = np.zeros((n,), dtype=np.int64)
            for i in range(n):
                c = i % 2
                y[i] = c
                # Augen geschlossen = erhöhte Alpha-Wellen (Bins 15-30), Augen offen = Desynchronisation
                base = rng.normal(0.3, 0.05, 64)
                if c == 0:  # Augen geschlossen
                    base[15:30] += rng.uniform(0.3, 0.5, 15)
                else:       # Augen offen
                    base += rng.normal(0.0, 0.08, 64)
                x[i, 0, 0] = np.clip(base, 0.0, 1.0).astype(np.float32)
            return x, y
        xtr, ytr = make_eye_data(3000)
        xte, yte = make_eye_data(600)
        return TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte)), info

    # 8. Biomedizin / EKG Herzrhythmus
    if key in ("ecg_data", "ecg_heartbeat"):
        train_ds, test_ds = _generate_synthetic_ecg(num_train=4000, num_test=800, seed=seed)
        return train_ds, test_ds, info

    raise ValueError(f"Nicht implementierter Datensatz-Schlüssel: {key}")


if __name__ == "__main__":
    print_available_datasets_table()
