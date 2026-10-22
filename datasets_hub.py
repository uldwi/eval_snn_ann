#!/usr/bin/env python3
"""
Datensatz-Hub für neuronale Netze (ANN & SNN) auf eingebetteten Systemen (ESP32).

Unterstützte Datensätze und Domänen:
1. Computer Vision / Bildverarbeitung:
   - "mnist": Handgeschriebene Ziffern (0-9), 1x28x28 = 784 Features, 10 Klassen
   - "fashion_mnist": Kleidungsstücke (Zalando), 1x28x28 = 784 Features, 10 Klassen
   - "cifar10": Objektkategorien (herunterskaliert auf 1x16x16=256 oder 1x32x32=1024), 10 Klassen

2. Audio / Keyword Recognition (Keyword Spotting - KWS):
   - "speech_commands" / "kws": Sprachbefehle ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"),
     1x16x16 = 256 Features (Zeit-Frequenz-Spektrogramm / Mel-Filterbänke), 10 Klassen

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
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
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
        "description": "Speech Commands Spektrogramme für 10 Schlüsselwörter (yes, no, up, down, left, right, on, off, stop, go)",
        "input_size": 256,
        "in_channels": 1,
        "in_height": 16,
        "in_width": 16,
        "num_classes": 10,
        "class_names": ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
    },
    "kws": {
        "domain": "Audio & Keyword Spotting (Sprachbefehlserkennung)",
        "description": "Alias für speech_commands (Keyword Spotting, 10 Klassen)",
        "input_size": 256,
        "in_channels": 1,
        "in_height": 16,
        "in_width": 16,
        "num_classes": 10,
        "class_names": ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
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
    "ecg_heartbeat": "ecg_data",
}


def normalize_dataset_name(name: str) -> str:
    """Normalisiert Alias-Namen auf den kanonischen Datensatznamen."""
    key = str(name).strip().lower()
    return DATASET_ALIASES.get(key, key)


def list_available_datasets() -> List[DatasetInfo]:
    """Gibt eine Liste aller verfügbaren Datensätze und deren Metadaten zurück."""
    unique_names = ["mnist", "fashion_mnist", "cifar10", "speech_commands", "har_time_series", "synthetic_time_series", "eeg_data", "eeg_eye_state", "ecg_data"]
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
# Deterministische Generatoren für Audio-, Sensor- und Biomedizin-Datensätze
# ---------------------------------------------------------------------------

def _generate_synthetic_speech_commands(
    num_train: int = 5000,
    num_test: int = 1000,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Erzeugt akustische Zeit-Frequenz-Repräsentationen (16x16 Mel-Spektrogramm-Äquivalente)
    für 10 Sprachbefehle mit formantenbasierten akustischen Charakteristiken.
    """
    rng = np.random.RandomState(seed)
    num_classes = 10
    h, w = 16, 16

    # Formant- und Energiemuster pro Schlüsselwort
    patterns = [
        # 0: "yes" - steigender hoher Formant (e->s)
        {"f1": 3, "f2": 11, "slope": 0.5, "noise": 0.2, "burst_pos": 0.7},
        # 1: "no" - tiefer fallender Formant (n->o)
        {"f1": 2, "f2": 6, "slope": -0.4, "noise": 0.1, "burst_pos": 0.2},
        # 2: "up" - steiler Anstieg mit Plosiv am Ende (p)
        {"f1": 4, "f2": 8, "slope": 0.8, "noise": 0.25, "burst_pos": 0.85},
        # 3: "down" - breites Frequenzspektrum sinkend
        {"f1": 5, "f2": 9, "slope": -0.6, "noise": 0.15, "burst_pos": 0.15},
        # 4: "left" - zweiphasiges Vokal-Frikativ-Muster
        {"f1": 4, "f2": 12, "slope": 0.2, "noise": 0.35, "burst_pos": 0.6},
        # 5: "right" - starker Diphtong mit Rauschkomponente
        {"f1": 3, "f2": 10, "slope": 0.6, "noise": 0.2, "burst_pos": 0.3},
        # 6: "on" - resonantes Mittelfrequenzband
        {"f1": 5, "f2": 7, "slope": 0.0, "noise": 0.1, "burst_pos": 0.5},
        # 7: "off" - langes Rauschband am Ende (ff)
        {"f1": 4, "f2": 8, "slope": -0.2, "noise": 0.5, "burst_pos": 0.7},
        # 8: "stop" - Plosiv (st) + Vokal (o) + Plosiv (p)
        {"f1": 2, "f2": 13, "slope": 0.0, "noise": 0.4, "burst_pos": 0.1},
        # 9: "go" - tiefer Vokal mit Obertönen
        {"f1": 2, "f2": 5, "slope": -0.3, "noise": 0.1, "burst_pos": 0.4},
    ]

    def make_samples(n_samples: int):
        x_data = np.zeros((n_samples, 1, h, w), dtype=np.float32)
        y_data = np.zeros((n_samples,), dtype=np.int64)

        for i in range(n_samples):
            cls = i % num_classes
            y_data[i] = cls
            pat = patterns[cls]

            spec = np.zeros((h, w), dtype=np.float32)
            time_axis = np.linspace(0, 1, w)

            f1_center = pat["f1"] + rng.uniform(-0.5, 0.5)
            f2_center = pat["f2"] + rng.uniform(-0.5, 0.5)
            slope = pat["slope"] + rng.uniform(-0.1, 0.1)

            for t_idx, t_val in enumerate(time_axis):
                cur_f1 = np.clip(f1_center + slope * t_val * 4, 0, h - 1)
                cur_f2 = np.clip(f2_center + slope * t_val * 3, 0, h - 1)

                # Formant-Energie Gauss
                for freq_bin in range(h):
                    d1 = (freq_bin - cur_f1) ** 2
                    d2 = (freq_bin - cur_f2) ** 2
                    spec[freq_bin, t_idx] += 0.8 * np.exp(-d1 / 1.8) + 0.6 * np.exp(-d2 / 2.2)

                # Plosiv / Konsonanten-Burst
                burst_diff = abs(t_val - pat["burst_pos"])
                if burst_diff < 0.15:
                    burst_energy = (1.0 - burst_diff / 0.15) * pat["noise"]
                    spec[:, t_idx] += burst_energy * rng.uniform(0.2, 1.0, size=(h,))

            # Rausch- und Varianzbeimischung
            noise = rng.normal(0.0, 0.05 + 0.05 * pat["noise"], size=(h, w))
            spec = np.clip(spec + noise, 0.0, 1.0)
            # Normalisierung auf 0.0 - 1.0
            if spec.max() > 0:
                spec = spec / spec.max()

            x_data[i, 0] = spec.astype(np.float32)
        return x_data, y_data

    x_tr, y_tr = make_samples(num_train)
    x_te, y_te = make_samples(num_test)

    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    test_ds = TensorDataset(torch.from_numpy(x_te), torch.from_numpy(y_te))
    return train_ds, test_ds


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
    key = str(name).strip().lower()
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

    # 4. Audio: Speech Commands (Keyword Spotting)
    if key in ("speech_commands", "kws"):
        train_ds, test_ds = _generate_synthetic_speech_commands(num_train=5000, num_test=1000, seed=seed)
        return train_ds, test_ds, info

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
