"""Synthetic datasets for smoke tests and toy time-series experiments."""
from typing import Tuple
import numpy as np
import torch
from scipy.signal import hilbert, resample, welch
from torch.utils.data import Dataset, TensorDataset

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
