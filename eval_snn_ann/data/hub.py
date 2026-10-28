"""Dataset loading dispatch and train/test split construction."""
import os
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms

from .registry import AVAILABLE_DATASETS, DatasetInfo, normalize_dataset_name
from .speech_commands import SpeechCommandsDataset
from .synthetic import (
    _generate_synthetic_ecg, _generate_synthetic_eeg,
    _generate_synthetic_general_time_series, _generate_synthetic_har,
)
from .physionet import _load_physionet_eeg_mi

def get_dataset(
    name: str = "mnist",
    data_dir: str = "./data",
    download: bool = True,
    seed: int = 42,
    augment_train: bool = False,
) -> Tuple[Dataset, Dataset, DatasetInfo]:
    """
    Lädt einen Datensatz anhand des Namens oder Alias.

    Gibt (train_dataset, test_dataset, dataset_info) zurück.
    """
    key = normalize_dataset_name(name)
    if key not in AVAILABLE_DATASETS:
        supported = ", ".join(f"'{k}'" for k in AVAILABLE_DATASETS.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Supported datasets are: {supported}.\n"
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

    if key in ("physionet_eeg_mi", "eeg_mi", "physionet_eeg_mi_temporal", "physionet_eeg_mi_csp_temporal", "physionet_eeg_mi_csp_spectral"):
        train_ds, validation_ds, test_ds = _load_physionet_eeg_mi(
            data_dir=data_dir,
            seed=seed,
            allow_download=download,
            temporal=key in ("physionet_eeg_mi_temporal", "physionet_eeg_mi_csp_temporal"),
            csp=key in ("physionet_eeg_mi_csp_temporal", "physionet_eeg_mi_csp_spectral"),
        )
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
    if key in ("speech_commands", "speech_commands_spikcommander", "kws"):
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
        spikcommander = key == "speech_commands_spikcommander"
        info.validation_dataset = SpeechCommandsDataset(validation_source, torchaudio, spikcommander=spikcommander)
        return (
            SpeechCommandsDataset(
                train_source, torchaudio, spikcommander=spikcommander,
                augment=spikcommander or augment_train,
            ),
            SpeechCommandsDataset(test_source, torchaudio, spikcommander=spikcommander),
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

    raise ValueError(f"Dataset key is not implemented: {key}")
