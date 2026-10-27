"""Dataset metadata, aliases, and discovery helpers."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset

@dataclass
class DatasetInfo:
    """Metadata and specifications for a dataset."""
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
    # 1. Computer vision
    "mnist": {
        "domain": "Computer vision",
        "description": "Classic MNIST dataset of handwritten digits (0-9)",
        "input_size": 784,
        "in_channels": 1,
        "in_height": 28,
        "in_width": 28,
        "num_classes": 10,
        "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    },
    "fashion_mnist": {
        "domain": "Computer vision",
        "description": "Fashion-MNIST dataset of Zalando clothing items",
        "input_size": 784,
        "in_channels": 1,
        "in_height": 28,
        "in_width": 28,
        "num_classes": 10,
        "class_names": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    },
    "cifar10": {
        "domain": "Computer vision",
        "description": "CIFAR-10 object categories (16x16 grayscale images for resource-efficient ESP32 inference)",
        "input_size": 256,
        "in_channels": 1,
        "in_height": 16,
        "in_width": 16,
        "num_classes": 10,
        "class_names": ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"],
    },

    # 2. Audio and keyword spotting
    "speech_commands": {
        "domain": "Audio and keyword spotting",
        "description": "Real recordings from Speech Commands v0.02; all 35 spoken words represented as Mel spectrograms",
        "input_size": 256,
        "in_channels": 1,
        "in_height": 16,
        "in_width": 16,
        "num_classes": 35,
        "class_names": ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"],
    },
    "speech_commands_spikcommander": {
        "domain": "Audio and keyword spotting (SpikCommander Mel sequence)",
        "description": "Speech Commands v0.02 with 8 kHz audio and 140 Mel bands as a temporal sequence",
        "input_size": 14000,
        "in_channels": 1,
        "in_height": 140,
        "in_width": 100,
        "num_classes": 35,
        "class_names": ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"],
    },
    "kws": {
        "domain": "Audio and keyword spotting",
        "description": "Alias for Speech Commands v0.02 with all 35 spoken words",
        "input_size": 256,
        "in_channels": 1,
        "in_height": 16,
        "in_width": 16,
        "num_classes": 35,
        "class_names": ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"],
    },

    # 3. Sensor data and time series
    "har_time_series": {
        "domain": "Sensor data and time series",
        "description": "Six-channel IMU sensor data (acceleration and gyroscope) for human activity recognition",
        "input_size": 120,
        "in_channels": 1,
        "in_height": 6,
        "in_width": 20,
        "num_classes": 6,
        "class_names": ["Walking", "Walking upstairs", "Walking downstairs", "Sitting", "Standing", "Laying"],
    },
    "uci_har": {
        "domain": "Sensor data and time series",
        "description": "Alias for har_time_series (6 activities, 6x20 IMU features)",
        "input_size": 120,
        "in_channels": 1,
        "in_height": 6,
        "in_width": 20,
        "num_classes": 6,
        "class_names": ["Walking", "Walking upstairs", "Walking downstairs", "Sitting", "Standing", "Laying"],
    },
    "synthetic_time_series": {
        "domain": "Sensor data and time series",
        "description": "Synthetic time series with four signal patterns (sine oscillation, impulse, trend, chirp frequency)",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 1,
        "in_width": 128,
        "num_classes": 4,
        "class_names": ["Sine oscillation", "Impulse/step", "Linear trend", "Chirp modulation"],
    },

    # 4. Biomedical signals and EEG
    "eeg_data": {
        "domain": "Biomedical signals (EEG)",
        "description": "Electroencephalogram (EEG) signals grouped into four mental states (delta, theta, alpha, beta)",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 8,
        "in_width": 16,
        "num_classes": 4,
        "class_names": ["Deep sleep (delta)", "Drowsy (theta)", "Relaxed (alpha)", "Focused (beta)"],
    },
    "eeg_bands": {
        "domain": "Biomedical signals (EEG)",
        "description": "Alias for eeg_data (four brainwave bands)",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 8,
        "in_width": 16,
        "num_classes": 4,
        "class_names": ["Deep sleep (delta)", "Drowsy (theta)", "Relaxed (alpha)", "Focused (beta)"],
    },
    "eeg_eye_state": {
        "domain": "Biomedical signals (EEG)",
        "description": "EEG state classification: eyes open vs. eyes closed",
        "input_size": 64,
        "in_channels": 1,
        "in_height": 1,
        "in_width": 64,
        "num_classes": 2,
        "class_names": ["Eyes closed", "Eyes open"],
    },
    "physionet_eeg_mi": {
        "domain": "Biomedical signals (EEG motor imagery)",
        "description": "Real PhysioNet EEGMMIDB recordings: imagined left- vs. right-fist movement, split by subject",
        "input_size": 1024,
        "in_channels": 1,
        "in_height": 64,
        "in_width": 16,
        "num_classes": 2,
        "class_names": ["Left fist (motor imagery)", "Right fist (motor imagery)"],
    },
    "physionet_eeg_mi_temporal": {
        "domain": "Biomedical signals (EEG motor imagery, temporal)",
        "description": "PhysioNet EEGMMIDB, 64-channel temporal envelopes of mu/beta activity with 128 steps",
        "input_size": 8192,
        "in_channels": 1,
        "in_height": 64,
        "in_width": 128,
        "num_classes": 2,
        "class_names": ["Left fist (motor imagery)", "Right fist (motor imagery)"],
    },
    "physionet_eeg_mi_csp_temporal": {
        "domain": "Biomedical signals (EEG motor imagery, CSP temporal)",
        "description": "PhysioNet EEGMMIDB, eight CSP components learned on training data as temporal mu/beta envelopes",
        "input_size": 1024,
        "in_channels": 1,
        "in_height": 8,
        "in_width": 128,
        "num_classes": 2,
        "class_names": ["Left fist (motor imagery)", "Right fist (motor imagery)"],
    },
    "physionet_eeg_mi_csp_spectral": {
        "domain": "Biomedical signals (EEG motor imagery, CSP spectral)",
        "description": "PhysioNet EEGMMIDB, eight CSP components learned on training data with 16 spectral bands",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 8,
        "in_width": 16,
        "num_classes": 2,
        "class_names": ["Left fist (motor imagery)", "Right fist (motor imagery)"],
    },

    # 5. Biomedical signals and cardiology (ECG)
    "ecg_data": {
        "domain": "Biomedical signals (ECG rhythm)",
        "description": "Electrocardiogram (ECG) heartbeat classification using the MIT-BIH arrhythmia standard (5 classes)",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 1,
        "in_width": 128,
        "num_classes": 5,
        "class_names": ["Normal beat (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion beat (F)", "Unknown/paced (Q)"],
    },
    "ecg_heartbeat": {
        "domain": "Biomedical signals (ECG rhythm)",
        "description": "Alias for ecg_data (five heartbeat rhythm classes)",
        "input_size": 128,
        "in_channels": 1,
        "in_height": 1,
        "in_width": 128,
        "num_classes": 5,
        "class_names": ["Normal beat (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion beat (F)", "Unknown/paced (Q)"],
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
    """Normalize dataset aliases to their canonical names."""
    key = str(name).strip().lower()
    return DATASET_ALIASES.get(key, key)

def list_available_datasets() -> List[DatasetInfo]:
    """Return the available datasets and their metadata."""
    unique_names = ["mnist", "fashion_mnist", "cifar10", "speech_commands", "speech_commands_spikcommander", "har_time_series", "synthetic_time_series", "eeg_data", "eeg_eye_state", "physionet_eeg_mi", "ecg_data"]
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
    """Print a table of available datasets."""
    datasets_list = list_available_datasets()
    print("\n" + "=" * 90)
    print("AVAILABLE DATASETS BY DOMAIN")
    print("=" * 90)
    print(f"{'Name':<22} | {'Domain':<32} | {'Input shape':<14} | {'Classes':<8}")
    print("-" * 90)
    for ds in datasets_list:
        shape_str = f"{ds.in_channels}x{ds.in_height}x{ds.in_width} ({ds.input_size})"
        print(f"{ds.name:<22} | {ds.domain:<32} | {shape_str:<14} | {ds.num_classes:<8}")
    print("=" * 90 + "\n")
