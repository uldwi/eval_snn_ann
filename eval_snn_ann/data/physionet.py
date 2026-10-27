"""PhysioNet EEG motor-imagery download and preprocessing."""
import os
import time
from typing import Tuple
import numpy as np
import torch
from scipy.signal import hilbert, resample, welch
from torch.utils.data import Dataset, TensorDataset

def _download_eegbci_run(raw_root: str, subject_id: int, run_id: int, allow_download: bool = True) -> str:
    """Lädt eine EEGBCI-EDF-Datei mit Resume und großzügigem Read-Timeout."""
    subject_name = f"S{subject_id:03d}"
    file_name = f"{subject_name}R{run_id:02d}.edf"
    dataset_dir = os.path.join(raw_root, "MNE-eegbci-data", "files", "eegmmidb", "1.0.0")
    subject_dir = os.path.join(dataset_dir, subject_name)
    os.makedirs(subject_dir, exist_ok=True)
    file_path = os.path.join(subject_dir, file_name)
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        return file_path
    if not allow_download:
        raise FileNotFoundError(
            f"EEGBCI file not found: {file_path}. Enable downloading or run dachs_setup.sh."
        )

    try:
        import requests
    except ImportError as exc:
        raise ImportError("Der PhysioNet-Download benötigt requests; installiere requirements.txt.") from exc

    partial_path = file_path + ".part"
    relative_path = f"eegmmidb/1.0.0/{subject_name}/{file_name}"
    sources = (
        ("AWS Open Data mirror", f"https://physionet-open.s3.amazonaws.com/{relative_path}"),
        ("PhysioNet HTTPS", f"https://physionet.org/files/{relative_path}"),
    )
    session = requests.Session()
    last_error = None
    try:
        for source_name, url in sources:
            for attempt in range(1, 5):
                offset = os.path.getsize(partial_path) if os.path.isfile(partial_path) else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                try:
                    print(
                        f"[EEG download: {source_name}] {subject_name}/{file_name}, attempt {attempt}/4"
                        + (f", resuming at byte {offset:,}" if offset else "")
                    )
                    with session.get(url, headers=headers, stream=True, timeout=(30, 600)) as response:
                        if response.status_code == 416 and offset:
                            os.remove(partial_path)
                            continue
                        response.raise_for_status()
                        append = offset > 0 and response.status_code == 206
                        if not append:
                            offset = 0
                        content_range = response.headers.get("Content-Range", "")
                        range_match = re.search(r"/([0-9]+)$", content_range)
                        if range_match:
                            expected_size = int(range_match.group(1))
                        else:
                            content_length = response.headers.get("Content-Length")
                            expected_size = offset + int(content_length) if content_length else None

                        mode = "ab" if append else "wb"
                        with open(partial_path, mode) as output:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    output.write(chunk)

                    actual_size = os.path.getsize(partial_path)
                    if expected_size is not None and actual_size != expected_size:
                        raise IOError(f"Unvollständiger Download ({actual_size}/{expected_size} Bytes)")
                    if actual_size == 0:
                        raise IOError("Leere EDF-Datei heruntergeladen")
                    os.replace(partial_path, file_path)
                    return file_path
                except (requests.RequestException, OSError) as exc:
                    last_error = exc
                    if attempt < 4:
                        time.sleep(min(2 ** (attempt - 1), 10))
    finally:
        session.close()

    raise RuntimeError(
        f"Download über AWS S3 und PhysioNet fehlgeschlagen: {subject_name}/{file_name}. "
        f"Unvollständige Daten bleiben unter {partial_path} und werden beim nächsten Versuch fortgesetzt."
    ) from last_error

def _load_physionet_eeg_mi(
    data_dir: str, seed: int, allow_download: bool = True, temporal: bool = False, csp: bool = False
) -> Tuple[Dataset, Dataset, Dataset]:
    """Lädt PhysioNet EEGMMIDB mit personengetrenntem Train/Val/Test-Split."""
    try:
        import mne
        from mne.datasets import eegbci
    except ImportError as exc:
        raise ImportError(
            "The real PhysioNet EEG motor imagery dataset requires MNE. "
            "Installiere die Projektabhängigkeiten mit `pip install -r requirements.txt`."
        ) from exc

    dataset_root = os.path.join(data_dir, "physionet_eeg_mi")
    raw_root = os.path.join(dataset_root, "raw")
    # v4 uses the band-limited analytic amplitude envelope, which preserves
    # temporal ERD/ERS changes while avoiding signed-current cancellation.
    cache_root = os.path.join(
        dataset_root,
        "raw_segments_csp_v1" if csp else ("processed_temporal_v4" if temporal else "processed_v1"),
    )
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
                features = cache["signals" if csp else "features"]
                labels = cache["labels"]
        else:
            subject_features = []
            subject_labels = []
            for run_id in (4, 8, 12):
                raw_path = _download_eegbci_run(
                    raw_root, int(subject_id), run_id, allow_download=allow_download
                )
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
                        if csp:
                            # Cache filtered samples; supervised CSP filters
                            # are fitted later using training subjects only.
                            subject_features.append(resample(segment, 128, axis=1).astype(np.float32))
                        elif temporal:
                            # Motor imagery is commonly expressed through
                            # mu/beta amplitude modulation. Use a positive
                            # envelope to avoid signed-current cancellation.
                            envelope = np.abs(hilbert(segment, axis=1))
                            sequence = resample(envelope, 128, axis=1)
                            subject_features.append(np.clip(sequence / 100e-6, 0.0, 1.0).astype(np.float32))
                        else:
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
                raise RuntimeError(f"No valid EEG motor imagery segments found for subject {subject_id}.")
            features = np.stack(subject_features)
            labels = np.asarray(subject_labels, dtype=np.int64)
            cache_payload = {"signals": features, "labels": labels} if csp else {"features": features, "labels": labels}
            np.savez_compressed(cache_path, **cache_payload)

        split = subject_split(int(subject_id))
        split_features[split].append(features)
        split_labels[split].append(labels)
        if int(subject_id) % 10 == 0:
            print(f"[PhysioNet EEGMI] Preprocessing: {subject_id}/109 subjects")

    if csp:
        try:
            from mne.decoding import CSP
        except ImportError as exc:
            raise ImportError(
                "CSP preprocessing requires scikit-learn. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc

        train_x = np.concatenate(split_features["train"], axis=0)
        train_y = np.concatenate(split_labels["train"], axis=0)
        csp_model = CSP(
            n_components=8,
            reg="ledoit_wolf",
            log=None,
            transform_into="csp_space",
            norm_trace=False,
        )
        csp_model.fit(train_x, train_y)
        transformed = {}
        for split in ("train", "validation", "test"):
            raw_x = np.concatenate(split_features[split], axis=0)
            csp_signals = csp_model.transform(raw_x)
            if temporal:
                transformed[split] = np.abs(hilbert(csp_signals, axis=2))
            else:
                sample_features = []
                for sample in csp_signals:
                    freqs, power = welch(sample, fs=64.0, nperseg=128, axis=1)
                    edges = np.linspace(8.0, 30.0, 17)
                    bands = []
                    for low, high in zip(edges[:-1], edges[1:]):
                        mask = (freqs >= low) & (freqs < high)
                        if np.any(mask):
                            band = power[:, mask].mean(axis=1)
                        else:
                            nearest = int(np.argmin(np.abs(freqs - (low + high) / 2.0)))
                            band = power[:, nearest]
                        bands.append(band)
                    log_power = 10.0 * np.log10(np.maximum(np.stack(bands, axis=1), 1e-30))
                    sample_features.append(log_power)
                transformed[split] = np.stack(sample_features)
        if temporal:
            # Component-wise scale is estimated on training subjects only.
            component_scale = np.percentile(transformed["train"], 99.0, axis=(0, 2))
            component_scale = np.maximum(component_scale, 1e-12)[None, :, None]
            for split in transformed:
                transformed[split] = np.clip(transformed[split] / component_scale, 0.0, 1.0).astype(np.float32)
        else:
            # Per-feature normalization is fitted on training subjects only.
            train_features = transformed["train"]
            feature_mean = train_features.mean(axis=0, keepdims=True)
            feature_std = np.maximum(train_features.std(axis=0, keepdims=True), 1e-6)
            for split in transformed:
                values = np.clip((transformed[split] - feature_mean) / feature_std, -3.0, 3.0)
                transformed[split] = ((values + 3.0) / 6.0).astype(np.float32)
        # The shared assembly below concatenates a list of per-subject arrays.
        # Wrap each already-merged CSP split once to preserve its sample axis.
        split_features = {split: [features] for split, features in transformed.items()}

    datasets_by_split = {}
    for split in ("train", "validation", "test"):
        x = np.concatenate(split_features[split], axis=0)[:, None, :, :]
        y = np.concatenate(split_labels[split], axis=0)
        datasets_by_split[split] = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

    return datasets_by_split["train"], datasets_by_split["validation"], datasets_by_split["test"]
