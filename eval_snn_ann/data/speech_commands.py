"""Real Speech Commands audio and Mel feature pipelines."""
import os
from typing import Any
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Dataset

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

    def __init__(self, source: Dataset, torchaudio_module, spikcommander: bool = False, augment: bool = False):
        self.source = source
        self.spikcommander = bool(spikcommander)
        self.augment = bool(augment)
        self.keyword_to_index = {word: i for i, word in enumerate(SPEECH_COMMANDS)}
        self.samples = []
        for i in range(len(source)):
            relative_path, sample_rate, label, _, _ = source.get_metadata(i)
            if label in self.keyword_to_index:
                # get_metadata() returns paths relative to the archive root (_archive),
                # while _path already includes the speech_commands_v0.02 subdirectory.
                self.samples.append((os.path.join(source._archive, relative_path), sample_rate, label))

        self.sample_rate = 8000 if self.spikcommander else 16000
        self.mel_transform = torchaudio_module.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=256 if self.spikcommander else 400,
            win_length=256 if self.spikcommander else 400,
            hop_length=80 if self.spikcommander else 160,
            n_mels=140 if self.spikcommander else 40,
            f_min=20.0,
            f_max=4000.0 if self.spikcommander else 7600.0,
            power=2.0,
        )
        self.feature_cache = [None] * len(self.samples)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _apply_spec_augment(features: torch.Tensor) -> torch.Tensor:
        """Apply one time and one frequency mask to a cached Mel feature map."""
        augmented = features.clone()
        mel_bins, frames = augmented.shape[-2:]
        # Preserve the original large-model masks while scaling them to smaller maps.
        max_freq_width = min(12, mel_bins) if mel_bins == 140 else max(1, int(mel_bins * 0.1))
        max_time_width = min(10, frames) if frames == 100 else max(1, int(frames * 0.1))
        freq_width = int(np.random.randint(0, max_freq_width + 1))
        if freq_width:
            freq_start = int(np.random.randint(0, mel_bins - freq_width + 1))
            augmented[:, freq_start:freq_start + freq_width, :] = 0
        time_width = int(np.random.randint(0, max_time_width + 1))
        if time_width:
            time_start = int(np.random.randint(0, frames - time_width + 1))
            augmented[:, :, time_start:time_start + time_width] = 0
        return augmented

    def load_normalized_audio(self, index, resample=True):
        """Lädt einen Clip als mono float32 PCM mit genau einer Sekunde Länge."""
        wav_path, sample_rate, _ = self.samples[index]
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
        if audio.ndim > 1:
            audio = audio.mean(axis=1, dtype=np.float32)
        if self.spikcommander and resample:
            # Properly anti-alias while converting the original 16-kHz GSC audio to 8 kHz.
            audio = resample_poly(audio, up=1, down=2).astype(np.float32, copy=False)
            sample_rate = 8000
        target_rate = 8000 if self.spikcommander and resample else 16000
        if audio.size < target_rate:
            audio = np.pad(audio, (0, target_rate - audio.size))
        else:
            audio = audio[:target_rate]
        return np.asarray(audio, dtype=np.float32)

    def __getitem__(self, index):
        cached = self.feature_cache[index]
        if cached is not None:
            result = cached
            if not self.augment:
                return result
            return self._apply_spec_augment(result[0]), result[1]

        _, sample_rate, label = self.samples[index]
        waveform = torch.from_numpy(self.load_normalized_audio(index).copy()).unsqueeze(0)

        mel = self.mel_transform(waveform)
        mel_db = 10.0 * torch.log10(mel.clamp_min(1e-10))
        mel_db = (mel_db - mel_db.max()).clamp(min=-80.0, max=0.0)
        mel = (mel_db + 80.0) / 80.0
        target_shape = (140, 100) if self.spikcommander else (16, 16)
        features = torch.nn.functional.adaptive_avg_pool2d(mel.unsqueeze(0), target_shape).squeeze(0)
        result = (features.to(torch.float32), self.keyword_to_index[label])
        self.feature_cache[index] = result
        if self.augment:
            return self._apply_spec_augment(result[0]), result[1]
        return result
