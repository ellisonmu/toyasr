"""
SpeechCommandsCTC / ManifestCommandsCTC

Google Speech Commands (v0.02 -- 35 single-word commands, ~1s clips @ 16kHz)
  - SpeechCommandsCTC -> log-mel spectrogram
  - ManifestCommandsCTC -> reads pre-compressed WAVs from CSV
"""

import csv
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio
import torchaudio.datasets.speechcommands as _speechcommands
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

_pkg_root = Path(__file__).parent
sys.path.insert(0, str(_pkg_root))
from spectrogram import log_mel_spectrogram

# torchaudio >=2.9 routes torchaudio.load() through TorchCodec, which needs
# system FFmpeg shared libs that aren't readily available on Windows. Speech
# Commands clips are plain 16kHz WAV, so load them via soundfile instead.
def _load_waveform_soundfile(root, filename, exp_sample_rate):
    audio, sample_rate = sf.read(os.path.join(root, filename), dtype="float32", always_2d=True)
    if exp_sample_rate != sample_rate:
        raise ValueError(f"sample rate should be {exp_sample_rate}, but got {sample_rate}")
    return torch.from_numpy(audio.T)  # (channel, time)

_speechcommands._load_waveform = _load_waveform_soundfile

SR = 16_000

# character-level CTC vocabulary: blank + a-z.
BLANK_IDX   = 0
CHARS       = "abcdefghijklmnopqrstuvwxyz"
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
VOCAB_SIZE  = len(CHARS) + 1 

def encode(label: str) -> torch.Tensor:
    return torch.tensor([CHAR_TO_IDX[c] for c in label.lower()], dtype=torch.long)

def decode(indices) -> str:
    return "".join(IDX_TO_CHAR[i] for i in indices if i != BLANK_IDX)

def _load_quantized(dataset, idx, quantize_fn):
    """Fetch one SPEECHCOMMANDS example and run it through quantize_fn."""
    waveform, sample_rate, label, *_ = dataset[idx]
    assert sample_rate == SR, f"expected {SR} Hz, got {sample_rate}"
    audio = waveform.squeeze(0).numpy().astype(np.float32)
    if quantize_fn is not None:
        audio = quantize_fn(audio)
    return audio, label

class SpeechCommandsCTC(Dataset):
    """for train.py """

    def __init__(self, root, subset="training", quantize_fn=None, download=True):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(root), download=download, subset=subset,
        )
        self.quantize_fn = quantize_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio, label = _load_quantized(self.dataset, idx, self.quantize_fn)
        audio = np.clip(audio, -1.0, 1.0)  # subtractive dither output can exceed [-1,1]
        feats = log_mel_spectrogram(audio) # (80, frames)
        feats = torch.from_numpy(feats.T).float() # (frames, 80) for batch_first LSTM
        return feats, encode(label)

def ctc_collate(batch):
    """Pad variable-length log-mel features/targets; return CTCLoss-ready tensors."""
    feats, targets = zip(*batch)
    input_lengths  = torch.tensor([f.shape[0] for f in feats],   dtype=torch.long)
    target_lengths = torch.tensor([len(t)     for t in targets], dtype=torch.long)
    feats_padded = pad_sequence(feats, batch_first=True)  # (B, T, 80)
    targets_cat  = torch.cat(targets) # CTCLoss expects concatenated targets
    return feats_padded, targets_cat, input_lengths, target_lengths

class ManifestCommandsCTC(Dataset):
    def __init__(self, manifest_csv, feature_type="waveform", quantize_fn=None):
        if feature_type not in ("waveform", "logmel"):
            raise ValueError("feature_type must be 'waveform' or 'logmel'")
        self.feature_type = feature_type
        self.quantize_fn  = quantize_fn
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            self.pairs = [(row["audio_file"], row["label"]) for row in csv.DictReader(f)]
        if not self.pairs:
            raise ValueError(f"Manifest is empty: {manifest_csv}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio_path, label = self.pairs[idx]
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        assert sample_rate == SR, f"expected {SR} Hz, got {sample_rate}"
        audio = audio.astype(np.float32)
        if self.quantize_fn is not None:
            audio = self.quantize_fn(audio)
        if self.feature_type == "logmel":
            feats = log_mel_spectrogram(audio)
            return torch.from_numpy(feats.T).float(), encode(label)
        return torch.from_numpy(audio).float(), encode(label)
