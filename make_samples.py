"""
make_samples.py -- generate audible 2-bit dithered WAV samples + spectrogram comparisons.

Picks one clip per word from the SpeechCommands dataset, applies four conditions:
  - original (no processing)
  - 2-bit, alpha=0.0  (pure uniform quantization, no dither)
  - 2-bit, alpha=0.5  (subtractive dither, half amplitude)
  - 2-bit, alpha=1.0  (full subtractive dither)

Output: python-files/toymodel/samples/
  {word}_original.wav
  {word}_2bit_0.0alpha.wav
  {word}_2bit_0.5alpha.wav
  {word}_2bit_1.0alpha.wav
  {word}_comparison.png
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_toymodel = Path(__file__).parent
_pkg_root  = _toymodel.parent
sys.path.insert(0, str(_pkg_root))
sys.path.insert(0, str(_pkg_root / "experiments"))
from experiments.dithering   import dither_app
from experiments.spectrogram import log_mel_spectrogram, HOP, SR

DATA_ROOT = _toymodel / "data" / "SpeechCommands" / "speech_commands_v0.02"
OUT_DIR   = _toymodel / "samples"
WORDS     = ["yes", "no", "go", "stop", "up"]
BITDEPTH  = 2
ALPHAS    = [0.0, 0.5, 1.0]

CONDITIONS = [("original", None)] + [(f"2bit_{a}alpha", a) for a in ALPHAS]


def load_clip(word: str) -> np.ndarray:
    """Load the first WAV in the word's folder, normalised to [-1, 1]."""
    clips = sorted((DATA_ROOT / word).glob("*.wav"))
    if not clips:
        raise FileNotFoundError(f"No WAVs found for word '{word}'")
    audio, sr = sf.read(str(clips[0]), dtype="float32", always_2d=False)
    assert sr == SR, f"Expected {SR} Hz, got {sr}"
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio /= peak
    return audio


def apply_condition(audio: np.ndarray, alpha) -> np.ndarray:
    """Return processed audio, clipped to [-1, 1] for safe WAV export."""
    if alpha is None:
        return audio.copy()
    processed = dither_app(audio, BITDEPTH, alpha, Qtype="uniform", Dtype="subtractive")
    pre_clip_max = np.max(np.abs(processed))
    processed = np.clip(processed, -1.0, 1.0)
    return processed, pre_clip_max


def plot_comparison(word: str, signals: dict, out_path: Path):
    labels = list(signals.keys())
    n = len(labels)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7),
                             gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle(f'2-bit dither comparison — "{word}"', fontsize=13)

    for col, label in enumerate(labels):
        sig = signals[label]
        t   = np.arange(len(sig)) / SR

        ax_w = axes[0, col]
        ax_w.plot(t, sig, linewidth=0.5, color="#2196F3")
        ax_w.set_xlim(0, t[-1])
        ax_w.set_ylim(-1.15, 1.15)
        ax_w.set_title(label.replace("_", "\n"), fontsize=8)
        ax_w.set_xlabel("time (s)", fontsize=7)
        if col == 0:
            ax_w.set_ylabel("amplitude", fontsize=7)
        ax_w.tick_params(labelsize=6)
        ax_w.grid(True, alpha=0.25)

        ax_s = axes[1, col]
        spec = log_mel_spectrogram(sig)
        im   = ax_s.imshow(spec, origin="lower", aspect="auto",
                           extent=[0, spec.shape[1] * HOP / SR, 0, 80],
                           vmin=-1, vmax=1, cmap="magma")
        ax_s.set_xlabel("time (s)", fontsize=7)
        if col == 0:
            ax_s.set_ylabel("mel band", fontsize=7)
        ax_s.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax_s, shrink=0.7, pad=0.02)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    for word in WORDS:
        print(f"\n--- {word} ---")
        audio = load_clip(word)
        print(f"  loaded {len(audio)} samples ({len(audio)/SR:.2f}s)")

        signals = {}
        for label, alpha in CONDITIONS:
            if alpha is None:
                sig = apply_condition(audio, None)
                pre_max = np.max(np.abs(sig))
            else:
                sig, pre_max = apply_condition(audio, alpha)
                print(f"  {label}: pre-clip max amplitude = {pre_max:.4f}")

            wav_path = OUT_DIR / f"{word}_{label}.wav"
            sf.write(str(wav_path), sig, SR, subtype="PCM_16")
            signals[label] = sig

        png_path = OUT_DIR / f"{word}_comparison.png"
        plot_comparison(word, signals, png_path)
        print(f"  saved WAVs + {png_path.name}")

    print(f"\nDone. {len(WORDS) * len(CONDITIONS)} WAV files + {len(WORDS)} PNGs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
