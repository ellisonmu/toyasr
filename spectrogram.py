import numpy as np
import librosa
import matplotlib.pyplot as plt

SR = 16000
N_FFT = 400       # 25ms window
HOP = 160         # 10ms stride
N_MELS = 80

_MEL_FB = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)


def log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    80-channel log magnitude mel spectrogram
    audio: float32 array in [-1, 1], shape (T,)
    returns: (80, frames) normalized to approx [-1, 1]
    """
    if len(audio) < 2:
        audio = np.zeros(N_FFT, dtype=np.float32)
    stft = librosa.stft(
        y=audio, n_fft=N_FFT, hop_length=HOP,
        window="hann", center=True, pad_mode="constant"
    )
    mel = _MEL_FB @ np.abs(stft) ** 2
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)  
    log_mel = (log_mel + 4.0) / 4.0 # shift to approx [-1, 1]
    return log_mel.astype(np.float32)

def plot_spectrogram(spec: np.ndarray, title: str = "", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(spec, origin="lower", aspect="auto",
                    extent=[0, spec.shape[1] * HOP / SR, 0, N_MELS])
    plt.colorbar(img, ax=ax, label="normalized log mag")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel channel")
    ax.set_title(title)
    return ax

def compare_spectrograms(specs: dict[str, np.ndarray]):
    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, spec) in zip(axes, specs.items()):
        plot_spectrogram(spec, title=name, ax=ax)
    plt.tight_layout()
    return fig
