import numpy as np
from quantization import uniform_quantization, amplitude_normalize

def mulaw_compand(x, mu=255):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

def maskhelper(alpha, N):
    return np.random.rand(N) < alpha

def dither_gen(bitdepth, N, alpha, Dclass=2):
    Delta = 2 / 2**bitdepth
    G = np.random.rand(N) - 0.5
    R = G * Delta
    if Dclass == 1:
        mask = maskhelper(alpha, N)
        return R * mask
    elif Dclass == 2:
        mask = maskhelper(alpha, N)
        return alpha * R * mask
    else:
        print("Invalid dithering class. Use 1 (masked), 2 (masked+scaled)")


def dither_app(input_signal, bitdepth, param, Qtype='uniform', Dtype='subtractive', compand='mulaw'):
    Delta = 2 / 2**bitdepth
    hrf = 1 / (1 + Delta)

    dither = dither_gen(bitdepth, input_signal.size, param, Dclass=2)
    
    companded = mulaw_compand(input_signal) if compand == 'mulaw' else input_signal

    dithering = companded*hrf + dither

    if Qtype == 'uniform':
        quantized = uniform_quantization(dithering, bitdepth)
    else:
        print("Invalid quantization type.")
        return

    if Dtype == 'subtractive':
        processed = quantized - dither
    elif Dtype == 'nonsubtractive':
        processed = quantized
    else:
        print("Invalid dithering type. Use 'subtractive' or 'non-subtractive'.")
        return
    
    normalized = processed / np.max(np.abs(processed))
    return normalized


class _DitherQuantizeFn:
    def __init__(self, bitdepth, dither_alpha, compand):
        self.bitdepth = bitdepth
        self.dither_alpha = dither_alpha
        self.compand = compand

    def __call__(self, audio):
        return dither_app(
            amplitude_normalize(audio), self.bitdepth, self.dither_alpha,
            compand=self.compand,
        )


class _MulawQuantizeFn:
    def __init__(self, bitdepth):
        self.bitdepth = bitdepth

    def __call__(self, audio):
        return uniform_quantization(mulaw_compand(amplitude_normalize(audio)), self.bitdepth)


class _PlainQuantizeFn:
    def __init__(self, bitdepth):
        self.bitdepth = bitdepth

    def __call__(self, audio):
        return uniform_quantization(amplitude_normalize(audio), self.bitdepth)


def build_quantize_fn(bitdepth, dither=False, mulaw=False, dither_alpha=1.0):
    """
    Compose the audio -> quantized-audio pipeline for one (bitdepth, dither,
    mulaw) run setting used by the evaluation sweep in train.py.

    dither=True routes through dither_app (subtractive dithering); note that
    when mulaw is also on, dither_app never applies the inverse mu-law
    expansion after subtracting the dither -- that omission is intentional,
    not a bug (see README Evaluation section).

    Returns a picklable callable object (not a closure/lambda) since
    DataLoader workers on Windows use the 'spawn' start method, which
    requires pickling the dataset (and this function) to send to workers.
    """
    if dither:
        compand = 'mulaw' if mulaw else 'none'
        return _DitherQuantizeFn(bitdepth, dither_alpha, compand)
    if mulaw:
        return _MulawQuantizeFn(bitdepth)
    return _PlainQuantizeFn(bitdepth)


