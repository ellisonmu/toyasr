import numpy as np

def amplitude_normalize(input):
    peak = np.max(np.abs(input))
    if peak == 0:
        return input
    return input / peak

def uniform_quantization(input, bitdepth, type='mid-rise'):
    input = np.clip(input, -1, 1)
    Delta = 2 / 2**bitdepth
    if type == 'mid-rise':
        return Delta * np.floor(input / Delta) + Delta/2
    elif type == 'mid-tread':
        return Delta * np.floor(input / Delta + 0.5)
    else:
        print("Invalid quantization type. Use 'mid-rise' or 'mid-tread'.")
