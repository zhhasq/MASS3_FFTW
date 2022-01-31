import numpy as np
import scipy.ndimage.filters as ndif

def z_norm(input):
    """input is ndarray"""
    if input.ndim == 1:
        return (input - input.mean()) / input.std()
    elif input.ndim == 2:
        results = np.empty(input.shape)
        for i in range(input.shape[0]):
            results[i, :] = z_norm(input[i, :])
        return results
    else:
        return None


def next_power_of_2(number):
    return int(np.ceil(np.log2(number)))


def mov_mean_std(input, w):
    input = np.array(input, dtype=np.double)
    input2 = input ** 2
    n = len(input)
    num_r = n - w + 1

    start = int(np.floor(w / 2))
    end = start + num_r - 1
    mean = ndif.uniform_filter1d(input, w)[start: end + 1]
    std2 = ndif.uniform_filter1d(input2, w)[start: end + 1] - mean ** 2
    return mean, std2


def p_corr(dot, m, mean_ts, mean_query, std2_ts, std2_query):
    return (dot - m * (mean_ts * mean_query)) / (m * np.sqrt(std2_ts * std2_query))

