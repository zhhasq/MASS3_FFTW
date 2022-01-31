import numpy as np
from mass_fft.mass_utils import *
from math import dist


def mass_naive(data, query):
    n = len(data)
    m = len(query)
    results = np.zeros(n - m + 1)
    for i in range(0, n - m + 1):
        results[i] = np.corrcoef(data[i : i + m], query)[0, 1]
    return results

def mass_naive_eu(data, query):
    n = len(data)
    m = len(query)
    results = np.zeros(n - m + 1)
    query = z_norm(query)
    for i in range(0, n - m + 1):
        results[i] = dist(z_norm(data[i: i + m]), query)
    return results
