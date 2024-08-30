import cupy as cp
import numpy as np
from numba import njit


def interpolate(image1: np.ndarray, image2: np.ndarray):
    # Move to GPU
    image1 = cp.asarray(image1, dtype=cp.float64)
    image2 = cp.asarray(image2, dtype=cp.float64)

    image1_linear = (image1 / 255) ** 2.2
    image2_linear = (image2 / 255) ** 2.2

    image_sum = image1_linear + image2_linear
    image_avg = image_sum / 2

    return cp.asnumpy((image_avg ** 0.4545455) * 255)


