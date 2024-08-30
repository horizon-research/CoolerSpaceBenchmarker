import cupy as cp
import numpy as np
from numba import njit

sRGB2XYZ = cp.asarray([
    [0.4124564, 0.2126729, 0.0193339],
    [0.3575761, 0.7151522, 0.119192],
    [0.1804375, 0.072175, 0.9503041]
])

XYZ2sRGB = cp.linalg.inv(sRGB2XYZ)

opRGB2XYZ = cp.asarray([
    [0.57667, 0.18556, 0.18823],
    [0.29734, 0.62736, 0.07529],
    [0.02703, 0.07069, 0.99134]
]).transpose()

XYZ2opRGB = cp.linalg.inv(opRGB2XYZ)


def rgb2op(srgb: np.ndarray):
    # Move to GPU
    srgb = cp.asarray(srgb, dtype=cp.float64)

    original_shape = srgb.shape
    srgb = srgb.reshape((-1, 3))
    xyz = ((srgb / 255) ** 2.2) @ sRGB2XYZ.copy()
    return cp.asnumpy((((xyz @ XYZ2opRGB.copy()) ** 0.4545455) * 255).reshape(original_shape))
