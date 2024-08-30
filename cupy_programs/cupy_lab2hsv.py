import cupy as cp
import numpy as np
from numba import njit

RGB2XYZ = cp.asarray([
    [
        0.4124564,
        0.2126729,
        0.0193339
    ],
    [
        0.3575761,
        0.7151522,
        0.119192
    ],
    [
        0.1804375,
        0.072175,
        0.9503041
    ]
])

XYZ2RGB = cp.linalg.inv(RGB2XYZ)

epsilon = 0.008856
kappa = 903.3


def lab2hsv(lab: np.ndarray):
    # Move to GPU
    lab = cp.asarray(lab, dtype=cp.float64)

    # Flatten to appease numba
    original_shape = lab.shape
    lab = lab.reshape((-1, 3))

    # LAB to XYZ math
    # Source: http://www.brucelindbloom.com/index.html?Eqn_Lab_to_XYZ.html
    l = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]

    fy = (l + 16) / 116
    fx = fy + (a / 500)
    fz = fy - (b / 200)

    fx3 = fx ** 3
    xp = np.where(fx3 > epsilon, fx3, (116 * fx - 16) / kappa)

    fy3 = fy ** 3
    yp = np.where(l > kappa * epsilon, fy3, l / kappa)

    fz3 = fz ** 3
    zp = np.where(fz3 > epsilon, fz3, (116 * fz - 16) / kappa)

    xyz = np.stack((.9504 * xp, yp, 1.089 * zp), axis=-1)

    rgb = (xyz @ XYZ2RGB) ** 0.4545

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    # RGB to HSV
    # max and min calculations need to be this convoluted because I want to avoid branches, which would make the program
    # unnecessarily slow. Additionally, numba doesn't currently support the np.max operator with the axis option.
    hsv_v = np.maximum(np.maximum(r, g), b)
    rgb_min = np.minimum(np.minimum(r, g), b)
    hsv_c = hsv_v - rgb_min

    red_case = np.where(np.isclose(hsv_v, r), ((g - b)/hsv_c) % 6, 0)
    green_case = np.where(np.isclose(hsv_v, g), (b - r)/hsv_c + 2, 0)
    blue_case = np.where(np.isclose(hsv_v, b), (r - g)/hsv_c + 4, 0)

    hsv_h = np.where(np.isclose(hsv_c, 0), 0, red_case + green_case + blue_case)
    hsv_s = np.where(np.isclose(hsv_v, 0), 0, hsv_c/hsv_v)

    hsv = np.stack((hsv_h, hsv_s, hsv_v), axis=-1)

    return cp.asnumpy(hsv.reshape(original_shape))


