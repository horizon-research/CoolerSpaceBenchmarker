import numpy as np
from numba import njit


@njit
def interpolate(image1: np.ndarray, image2: np.ndarray):
    image1_linear = (image1 / 255) ** 2.2
    image2_linear = (image2 / 255) ** 2.2

    image_sum = image1_linear + image2_linear
    image_avg = image_sum / 2

    return (image_avg ** 0.4545455) * 255


