import numpy as np
from numba import njit


RGB2XYZ = np.asarray([
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

XYZ2RGB = np.linalg.inv(RGB2XYZ)

XYZ2LMS = np.asarray([
    [
        0.210576,
        -0.417076,
        0
    ],
    [
        0.855098,
        1.17726,
        0
    ],
    [
        -0.0396983,
        0.0786283,
        0.516835
    ]
])

LMS2XYZ = np.linalg.inv(XYZ2LMS)


@njit
def colorblind(image: np.ndarray, cvd: np.ndarray) -> np.ndarray:
    # Convert to LMS
    original_shape = image.shape
    image = image.reshape((-1, 3))

    image_linear = (image / 255) ** 2.2
    image_lms = image_linear @ RGB2XYZ @ LMS2XYZ

    # Colorblind simulation
    colorblind_lms = image_lms @ cvd

    # Convert back
    colorblind_linear = colorblind_lms @ LMS2XYZ @ XYZ2RGB
    return ((colorblind_linear ** 0.45455) * 255).reshape(original_shape)
