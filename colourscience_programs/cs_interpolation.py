import colour
import numpy as np


def interpolate(image1: np.ndarray, image2: np.ndarray):
    image1_xyz = colour.sRGB_to_XYZ(image1)
    image2_xyz = colour.sRGB_to_XYZ(image2)

    mix_xyz = (image1_xyz + image2_xyz) / 2
    return colour.XYZ_to_sRGB(mix_xyz)
