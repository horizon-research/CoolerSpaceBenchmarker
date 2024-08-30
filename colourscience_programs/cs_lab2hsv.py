import colour
import numpy as np


def lab2hsv(lab: np.ndarray):
    xyz = colour.Lab_to_XYZ(lab)
    srgb = colour.XYZ_to_sRGB(xyz)
    hsv = colour.RGB_to_HSV(srgb)
    return hsv
