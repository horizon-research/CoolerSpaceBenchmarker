import colour
import numpy as np


def rgb2op(rgb: np.ndarray):
    xyz = colour.sRGB_to_XYZ(rgb)
    oprgb = colour.XYZ_to_RGB(xyz, colourspace=colour.models.RGB_COLOURSPACE_ADOBE_RGB1998)
    return oprgb
