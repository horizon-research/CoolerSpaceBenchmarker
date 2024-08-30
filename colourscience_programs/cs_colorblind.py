import colour
import numpy as np


xyz_to_lms = np.asarray([
    [0.4002, 0.7076, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0, 0, 0.9182]
]).transpose()

lms_to_xyz = np.linalg.inv(xyz_to_lms)


def colorblind(image: np.ndarray, colorblind_matrix: np.ndarray):
    # Convert image from sRGB to XYZ
    xyz_image = colour.sRGB_to_XYZ(image)
    # Convert image from XYZ to LMS
    lms_image = xyz_image @ xyz_to_lms
    # Apply single-plane color blindness transformation
    lms_image_modulated = lms_image @ colorblind_matrix
    # Convert image back to sRGB and return
    xyz_image_modulated = lms_image_modulated @ lms_to_xyz
    return colour.XYZ_to_sRGB(xyz_image_modulated)
