import colour
import numpy as np


def adapt(original_image: np.ndarray, original_illuminant: np.ndarray, target_illuminant: np.ndarray):
    original_spectral = colour.SpectralDistribution(original_illuminant, range(390, 835, 5))
    target_spectral = colour.SpectralDistribution(target_illuminant, range(390, 835, 5))

    original_xyz = colour.colorimetry.sd_to_XYZ(original_spectral, k=1/683)
    target_xyz = colour.colorimetry.sd_to_XYZ(target_spectral, k=1/683)

    # Convert image to XYZ
    image_xyz = colour.sRGB_to_XYZ(original_image, chromatic_adaptation_transform="Von Kries")

    # Perform chromatic adaptation
    modulated_image = colour.adaptation.chromatic_adaptation(image_xyz, original_xyz, target_xyz)

    return modulated_image
