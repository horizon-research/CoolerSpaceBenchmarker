import coolerspace as cs
import sys

# Compilation Arguments
path = sys.argv[1]
shape_y = int(sys.argv[2])
shape_x = int(sys.argv[3])

# Inputs
original_illuminant = cs.create_input("original_illuminant", [1], cs.LightSpectrum)
target_illuminant = cs.create_input("target_illuminant", [1], cs.LightSpectrum)
image = cs.create_input("image", [shape_y, shape_x], cs.sRGB)

# Convert image to LMS
image_lms = cs.LMS(image)

# Calculating factor to adjust lms cones by
original_illuminant_matrix = cs.Matrix(cs.LMS(original_illuminant))
target_illuminant_matrix = cs.Matrix(cs.LMS(target_illuminant))
abc = target_illuminant_matrix / original_illuminant_matrix

# Project to diagonal matrix
identity_3x3 = cs.Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
project_to_3x3 = cs.Matrix([
    [1],
    [1],
    [1]
])
abc_3x3 = cs.matmul(project_to_3x3, abc)
abc_diagonal = abc_3x3 * identity_3x3

# Apply modulation
modulated_image_lms = cs.matmul(image_lms, abc_diagonal)
modulated_image_srgb = cs.sRGB(modulated_image_lms)

# Compilation
cs.create_output(modulated_image_srgb)
cs.compile(path)

