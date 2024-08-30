import coolerspace as cs
import sys

path = sys.argv[1]
shape_y = int(sys.argv[2])
shape_x = int(sys.argv[3])

# Inputs
image = cs.create_input("image", [shape_y, shape_x], cs.sRGB)
colorblind_matrix = cs.create_input("colorblind_matrix", [3, 3], cs.Matrix)

# Convert image to LMS
image_lms = cs.LMS(image)

# Apply colorblindness matrix
colorblind_image_lms = cs.matmul(image_lms, colorblind_matrix)

# Convert back
colorblind_image = cs.sRGB(colorblind_image_lms)

# Compilation
cs.create_output(colorblind_image)
cs.compile(path)
