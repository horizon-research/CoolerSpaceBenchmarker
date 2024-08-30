import coolerspace as cs
import sys

# Compilation arguments
path = sys.argv[1]
shape_y = int(sys.argv[2])
shape_x = int(sys.argv[3])

# Inputs
image1 = cs.create_input("image1", [shape_y, shape_x], cs.sRGB)
image2 = cs.create_input("image2", [shape_y, shape_x], cs.sRGB)

# Interpolate between the two images by half
mixed = image1 * 0.5 + image2 * 0.5

# Compilation
cs.create_output(mixed)
cs.compile(path)
