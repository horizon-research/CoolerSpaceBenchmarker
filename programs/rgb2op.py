import coolerspace as cs
import sys

path = sys.argv[1]
shape_y = int(sys.argv[2])
shape_x = int(sys.argv[3])

# Inputs
srgb = cs.create_input("image", [shape_y, shape_x], cs.sRGB)

# Conversion from sRGB to opRGB
op = cs.opRGB(srgb)

# Compilation
cs.create_output(op)
cs.compile(path)
