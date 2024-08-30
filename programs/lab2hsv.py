import coolerspace as cs
import sys

path = sys.argv[1]
shape_y = int(sys.argv[2])
shape_x = int(sys.argv[3])

# Inputs
lab = cs.create_input("image", [shape_y, shape_x], cs.LAB)

# Simple conversion
hsv = cs.HSV(lab)

# Compilation
cs.create_output(hsv)
cs.compile(path)

