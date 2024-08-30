import coolerspace as cs
import sys

path = sys.argv[1]
shape_y = int(sys.argv[2])
shape_x = int(sys.argv[3])

image = cs.create_input("image", [shape_y, shape_x], cs.sRGB)

hsv = cs.HSV(image)

value_adjustment = cs.Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0.5]
])

hsv_matrix = cs.Matrix(hsv)
hsv_reduced_value = cs.matmul(hsv_matrix, value_adjustment)

hsv = cs.HSV(hsv_reduced_value)
cs.create_output(hsv)

cs.compile(path)
