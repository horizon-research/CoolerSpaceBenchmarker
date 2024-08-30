import numpy as np
import time
import pandas as pd
from PIL import Image

from cupy_adaptation import adapt
from cupy_interpolation import interpolate
from cupy_colorblind import colorblind
from cupy_rgb2op import rgb2op
from cupy_lab2hsv import lab2hsv
from cupy_mixing import mixing

# Settings
TRIALS = 101

# Inputs
image_2880_5120 = np.ascontiguousarray(np.asarray(Image.open("../images/turaco_5k.png")).astype(np.double)[:, :, 0:3])
image_2160_3840 = np.asarray(Image.open("../images/nightjar_4kuhd.png")).astype(np.double)[:, :, 0:3]

adaptation_original_illuminant = np.asarray([
    23.289217, 21.678232, 20.067247, 21.226551, 22.385855, 27.010095, 31.634336, 33.395136, 35.155936, 34.238822,
    33.321707, 37.926632, 42.531557, 48.077153, 53.622750, 56.375462, 59.128174, 61.194617, 63.261060, 65.987549,
    68.714039, 70.422382, 72.130726, 75.794712, 79.458699, 81.187633, 82.916567, 84.553845, 86.191122, 89.841377,
    93.491632, 94.697762, 95.903892, 97.802676, 99.701460, 99.850730, 100.000000, 99.818633, 99.637266, 101.536536,
    103.435806, 105.290685, 107.145563, 112.925216, 118.704869, 122.396039, 126.087210, 128.018091, 129.948972,
    129.644313, 129.339655, 135.804835, 142.270015, 142.339123, 142.408231, 147.687385, 152.966540, 159.272415,
    165.578290, 164.689066, 163.799843, 152.122902, 140.445960, 146.575115, 152.704269, 150.113597, 147.522925,
    134.679552, 121.836180, 128.614117, 135.392055, 139.710190, 144.028324, 132.669019, 121.309714, 106.190606,
    91.071498, 110.771973, 130.472449, 126.245983, 122.019517, 0, 0, 0, 0, 0, 0, 0, 0
])
adaptation_target_illuminant = np.asarray([
    49.428921, 51.759850, 54.090779, 68.025953, 81.961127, 86.313504, 90.665880, 91.655559, 92.645238, 89.319349,
    85.993460, 95.060474, 104.127488, 110.199807, 116.272126, 116.704928, 117.137730, 115.706664, 114.275597,
    114.837711, 115.399824, 111.895876, 108.391928, 108.703479, 109.015030, 108.268521, 107.522013, 106.057800,
    104.593588, 106.069007, 107.544427, 105.928756, 104.313086, 104.157277, 104.001468, 102.000734, 100.000000,
    98.184891, 96.369783, 96.119049, 95.868315, 92.348821, 88.829328, 89.531304, 90.233281, 90.059665, 89.886050,
    88.959839, 88.033629, 85.844284, 83.654939, 83.904045, 84.153151, 82.327113, 80.501074, 80.631971, 80.762868,
    81.835533, 82.908199, 80.915868, 78.923538, 74.590608, 70.257678, 71.238575, 72.219472, 73.564167, 74.908862,
    68.486759, 62.064656, 66.225988, 70.387321, 73.001133, 75.614945, 69.824749, 64.034552, 55.396835, 46.759117,
    57.025699, 67.292280, 65.562176, 63.832072, 0, 0, 0, 0, 0, 0, 0, 0,
])

# Functions


benchmarks = {
    "adaptation": (
        adapt,
        {
            "original_image": image_2880_5120,
            "original_illuminant": adaptation_original_illuminant,
            "target_illuminant": adaptation_target_illuminant
        }
    ),
    "interpolation": (
        interpolate,
        {
            "image1": image_2880_5120,
            "image2": image_2880_5120
        }
    ),

    "rgb2op": (
        rgb2op,
        {
           "srgb": image_2880_5120
        }
    ),

    "colorblind": (
        colorblind,
        {
            "image": image_2880_5120,
            "cvd": np.asarray([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9]
            ])
        }
    ),
    "lab2hsv": (
        lab2hsv,
        {
            "lab": image_2160_3840
        }
    ),
    "mixing": (
        mixing,
        {
            "k1": np.random.rand(600, 600, 89),
            "k2": np.random.rand(600, 600, 89),
            "s1": np.random.rand(600, 600, 89),
            "s2": np.random.rand(600, 600, 89),
            "d1": np.random.rand(600, 600, 89),
            "d2": np.random.rand(600, 600, 89),
            "light": np.random.rand(600, 600, 89),
        }
    )
}


benchmark_name_list = list()
time_list = list()


for benchmark_name, benchmark in benchmarks.items():
    for trial in range(TRIALS):
        start_time = time.time()
        benchmark[0](**benchmark[1])
        end_time = time.time()

        run_time = end_time - start_time
        time_list.append(run_time)
        benchmark_name_list.append(benchmark_name)

        print("Benchmark {}, Trial {}, Time {}".format(benchmark_name, trial, run_time))


time_df = pd.DataFrame({
    "program": benchmark_name_list,
    "time": time_list
})

time_df.to_csv("./cupy_testing.csv")
