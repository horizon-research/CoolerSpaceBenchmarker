import pandas as pd
import numpy as np
import onnxruntime as ort
import time
import math
import datetime
import os
from PIL import Image

from onnxruntime.capi.onnxruntime_pybind11_state import ExecutionMode

# Trial count
TRIAL_COUNT = 101

# region Input vectors
# Images
image_1080_1920 = np.asarray(Image.open("images/myna_fhd.png")).astype(np.double)[:, :, 0:3]
image_1440_2560 = np.asarray(Image.open("images/warbler_qhd.png")).astype(np.double)[:, :, 0:3]
image_1800_3200 = np.asarray(Image.open("images/kingfisher_qhdp.png")).astype(np.double)[:, :, 0:3]
image_2160_3840 = np.asarray(Image.open("images/nightjar_4kuhd.png")).astype(np.double)[:, :, 0:3]
image_2880_5120 = np.asarray(Image.open("images/turaco_5k.png")).astype(np.double)[:, :, 0:3]
# Adaptation
adaptation_original_illuminant = np.asarray([[
    23.289217, 21.678232, 20.067247, 21.226551, 22.385855, 27.010095, 31.634336, 33.395136, 35.155936, 34.238822, 33.321707, 37.926632, 42.531557, 48.077153, 53.622750, 56.375462, 59.128174, 61.194617, 63.261060, 65.987549, 68.714039, 70.422382, 72.130726, 75.794712, 79.458699, 81.187633, 82.916567, 84.553845, 86.191122, 89.841377, 93.491632, 94.697762, 95.903892, 97.802676, 99.701460, 99.850730, 100.000000, 99.818633, 99.637266, 101.536536, 103.435806, 105.290685, 107.145563, 112.925216, 118.704869, 122.396039, 126.087210, 128.018091, 129.948972, 129.644313, 129.339655, 135.804835, 142.270015, 142.339123, 142.408231, 147.687385, 152.966540, 159.272415, 165.578290, 164.689066, 163.799843, 152.122902, 140.445960, 146.575115, 152.704269, 150.113597, 147.522925, 134.679552, 121.836180, 128.614117, 135.392055, 139.710190, 144.028324, 132.669019, 121.309714, 106.190606, 91.071498, 110.771973, 130.472449, 126.245983, 122.019517, 0, 0, 0, 0, 0, 0, 0, 0
]])
adaptation_target_illuminant = np.asarray([[
    49.428921, 51.759850, 54.090779, 68.025953, 81.961127, 86.313504, 90.665880, 91.655559, 92.645238, 89.319349, 85.993460, 95.060474, 104.127488, 110.199807, 116.272126, 116.704928, 117.137730, 115.706664, 114.275597, 114.837711, 115.399824, 111.895876, 108.391928, 108.703479, 109.015030, 108.268521, 107.522013, 106.057800, 104.593588, 106.069007, 107.544427, 105.928756, 104.313086, 104.157277, 104.001468, 102.000734, 100.000000, 98.184891, 96.369783, 96.119049, 95.868315, 92.348821, 88.829328, 89.531304, 90.233281, 90.059665, 89.886050, 88.959839, 88.033629, 85.844284, 83.654939, 83.904045, 84.153151, 82.327113, 80.501074, 80.631971, 80.762868, 81.835533, 82.908199, 80.915868, 78.923538, 74.590608, 70.257678, 71.238575, 72.219472, 73.564167, 74.908862, 68.486759, 62.064656, 66.225988, 70.387321, 73.001133, 75.614945, 69.824749, 64.034552, 55.396835, 46.759117, 57.025699, 67.292280, 65.562176, 63.832072, 0, 0, 0, 0, 0, 0, 0, 0,
]])


# endregion


# region TestCase data structure
class TestCase:
    def __init__(self,
                 shape: list[int],
                 gpu: bool,
                 cpu: bool,
                 inputs: dict[str, np.ndarray]):
        self.shape = shape
        self.gpu = gpu
        self.cpu = cpu
        self.inputs = inputs


# endregion


# region Test case database
cases = {
    ("adaptation", "programs/adaptation.py"): [
        TestCase(
            shape=[1080, 1920],
            gpu=True,
            cpu=True,
            inputs={
                "original_illuminant": adaptation_original_illuminant,
                "target_illuminant": adaptation_target_illuminant,
                "image": image_1080_1920
            }
        ),
        TestCase(
            shape=[1440, 2560],
            gpu=True,
            cpu=True,
            inputs={
                "original_illuminant": adaptation_original_illuminant,
                "target_illuminant": adaptation_target_illuminant,
                "image": image_1440_2560
            }
        ),
        TestCase(
            shape=[1800, 3200],
            gpu=True,
            cpu=True,
            inputs={
                "original_illuminant": adaptation_original_illuminant,
                "target_illuminant": adaptation_target_illuminant,
                "image": image_1800_3200
            }
        ),
        TestCase(
            shape=[2160, 3840],
            gpu=True,
            cpu=True,
            inputs={
                "original_illuminant": adaptation_original_illuminant,
                "target_illuminant": adaptation_target_illuminant,
                "image": image_2160_3840
            }
        ),
        TestCase(
            shape=[2880, 5120],
            gpu=True,
            cpu=True,
            inputs={
                "original_illuminant": adaptation_original_illuminant,
                "target_illuminant": adaptation_target_illuminant,
                "image": image_2880_5120
            }
        )
    ],
    ("rgb2op", "programs/rgb2op.py"): [
        TestCase(
            shape=[1080, 1920],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_1080_1920
            }
        ),
        TestCase(
            shape=[1440, 2560],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_1440_2560
            }
        ),
        TestCase(
            shape=[1800, 3200],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_1800_3200
            }
        ),
        TestCase(
            shape=[2160, 3840],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_2160_3840
            }
        ),
        TestCase(
            shape=[2880, 5120],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_2880_5120
            }
        )
    ],
    ("colorblind", "programs/colorblind.py"): [
        TestCase(
            shape=[1080, 1920],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_1080_1920,
                "colorblind_matrix": np.asarray(
                    [
                        [0, 0, 0],
                        [2.02344, 1, 0],
                        [-2.52581, 0, 1]
                    ]
                )
            }
        ),
        TestCase(
            shape=[1440, 2560],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_1440_2560,
                "colorblind_matrix": np.asarray(
                    [
                        [0, 0, 0],
                        [2.02344, 1, 0],
                        [-2.52581, 0, 1]
                    ]
                )
            }
        ),
        TestCase(
            shape=[1800, 3200],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_1800_3200,
                "colorblind_matrix": np.asarray(
                    [
                        [0, 0, 0],
                        [2.02344, 1, 0],
                        [-2.52581, 0, 1]
                    ]
                )
            }
        ),
        TestCase(
            shape=[2160, 3840],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_2160_3840,
                "colorblind_matrix": np.asarray(
                    [
                        [0, 0, 0],
                        [2.02344, 1, 0],
                        [-2.52581, 0, 1]
                    ]
                )
            }
        ),
        TestCase(
            shape=[2880, 5120],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_2880_5120,
                "colorblind_matrix": np.asarray(
                    [
                        [0, 0, 0],
                        [2.02344, 1, 0],
                        [-2.52581, 0, 1]
                    ]
                )
            }
        )
    ],
    ("interpolation", "programs/interpolation.py"): [
        TestCase(
            shape=[1080, 1920],
            gpu=True,
            cpu=True,
            inputs={
                "image1": image_1080_1920,
                "image2": np.random.rand(1080, 1920, 3)
            }
        ),
        TestCase(
            shape=[1440, 2560],
            gpu=True,
            cpu=True,
            inputs={
                "image1": image_1440_2560,
                "image2": np.random.rand(1440, 2560, 3)
            }
        ),
        TestCase(
            shape=[1800, 3200],
            gpu=True,
            cpu=True,
            inputs={
                "image1": image_1800_3200,
                "image2": np.random.rand(1800, 3200, 3)
            }
        ),
        TestCase(
            shape=[2160, 3840],
            gpu=True,
            cpu=True,
            inputs={
                "image1": image_2160_3840,
                "image2": np.random.rand(2160, 3840, 3)
            }
        ),
        TestCase(
            shape=[2880, 5120],
            gpu=True,
            cpu=True,
            inputs={
                "image1": image_2880_5120,
                "image2": np.random.rand(2880, 5120, 3)
            }
        )
    ],
    ("mixing", "programs/mixing.py"): [
        TestCase(
            shape=[200, 200],
            gpu=True,
            cpu=True,
            inputs={
                "scattering1": np.random.rand(200, 200, 89),
                "scattering2": np.random.rand(200, 200, 89),
                "absorption1": np.random.rand(200, 200, 89),
                "absorption2": np.random.rand(200, 200, 89),
                "light": np.random.rand(200, 200, 89),
                "density1": np.random.rand(200, 200, 1),
                "density2": np.random.rand(200, 200, 1)
            }
        ),
        TestCase(
            shape=[400, 400],
            gpu=True,
            cpu=True,
            inputs={
                "scattering1": np.random.rand(400, 400, 89),
                "scattering2": np.random.rand(400, 400, 89),
                "absorption1": np.random.rand(400, 400, 89),
                "absorption2": np.random.rand(400, 400, 89),
                "light": np.random.rand(400, 400, 89),
                "density1": np.random.rand(400, 400, 1),
                "density2": np.random.rand(400, 400, 1)
            }
        ),
        TestCase(
            shape=[600, 600],
            gpu=True,
            cpu=True,
            inputs={
                "scattering1": np.random.rand(600, 600, 89),
                "scattering2": np.random.rand(600, 600, 89),
                "absorption1": np.random.rand(600, 600, 89),
                "absorption2": np.random.rand(600, 600, 89),
                "light": np.random.rand(600, 600, 89),
                "density1": np.random.rand(600, 600, 1),
                "density2": np.random.rand(600, 600, 1)
            }
        ),

    ],
    ("lab2hsv", "programs/lab2hsv.py"): [
        TestCase(
            shape=[1080, 1920],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_1080_1920
            }
        ),
        TestCase(
            shape=[1440, 2560],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_1440_2560
            }
        ),
        TestCase(
            shape=[1800, 3200],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_1800_3200
            }
        ),
        TestCase(
            shape=[2160, 3840],
            gpu=True,
            cpu=True,
            inputs={
                "image": image_2160_3840
            }
        ),
    ]
}


# endregion


# region Path calculation
def get_compiled_path(test_name: str, test_shape: list[int]) -> str:
    return "compiled/" + test_name + "-" + "_".join([str(dim) for dim in test_shape]) + ".onnx"


def get_optimized_path(test_name: str, test_shape: list[int]) -> str:
    return "optimized/" + test_name + "-" + "_".join([str(dim) for dim in test_shape]) + ".onnx"


def get_compiled_ort_path(test_name: str, test_shape: list[int]) -> str:
    return "compiled/" + test_name + "-" + "_".join([str(dim) for dim in test_shape]) + ".ort"


def get_optimized_ort_path(test_name: str, test_shape: list[int]) -> str:
    return "optimized/" + test_name + "-" + "_".join([str(dim) for dim in test_shape]) + ".ort"



def get_compile_data_path(time_string: str) -> str:
    return "results/compile/" + time_string + ".csv"


def get_optimize_data_path(time_string: str) -> str:
    return "results/optimize/" + time_string + ".csv"


def get_run_data_path(time_string: str) -> str:
    return "results/run/" + time_string + ".csv"


# endregion


# region Compilation benchmarking function
# Dataframe for compilation times
compile_data = pd.DataFrame(columns=['test', 'y', 'x', 'trial', 'time'])


# Function to benchmark compilation time for a single test case
def benchmark_compile(test_name: str, test_program: str, test_case: TestCase):
    output_path = get_compiled_path(test_name, test_case.shape)
    for trial in range(1):
        # Delete compiled file if it already exists
        try:
            os.remove(output_path)
        except OSError:
            pass

        # Perform compilation benchmark
        start_time = time.time()
        os.system(" ".join(["taskset -c 4", "python3", test_program, output_path, str(test_case.shape[0]), str(test_case.shape[1])]))
        end_time = time.time()

        # Ensure that the compiled file was actually generated before recording time
        if os.path.isfile(output_path):
            print(test_name, " by ".join([str(dim) for dim in test_case.shape]), "compilation trial", trial, "success")
            compile_time = end_time - start_time
            compile_data.loc[len(compile_data)] = pd.Series({
                'test': test_name,
                'y': test_case.shape[0],
                'x': test_case.shape[1],
                'trial': trial,
                'time': compile_time
            })
        else:
            print(test_name, " by ".join([str(dim) for dim in test_case.shape]), "compilation trial", trial, "FAIL")
            break


# endregion


# region Optimization benchmarking function
# Dataframe for optimization times
optimize_data = pd.DataFrame(columns=['test', 'y', 'x', 'trial', 'time'])


# Function to benchmark optimization time for a single test case
def benchmark_optimize(test_name: str, test_case: TestCase):
    compiled_path = get_compiled_path(test_name, test_case.shape)
    optimized_path = get_optimized_path(test_name, test_case.shape)

    # Ensure that the compiled path exists, otherwise abort
    if not os.path.isfile(compiled_path):
        print(test_name, " by ".join([str(dim) for dim in test_case.shape]),
              "optimization benchmarking failed due to missing compiled executable")
        return

    for trial in range(1):
        # Delete optimized file if it already exists
        try:
            os.remove(optimized_path)
        except OSError:
            pass

        # Benchmark optimization
        start_time = time.time()
        os.system(" ".join(["taskset -c 4", "python3 -m onneggs", compiled_path, optimized_path]))
        end_time = time.time()

        # Ensure that the optimized file was actually generated before recording time
        if not os.path.isfile(optimized_path):
            print(test_name, " by ".join([str(dim) for dim in test_case.shape]), "optimization trial", trial,
                  "failed due to missing output")
            break

        print(test_name, " by ".join([str(dim) for dim in test_case.shape]), "optimization trial", trial, "succeeded")
        optimize_time = end_time - start_time
        optimize_data.loc[len(optimize_data)] = pd.Series({
            'test': test_name,
            'y': test_case.shape[0],
            'x': test_case.shape[1],
            'trial': trial,
            'time': optimize_time
        })


# endregion


# region Running benchmarking function
run_data = pd.DataFrame(columns=['test', 'y', 'x', 'processor', 'executable', 'trial', 'time'])


def benchmark_run(test_name: str, test_case: TestCase):
    unoptimized_path = get_compiled_ort_path(test_name, test_case.shape)
    optimized_path = get_optimized_ort_path(test_name, test_case.shape)

    # Ensure that the executable paths exist, otherwise abort
    if not os.path.isfile(unoptimized_path):
        print(test_name, " by ".join([str(dim) for dim in test_case.shape]),
              "running benchmarking failed due to missing unoptimized executable")
        return

    if not os.path.isfile(optimized_path):
        print(test_name, " by ".join([str(dim) for dim in test_case.shape]),
              "running benchmarking failed due to missing optimized executable")
        return

    # CPU benchmarking
    if test_case.cpu:
        session_options = ort.SessionOptions()
        session_options.execution_mode = ExecutionMode.ORT_PARALLEL

        cpu_unoptimized_session = ort.InferenceSession(
            unoptimized_path,
            providers=["CPUExecutionProvider"]
        )
        cpu_optimized_session = ort.InferenceSession(
            optimized_path,
            providers=["CPUExecutionProvider"]
        )

        for trial in range(TRIAL_COUNT):
            results = benchmark_run_single_trial(cpu_unoptimized_session, cpu_optimized_session, test_case.inputs)

            if results is False:
                print(test_name, " by ".join([str(dim) for dim in test_case.shape]), "running cpu trial", trial,
                      "failed due to output mismatch")
                break

            unoptimized_time, optimized_time = results
            run_data.loc[len(run_data)] = pd.Series({
                'test': test_name,
                'y': test_case.shape[0],
                'x': test_case.shape[1],
                'processor': "cpu",
                'executable': "unoptimized",
                'trial': trial,
                'time': unoptimized_time
            })
            run_data.loc[len(run_data)] = pd.Series({
                'test': test_name,
                'y': test_case.shape[0],
                'x': test_case.shape[1],
                'processor': "cpu",
                'executable': "optimized",
                'trial': trial,
                'time': optimized_time
            })

            print(test_name, " by ".join([str(dim) for dim in test_case.shape]), "cpu running trial", trial,
                  "succeeded with", unoptimized_time, "unoptimized and", optimized_time, "optimized")

    # GPU benchmarking
    if test_case.gpu:
        session_options = ort.SessionOptions()
        session_options.execution_mode = ExecutionMode.ORT_PARALLEL

        gpu_unoptimized_session = ort.InferenceSession(
            unoptimized_path,
            providers=["CUDAExecutionProvider"]
        )
        gpu_optimized_session = ort.InferenceSession(
            optimized_path,
            providers=["CUDAExecutionProvider"]
        )

        for trial in range(TRIAL_COUNT):
            results = benchmark_run_single_trial(gpu_unoptimized_session, gpu_optimized_session, test_case.inputs)

            if results is False:
                print(test_name, " by ".join([str(dim) for dim in test_case.shape]), "running gpu trial", trial,
                      "failed due to unoptimized optimized output mismatch")
                break

            unoptimized_time, optimized_time = results
            run_data.loc[len(run_data)] = pd.Series({
                'test': test_name,
                'y': test_case.shape[0],
                'x': test_case.shape[1],
                'processor': "gpu",
                'executable': "unoptimized",
                'trial': trial,
                'time': unoptimized_time
            })
            run_data.loc[len(run_data)] = pd.Series({
                'test': test_name,
                'y': test_case.shape[0],
                'x': test_case.shape[1],
                'processor': "gpu",
                'executable': "optimized",
                'trial': trial,
                'time': optimized_time
            })

            print(test_name, " by ".join([str(dim) for dim in test_case.shape]), "gpu running trial", trial,
                  "succeeded with", unoptimized_time, "unoptimized and", optimized_time, "optimized")


def benchmark_run_single_trial(
        unoptimized_session: ort.InferenceSession,
        optimized_session: ort.InferenceSession,
        inputs: dict
):
    # Benchmark unoptimized
    unoptimized_start_time = time.time()
    unoptimized_output = unoptimized_session.run(None, inputs)
    unoptimized_end_time = time.time()

    # Benchmark optimized
    optimized_start_time = time.time()
    optimized_output = optimized_session.run(None, inputs)
    optimized_end_time = time.time()

    # Check to ensure equality
    unoptimized_output_numpy = np.nan_to_num(np.asarray(unoptimized_output))
    optimized_output_numpy = np.nan_to_num(np.asarray(optimized_output))

    if not np.isclose(np.nan_to_num(unoptimized_output_numpy), np.nan_to_num(optimized_output_numpy)).all():
        print(np.asarray(unoptimized_output)[0, 0])
        print(np.asarray(optimized_output)[0, 0])
        return False

    return unoptimized_end_time - unoptimized_start_time, optimized_end_time - optimized_start_time


# endregion


# region Main function
def main():
    # Compilation benchmarking
    for (test_name, test_function), test_cases in cases.items():
        for test_case in test_cases:
            benchmark_compile(test_name, test_function, test_case)

    # Optimization benchmarking
    for (test_name, test_function), test_cases in cases.items():
        for test_case in test_cases:
            benchmark_optimize(test_name, test_case)

    # Running benchmarking
    for (test_name, test_function), test_cases in cases.items():
        for test_case in test_cases:
            benchmark_run(test_name, test_case)

    # Output results to csv files
    now = datetime.datetime.now()
    current_time_string = now.strftime("%m-%d-%Y-%H-%M-%S")

    compile_data.to_csv(get_compile_data_path(current_time_string), index=False)
    optimize_data.to_csv(get_optimize_data_path(current_time_string), index=False)
    run_data.to_csv(get_run_data_path(current_time_string), index=False)


# Main boilerplate
if __name__ == "__main__":
    main()
# endregion
