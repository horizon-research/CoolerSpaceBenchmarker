# CoolerSpace Benchmarker
This is the benchmarker for the CoolerSpace library.
The CoolerSpace GitHub repository can be found [here](https://github.com/horizon-research/CoolerSpace).

## Image Credits
We have used several Unsplash photographs to benchmark the efficacy of CoolerSpace.
Please refer to [this file](images/credits.md) for links to the original photographs and credits.

## What's included in this repository?
We have included 4 different benchmarker programs in this repository.
1. [benchmarker.py](benchmarker.py): This program benchmarks compilation time, optimization time, and runtime of CoolerSpace on several programs in the [programs directory](programs)
2. [numba_programs/benchmark.py](numba_programs/benchmark.py): This program benchmarks equivalent programs written in numba
3. [cupy_programs/benchmark.py](cupy_programs/benchmark.py): This program benchmarks equivalent programs written in CuPy
4. [colourscience_programs/benchmark.py](colourscience_programs/benchmark.py): This program benchmarks equivalent programs written in ColourScience

The `cases` variable in each of these benchmark programs can be modified to choose which programs to benchmark at what resolutions.
The `TRIAL_COUNT` variable is used to set the number of trials that are run for every test case.

## Dependencies
Nvidia CUDA dependencies are required depending on which version of ONNX Runtime you have installed.
Please refer to [this website](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) to look up which version of CUDA and cuDNN you need to run [benchmarker.py](benchmarker.py).
