# Aiyagari (1994) Model with CUDA 

## Description

This repository contains a sample code for computing the Aiyagari (1994) model in CUDA, Nvidia's parallel computing platform that allows to employ GPUs (Graphics Processeing Units) for general-purpose computation. For a quick introduction to CUDA, please check [CUDA Introduction](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

### Why CUDA for Economic Models? 

The inherent parallel nature of many algorithms (such as Value Function Iteration, Policy Function Iteration, Endogenous Grid Method, etc.) used to solve Dynamic Stochastic General Equilibrium (DSGE) models makes CUDA a solid choice for achieving substantial speed-ups. Taking the Aiyagari model as an example, rather than processing each point (a, z) on the state space sequentially, as in a standard MATLAB implementation, CUDA enables to process all points in parallel. This is achieved by offloading the computation to the GPU and assigning each index on the state space to a CUDA Core.

### Benchmark between a standard implementation in MATLAB and CUDA

To provide a reference on the expected speed-ups with CUDA compared to a standard vectorized implementation in MATLAB (code can also be found on this repository), the figure below provides data on the execution time for 100 calls of the VFI algorithm in each language (the MATLAB implementation was set to the same conditions and parameterization). The comparison against MATLAB was made as it is the most widely used language in Economics.

<img src = https://github.com/markoirisarri/AiyagariModelCUDA/blob/master/Figures/matlab_cuda_execution_times.png width = 700>

#### Observations

* For small dimensions (dima = 100, dimz = 7) both implementations attain the solution in little time. At this point the GPU is not under full utilization.
* As the dimensionality increases, the performance gap between both implementations widens. In particular, once we reach full utilization of the GPU (dima = 10.000) we observe a relative speed-up of around x1750, with the CUDA implementation taking less than one second to perform 100 calls to the VFI while MATLAB's sequential implementation requires about 25 minutes.

In summary, CUDA provides a great opportunity to exploit the inherent parallel nature of the most common algorithms to solve DSGE models (VFI, PFI, EGM, ...). The obtained speed-ups can be particularly beneficial when performing tasks that require multiple evaluations of the model, such as calibration and optimal policy analysis. 

## Model Output

This is the obtained output on a RTX 3070 GPU:

<img src = https://github.com/markoirisarri/AiyagariModelCUDA/blob/master/Figures/modelOutput.PNG width = 700>

## Structure

The structure of the code is as follows:

CUDA Folder:
* Main.cu: main file, it allocates CPU-GPU accessible memory for the model input structures and calls update_r(), which evaluates the Aiyagari model for a given guess on the interest rate.
  * model_inputs.cuh: header file containing the declaration of the three model input structures (parameters, Grids and prices).
  * cpu_functions.cuh: header file containing the declaration of the functions that run on the CPU (these are small dimensional or sequential in nature).
    * cpu_general_functions.cu: source file containing the definition of general-purpose functions for solving DSGE models, such as the Tauchen(1986) method and grid creation.
    * calibration_functions.cu: source file containing the definition of the functions that construct the calibration targets. This version of the code only computes the Gini coefficient as an example.
    * update_r_model_solver.cu: source file containing the definition of the functions that call the ones that solve the model. model_solver() calls the functions performing the VFI, Panel Simulation and Aggregation while update_r() evaluates model_solver() at a given guess for the interest rate.
  * gpu_functions.cuh: header file containing the declaration of the functions that are offloaded to the GPU (the ones benefiting from parallel processing).
    * reset.cu: source file containing the definition of the functions that reset the panel simulation values to zero for the next iteration.
    * aggregation.cu: source file containing the definition of a reduction kernel that sums all the values of the simulated panel to construct the aggregates of the economy.
    * VFI.cu: source file containing the definition of the function that performs Value Function Iteration.
    * random_gpu_generator.cu: source file containing the definiton of a function that calls curand_uniform() to generate random numbers to obtain the realizations of the idiosyncratic shocks.
    * panel_simulation.cu: source file containing the definition of the function that performs the panel simulation of the Aiyagari model. The simulation is sequential in the time dimension and parallel in the agents dimension.

MATLAB Folder:
* Main_VFI.m, tauchen_method_1986.m and golden_section_search.m are the files to run the model counterpart in MATLAB
  
## Requirements:

* CUDA-compatible GPU
  * In particular, this code relies on the Cooperative Groups API to perform within-kernel thread synchronization. This requires compute capability 3.0+ [table](https://developer.nvidia.com/cuda-gpus) 
* CUDA toolkit
  * Can be downloaded from: [link](https://developer.nvidia.com/cuda-toolkit)
* Visual Studio 2019 or 2022
  * Please make sure to specify the correct compute capability of your GPU in project settings > CUDA C/C++ > Device > Code Generation > compute_xx, sm_xx It is set to compute_86, sm_86 for a RTX 3070 (or any RTX 3000 family card)
* Nvidia Graphics Drivers
  * Will be often installed or updated as part of the CUDA toolkit

