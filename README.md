# Aiyagari (1994) Model with CUDA 

## Description

This repository contains a sample code for computing the Aiyagari (1994) model in CUDA, Nvidia's parallel computing platform that allows to employ GPUs (Graphics Processeing Units) for general-purpose computation. For a quick introduction to CUDA, please check [CUDA Introduction](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)

### Why CUDA for Economic Models? 

The inherent parallel nature of many algorithms (such as Value Function Iteration, Policy Function Iteration, Endogenous Grid Method, etc.) used to solve Dynamic Stochastic General Equilibrium (DSGE) models makes CUDA a solid choice for achieving substantial speed-ups. Taking the Aiyagari model as an example, rather than processing each point (a, z) on the state space sequentially, as in a standard MATLAB implementation, CUDA enables to process all points in parallel. This is achieved by offloading the computation to the GPU and assigning each index on the state space to a CUDA Core.

## Model Output

This is the obtained output on a RTX 3070 GPU:

<img src = https://github.com/markoirisarri/AiyagariModelCUDA/blob/master/modelOutput.PNG width = 700>

## Structure

The structure of the code is as follows:

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
    
## Requirements:

* CUDA-compatible GPU
  * In particular, this code relies on the Cooperative Groups API to perform within-kernel thread synchronization. This requires compute capability 3.0+ [table](https://developer.nvidia.com/cuda-gpus) 
* CUDA toolkit
  * Can be downloaded from: [link](https://developer.nvidia.com/cuda-toolkit)
* Visual Studio 2019 or 2022
* Nvidia Graphics Drivers
  * Will be often installed or updated as part of the CUDA toolkit

