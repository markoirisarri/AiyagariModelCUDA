#pragma once

// C++ headers

#include <iostream>

// CUDA headers

#include<cuda.h>
#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<curand.h>
#include<curand_kernel.h>
#include<crt/device_functions.h>

// Project headers

#include "model_inputs.cuh"

// This header files contains the function declarations for the functions that run on the GPU

__global__ void generate_random_gpu(parameters p, grids Grids, prices p_prices); // generates random numbers on GPU

__global__ void VFI_optimized(parameters p, grids Grids, prices prices); // performs the VFI of the Aiyagari model

__global__ void panel_simulation(const parameters p, grids Grids, prices prices); // simulates the panel once the policy functions are obtained

__global__ void reset(parameters p, grids Grids, prices prices); // resets the values for the panel simulation to zero for the next simulation

__global__ void reset_burnin(parameters p, grids Grids, prices prices); // sets the values in the burn-in period to zero 

__global__ void reduction_kernel(float* aux_vector, int burn_in, int size,  float* data_vector,  float* cumulator_scalar); // this kernel sums all the values in the simulation, allowing to compute the aggregates









