

// C++ headers

#include <iostream>

// CUDA headers

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Project headers

#include "gpu_functions.cuh"
#include "model_inputs.cuh"

__global__ void generate_random_gpu(parameters p, grids Grids, prices prices) {

	/*--------------------------------------------------------------------------------------
	
	This function generates random numbers between 0 and 1 using curand.h's curand_uniform()
	Initializing the Grids.ptr_states is required for this to work

	----------------------------------------------------------------------------------------*/

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < *p.number_people * *p.number_periods; i += blockDim.x * gridDim.x) {

		curand_init(i, i, i, &Grids.ptr_states[i]);
		Grids.ptr_random_numbers[i] = curand_uniform(&Grids.ptr_states[i]);

	}

}