// C++ headers

#include <iostream>

// CUDA headers

#include<cuda.h>
#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<curand.h>
#include<crt/device_functions.h>

// Project headers

#include "gpu_functions.cuh"
#include "model_inputs.cuh"

namespace cg = cooperative_groups;

__global__ void reduction_kernel(float* aux_vector, int burn_in, int size,  float* data_vector,  float* cumulator_scalar) {
	
	/*-------------------------------------------------------------------------------

	This function reduces (sums) the elements in data_vector 
	into a single scalar (cumulator). It performs strided sums 
	at the block, grid and set of grids levels.

	- Aux_vector: float pointer of auxiliary vector for partial sums 
	- size: intenger specifying the size of the array to be reduced
	- burn_in: integer specifying elements to be ommited due to burn-in
	- data_vector: float pointer of the data array to be reduced
	- cumulator_scalar: float pointer to a scalar where the sum is stored

	----------------------------------------------------------------------------------*/

	cooperative_groups::grid_group g = cooperative_groups::this_grid();

	// set auxiliary vector to zero

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {

		aux_vector[i] = 0.0f;

	}

	g.sync();

	for (int i = blockDim.x * blockIdx.x + threadIdx.x + burn_in; i < size; i += gridDim.x * blockDim.x) {

		// initialize shared memory

		extern __shared__ float aux_shared[];

		cg::thread_block block = cg::this_thread_block();

		block.sync();

		// copy data from global memory to shared memory

		aux_shared[threadIdx.x] = data_vector[i];

		block.sync();
		
		// peform within block reduction

		for (unsigned int s = 1; s < blockDim.x; s *= 2) {
			int index = 2 * s * threadIdx.x;
			if (index  < blockDim.x && burn_in + int((i - burn_in + 1 ) / (gridDim.x * blockDim.x)) * (gridDim.x * blockDim.x) + blockIdx.x * blockDim.x + index + s < size) {

				aux_shared[index] += aux_shared[index + s];

			}

			block.sync();

		}
		
		block.sync();
		
		// write back the block-level sum to global memory

		if (threadIdx.x == 0) {

			aux_vector[i] = aux_shared[threadIdx.x];

		}

		block.sync();
	}

	g.sync();
	
	// perform sum at the grid level

	for (int s = 1; s < gridDim.x; s *= 2) {
	
		// the block-level sums in the previous steps are at the threadIdx.x = 0 in each block
		// this piece of code sums the threadIdx.x = 0 across blocks, obtaining the entire sum at the grid level

		for (int i = blockDim.x * blockIdx.x + threadIdx.x + burn_in ; i < size; i += gridDim.x * blockDim.x) {

			int s_aux = s * blockDim.x;
			int index = 2 * s_aux * blockIdx.x;

			if (threadIdx.x == 0) {

			if (index + s_aux < gridDim.x * blockDim.x) {

					aux_vector[burn_in + int((i - burn_in + 1) / (gridDim.x * blockDim.x)) * gridDim.x * blockDim.x + index] += aux_vector[burn_in + int((i - burn_in + 1) / (gridDim.x * blockDim.x)) * gridDim.x * blockDim.x + index + s_aux];

				}

			}

		}

		g.sync();

	}

	g.sync();

	// perform sum at set of grids level
	
	for (int s = 1; s < int((size + 1) / (gridDim.x * blockDim.x)) - int((burn_in + 1) / (gridDim.x * blockDim.x)); s *= 2) {

		// in ech grid, the cumulative sum is at blockDim.x * blockIdx.x + threadIdx.x = 0
		// this piece of code goes grid by grid summing it into i == burn_in

		for (int i = blockDim.x * blockIdx.x + threadIdx.x + burn_in; i < size; i += gridDim.x * blockDim.x) {

			int s_aux = s * (gridDim.x * blockDim.x);
			int index = 2 * s_aux * int((i - burn_in + 1) / (gridDim.x * blockDim.x)) + burn_in;

			if (blockDim.x * blockIdx.x + threadIdx.x == 0) {

				if (index + s_aux < size) {

					aux_vector[index] += aux_vector[index + s_aux];
				}
			}

		}

		g.sync();

	}
	
	g.sync();

	// write final result to global memory

	for (int i = blockDim.x * blockIdx.x + threadIdx.x + burn_in ; i < size; i += gridDim.x * blockDim.x) {
		if (i == burn_in) {
			atomicAdd(&cumulator_scalar[0], aux_vector[i]);
		}
	}

	g.sync();
	
}




