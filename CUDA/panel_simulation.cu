
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

__global__ void panel_simulation(const parameters p, grids Grids, prices prices) {

	/*---------------------------------------------------------
	
	This function takes as input the structures of pointers 
	in model_input.cuh and computes the variables of interest 
	in the panel simulation

	----------------------------------------------------------*/

	cg::grid_group g = cg::this_grid();

	// 1st dimension: agents
	// 2nd dimesion: time

	// registers for interpolation

	float fast_term_upper = 1.0f;
	float fast_term_lower = 1.0f;
	float interpolant;
	float dx_a = 1 / (Grids.ptr_agrid[1] - Grids.ptr_agrid[0]);

	// registers indicating the indexes during simulation

	int id = 1; // present index (i + dim_people * t), person i at time t, row major in agents
	int id_future = 1; // person i at time t+1
	int id_past = 1; // person i at time t-1

	// registers indicatig the position on the grid for the policy functions

	int index_a;
	int index_z;

	int linear_index;

	for (int t = 1; t < *p.number_periods - 1; t++) {

		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < *p.number_people; i += blockDim.x * gridDim.x) {


			id = i + *p.number_people * t;
			id_future = id + *p.number_people;
			id_past = id - *p.number_people;

			// Grids.ptr_random_draws_z is the INDEX of the current realization

			for (int index_z = 0; index_z < *p.dimz; index_z++) {

				// This employed the random numbers between 0 and 1 and gets the next realization z' based on the bin it falls 
				// in the (column) cumulative of the transition matrix

				if (Grids.ptr_random_numbers[id] > Grids.ptr_cum_sum_pi_z[index_z + (*p.dimz + 1) * Grids.ptr_random_draws_z[id]]) {

					Grids.ptr_random_draws_z[i + *p.number_people * t + *p.number_people] = index_z;

				}

			}

			// enforce that the assets are always within the domain

			Grids.ptr_assets_simulation[id] = fminf(Grids.ptr_assets_simulation[id], Grids.ptr_agrid[*p.dima - 1]);
			Grids.ptr_assets_simulation[id] = fmaxf(Grids.ptr_assets_simulation[id], Grids.ptr_agrid[0]);

			// histogram method to obtain indexes for assets

			index_a = int(((Grids.ptr_assets_simulation[id]) - *p.min_a) * dx_a);
			index_z = Grids.ptr_random_draws_z[id];

			// obtain global index

			linear_index = index_a + *p.dima * index_z;

			// weight in the interpolation to the upper grid point

			fast_term_upper = (Grids.ptr_amat_vec[index_a + 1] - (Grids.ptr_assets_simulation[id])) * dx_a;

			// weight in interpolation to the lower grid point

			fast_term_lower = 1.0f - fast_term_upper;

			/* fill the simulation grids of the variables of interest by evaluating the policy functions at (a_it,z_it) with linear interpolation  */

			// SWF

			Grids.ptr_SWF_value_function_simulation[id] = fast_term_upper * Grids.ptr_value[linear_index] +
				fast_term_lower * Grids.ptr_value[linear_index + 1];

			// Assets

			Grids.ptr_assets_simulation[id_future] = fast_term_upper * Grids.ptr_policy[linear_index]
				+ fast_term_lower * Grids.ptr_policy[linear_index + 1];

			// Indicator zero wealth

			Grids.ptr_indicator_zero_wealth[id] = (Grids.ptr_assets_simulation[id] < Grids.ptr_agrid[1]);

			// Labour Income

			Grids.ptr_income_simulation_labour[id] = fast_term_upper * Grids.ptr_income_labour[linear_index]
				+ fast_term_lower * Grids.ptr_income_labour[linear_index + 1];

			// Capital Income

			Grids.ptr_income_simulation_capital[id] = fast_term_upper * Grids.ptr_income_capital[linear_index]
				+ fast_term_lower * Grids.ptr_income_capital[linear_index + 1];

			// Consumption

			Grids.ptr_consumption_simulation[id] = fast_term_upper * Grids.ptr_consumption[linear_index]
				+ fast_term_lower * Grids.ptr_consumption[linear_index + 1];

			// Labour Supply

			Grids.ptr_labour_supply_simulation[id] = expf(Grids.ptr_zgrid[Grids.ptr_random_draws_z[id]]);

		}

	}

}
