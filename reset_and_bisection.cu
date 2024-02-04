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

__global__ void reset_burnin(parameters p, grids Grids, prices prices) {

	/*-------------------------------------------------------------------
	
	This function sets the values of the panel simulation to zero for 
	those indexes in the burn-in period

	---------------------------------------------------------------------*/

	float zero = 0.0f;

	// This resets all values in the burn-in period to 0 so that the reduction kernel can operate appropiately

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < *p.number_people * 1500 ; i += gridDim.x * blockDim.x) {

		Grids.ptr_random_draws_z[i] = zero;
		Grids.ptr_assets_simulation[i] = zero;
		Grids.ptr_income_simulation_capital[i] = zero;
		Grids.ptr_income_simulation_labour[i] = zero;
		Grids.ptr_consumption_simulation[i] = zero;
		Grids.ptr_SWF[i] = zero;
		Grids.ptr_SWF_value_function_simulation[i] = zero;
		Grids.ptr_indicator_zero_wealth[i] = zero;
		Grids.ptr_random_draws_z[i] = zero;
		Grids.ptr_labour_supply_simulation[i] = zero;

	}

}

__global__ void reset(parameters p, grids Grids, prices prices) {

	/*-----------------------------------------------------------
	
	This function sets the aggregates and all panel simulation grids
	to zero for the next iteration

	------------------------------------------------------------*/

	float zero = 0.0f;

	Grids.ptr_total_assets[0] = zero;
	Grids.ptr_total_consumption_simulation[0] = zero;
	Grids.ptr_SWF[0] = zero;
	Grids.ptr_total_income_simulation_capital[0] = zero;
	Grids.ptr_total_income_simulation_capital[0] = zero;
	Grids.ptr_moment_zero_wealth[0] = zero;
	Grids.ptr_total_labour_supply[0] = zero;

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < *p.number_people * *p.number_periods; i += gridDim.x * blockDim.x) {

		Grids.ptr_random_draws_z[i] = zero;
		Grids.ptr_assets_simulation[i] = zero;
		Grids.ptr_income_simulation_capital[i] = zero;
		Grids.ptr_income_simulation_labour[i] = zero;
		Grids.ptr_consumption_simulation[i] = zero;
		Grids.ptr_SWF[i] = zero;
		Grids.ptr_SWF_value_function_simulation[i] = zero;
		Grids.ptr_indicator_zero_wealth[i] = zero;
		Grids.ptr_random_draws_z[i] = zero;
		Grids.ptr_labour_supply_simulation[i] = zero;

	}
}