

#define CHECK(call) \
{ \
const cudaError_t error = call; \
if (error != cudaSuccess) \
{ \
printf("Error: %s:%d, ", __FILE__, __LINE__); \
printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
exit(1); \
} \
}

// C++ headers

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

// CUDA headers

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

// Project headers

#include "cpu_functions.cuh"
#include "gpu_functions.cuh"
#include "model_inputs.cuh"

void update_r(parameters p, grids Grids, prices p_prices, void* KernelArgs[]) {

	/*-------------------------------------------------------------------
	
	This function takes as inputs the structures on the parameters, grids 
	and  prices of the model and evaluates the function model_solver()
	for a given guess on the GE interest rate 
	
	--------------------------------------------------------------------*/

	float diff = 1;
	float r_old = 1;
	int	iter = 0;

		while ((abs(diff) > *p.tol_r && iter < 30)) {

			r_old = *p_prices.r;

			model_solver(p, Grids, p_prices, KernelArgs);

			diff = (*p_prices.r - r_old);
		
			*p_prices.r = *p.relaxation * *p_prices.r + (1.0f - *p.relaxation) * r_old;
			
			iter++;
			++*p.outer_iter;

		}

		std::cout << "\n" << " Diff in interest rate is: " << diff << " General Eq. Computed, interest rate is: " << *p_prices.r << "\n";

}


void model_solver(parameters p, grids Grids, prices prices, void* KernelArgs[]) {

	/*---------------------------------------------------------------------------------------
	
	This function takes as inputs the structures on the parameters, 
	grids and prices of the model, along with the vector of arrays KernelArgs 
	required for launching the cudaLaunchCooperativeKernel() and computes the 
	optimal policies, performs the panel simulation and computes the aggregates of the model
	
	-----------------------------------------------------------------------------------------*/

	// update the prices

	*prices.w = (1.0f - *p.alpha_c) * pow(((*prices.r + *p.delta) / (*p.alpha_c)), (*p.alpha_c / (*p.alpha_c - 1.0f)));

	*prices.R = *prices.r + *p.delta;

	// these are parameters for the optimal launch configuration of the CUDA kernels

	int BLOCKS3 = 0;
	int THREADS3 = 0;

	int BLOCKS3_aux = 0;
	int THREADS3_aux = 0;

	// set the dim3 objects for all kernels

	//reset kernel
	cudaOccupancyMaxPotentialBlockSize(&BLOCKS3_aux, &THREADS3_aux, reset, 0, 0);
	dim3 dimBlock3_reset(THREADS3_aux, 1, 1);
	dim3 dimGrid3_reset(BLOCKS3_aux, 1, 1);

	// vfi kernel
	cudaOccupancyMaxPotentialBlockSize(&BLOCKS3_aux, &THREADS3_aux, VFI, 0, 0);
	dim3 dimBlock3_vfi(THREADS3_aux, 1, 1);
	dim3 dimGrid3_vfi(BLOCKS3_aux, 1, 1);

	// panel simulation kernel
	cudaOccupancyMaxPotentialBlockSize(&BLOCKS3_aux, &THREADS3_aux, panel_simulation, 0, 0);
	dim3 dimBlock3_panel_simulation(THREADS3_aux, 1, 1);
	dim3 dimGrid3_panel_simulation(BLOCKS3_aux, 1, 1);

	// this specifies the range of values for which the summation should be done 

	int reduction_start_index = (*p.number_periods - *p.burn_in) * *p.number_people;
	int reduction_end_index = (*p.number_periods - 1) * *p.number_people;
	
	// this is a boolean indicating whether the user wants to check the error in the 
	// VFI's policy function

	bool check_vfi_error = false;
	bool naive_benchmark_vfi = false;
	int benchmark_iterations = 100; // set naive benchmark to true to obtain the results in the speed-ups figure

	// these are locals for aggregate labour, aggregate capital and aggregate output

	volatile float L_c = 0;
	volatile float K_c = 1;
	float Y = 0;

	// start by reseting all the aggregates and panel simulation grids from the previous iteration

	cudaLaunchCooperativeKernel((void*)reset, dimGrid3_reset, dimBlock3_reset, KernelArgs);
	CHECK(cudaDeviceSynchronize());

	// perform the VFI with Howard Improvement, in order to obtain the Value and Policy functions

	if (naive_benchmark_vfi == false) {

		cudaLaunchCooperativeKernel((void*)VFI, dimGrid3_vfi, dimBlock3_vfi, KernelArgs);
		CHECK(cudaDeviceSynchronize());

	}
	else {

		// set same prices as in Matlab Code 

		*prices.r = 0.02;
		*prices.w = 2;

		auto start = std::chrono::steady_clock::now();

		for (int i = 0; i < benchmark_iterations; i++) {


			// Here please note we are being conservative on the speed-up 
			// since we are synchronizing CPU-GPU memory at every call of
			// VFI with cudaDeviceSynchronize()

			// For more rigurous profiling and being able to identify bottlenecks in the kernel one can use Nsight Compute

			cudaLaunchCooperativeKernel((void*)VFI, dimGrid3_vfi, dimBlock3_vfi, KernelArgs);
			CHECK(cudaDeviceSynchronize());

		}

		auto end = std::chrono::steady_clock::now();

		std::chrono::duration<float> elapsed_seconds = end - start;
		std::cout << "\n Elapsed time for " << benchmark_iterations << " calls to VFI: " << elapsed_seconds.count() << " seconds. \n";

		std::cout << "\n \n Naive Benchmark completed, terminating execution"; 

		exit(0);

	}

	// check the error in the policies if desired

	if (check_vfi_error == true) {

		float sum_error = 0;
		float max_error = 0;

		for (int i = 0; i < *p.dim_total; i++) {

			sum_error += fabsf(Grids.ptr_policy[i] - Grids.ptr_check_policy[i]);
			max_error = fmaxf(max_error, fabsf(Grids.ptr_policy[i] - Grids.ptr_check_policy[i]));
		}

		std::cout << "\n Total Error in VFI: " << sum_error << " Max Abs. Error Policies" << max_error << "\n";

	}

	// perform the panel simulation with the obtained policy functions

	cudaLaunchCooperativeKernel((void*)panel_simulation, dimGrid3_panel_simulation, dimBlock3_panel_simulation, KernelArgs);
	CHECK(cudaDeviceSynchronize());
		
	// set the values for the indexes in the burn-in period to zero so that they are not summed

	cudaLaunchCooperativeKernel((void*)reset_burnin, dimGrid3_reset, dimBlock3_reset, KernelArgs);
	cudaDeviceSynchronize();

	// perform reduction on the obtained series for the variables of interest 

	cudaOccupancyMaxPotentialBlockSize(&BLOCKS3, &THREADS3, reduction_kernel, THREADS3*sizeof(float), 0);
	CHECK(cudaDeviceSynchronize());

	// Assets
	reduction_kernel <<<BLOCKS3,THREADS3,THREADS3*sizeof(float)>>> (Grids.ptr_aux_vector_sums, reduction_start_index, reduction_end_index, Grids.ptr_assets_simulation, Grids.ptr_total_assets);
	cudaDeviceSynchronize();
	// Consumption
	reduction_kernel <<<BLOCKS3,THREADS3,THREADS3*sizeof(float)>>> (Grids.ptr_aux_vector_sums, reduction_start_index, reduction_end_index, Grids.ptr_consumption_simulation, Grids.ptr_total_consumption_simulation);
	cudaDeviceSynchronize();
	// Labour Income
	reduction_kernel <<<BLOCKS3,THREADS3,THREADS3*sizeof(float)>>> (Grids.ptr_aux_vector_sums, reduction_start_index, reduction_end_index, Grids.ptr_income_simulation_labour, Grids.ptr_total_income_simulation_labour);
	cudaDeviceSynchronize();
	// Capital Income
	reduction_kernel <<<BLOCKS3,THREADS3,THREADS3*sizeof(float)>>> (Grids.ptr_aux_vector_sums, reduction_start_index, reduction_end_index, Grids.ptr_income_simulation_capital, Grids.ptr_total_income_simulation_capital);
	cudaDeviceSynchronize();
	// Moment Zero Wealth
	reduction_kernel <<<BLOCKS3,THREADS3,THREADS3*sizeof(float)>>> (Grids.ptr_aux_vector_sums, reduction_start_index, reduction_end_index, Grids.ptr_indicator_zero_wealth, Grids.ptr_moment_zero_wealth);
	cudaDeviceSynchronize();
	// SWf
	reduction_kernel <<<BLOCKS3,THREADS3,THREADS3*sizeof(float)>>> (Grids.ptr_aux_vector_sums, reduction_start_index, reduction_end_index, Grids.ptr_SWF_value_function_simulation, Grids.ptr_SWF);
	cudaDeviceSynchronize();
	// Labour Supply
	reduction_kernel <<<BLOCKS3,THREADS3,THREADS3*sizeof(float)>>> (Grids.ptr_aux_vector_sums, reduction_start_index, reduction_end_index, Grids.ptr_labour_supply_simulation, Grids.ptr_total_labour_supply);
	cudaDeviceSynchronize();

	// compute the aggregates from the obtained cumulative sums across periods and agents 

	Grids.ptr_total_assets[0] = Grids.ptr_total_assets[0] / (reduction_end_index - reduction_start_index);
	Grids.ptr_total_consumption_simulation[0] = Grids.ptr_total_consumption_simulation[0] / (reduction_end_index - reduction_start_index);
	Grids.ptr_total_income_simulation_capital[0] = Grids.ptr_total_income_simulation_capital[0] / (reduction_end_index - reduction_start_index);
	Grids.ptr_total_income_simulation_labour[0] = Grids.ptr_total_income_simulation_labour[0] / (reduction_end_index - reduction_start_index);
	Grids.ptr_moment_zero_wealth[0] = Grids.ptr_moment_zero_wealth[0] / (reduction_end_index - reduction_start_index);
	Grids.ptr_total_labour_supply[0] = Grids.ptr_total_labour_supply[0] / (reduction_end_index - reduction_start_index);
	Grids.ptr_SWF[0] = Grids.ptr_SWF[0] / (reduction_end_index - reduction_start_index);

	// obtain aggregate capital, labour and production 

	K_c = Grids.ptr_total_assets[0];
	L_c = Grids.ptr_total_labour_supply[0];
	Y = pow(K_c, *p.alpha_c) * pow(L_c, 1.0f - *p.alpha_c);

	// compute the Gini coefficient

	compute_gini(p, Grids, prices);
	CHECK(cudaDeviceSynchronize());

	// update the interest rate 

	*prices.r = (*p.alpha_c * pow((K_c/L_c), *p.alpha_c - 1.0f) - *p.delta);
	*prices.r = (*prices.r > 0 && (*prices.r < 1 / *p.beta - 1 + 0.2f)) * *prices.r + (*prices.r <= 0) * *prices.r + (*prices.r >= 1 / *p.beta - 1 + 0.2f) * (1 / *p.beta - 1 + 0.2f);

	// output information on the iteration

	std::cout << "\n Iteration: " << *p.outer_iter << " Total Capital: " << K_c << " Interest rate: " << *prices.r << "\n";


}

