// standard C++ headers

#include <iostream>
#include <chrono>

// CUDA headers

#include<cuda.h>
#include<cuda_runtime.h>

// Project headers

#include "cpu_functions.cuh"
#include "gpu_functions.cuh"
#include "model_inputs.cuh"

// Error checking function for CUDA kernels (only applicable on CudaDeviceSynchronize() )

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

void main() {

	/*-------------------------------------------------------------------------------
	 This section initializes all the necessary arrays to compute the Aiyagari model 

	 General procedure:

	 - Initialize a pointer
	 - Call cudaMallocManaged() to allocate memory accessible to both CPU and GPU 
	 - Assign the address of the pointer to the pointers in the model_inputs.cuh structures

	--------------------------------------------------------------------------------*/

	// state space dimensions

	const int dima = 100000; // dimension for assets
	const int dimz = dimz_global; // dimension for idiosyncratic productivity
	const int dim_total = dima * dimz; // total dimension

	// specify min and max values for assets grid

	const float min_a = (0.01);
	const float max_a = (29.99 + min_a);
	const float dx_a = (max_a - min_a) / (dima - 1); // step between elememnts on grid

	// Assign pointers to grids of assets, E productivity and labour productivity

	float* agrid;
	float* zgrid;

	// cudaMallocManaged allocates memory accessible to both CPU and GPU 
	// a cudaDeviceSynchronize() call is required for data to be 
	// synchronized across CPU and GPU after a modification is done

	cudaMallocManaged(&agrid, dima * sizeof(float));
	cudaMallocManaged(&zgrid, dimz * sizeof(float));

	// Create linear grid for assets

	for (int i = 0; i < dima; i++) {

		agrid[i] = min_a + dx_a * i;
	}

	// Specify AR(1) process for productivity

	float mean_z = 0;
	float rho_z = 0.9;
	float sigma_z = 0.08717;

	// Allocate memory for transition and ergodic
	// matrices for idiosyncratic productivity

	float* Pi_z;
	float* Pi_z_ergodic;
	float* aux_vector;

	cudaMallocManaged(&Pi_z, dimz * dimz * sizeof(float));
	cudaMallocManaged(&Pi_z_ergodic, dimz * sizeof(float));
	cudaMallocManaged(&aux_vector, dimz * sizeof(float));

	// Vectorize the grids and create linear index (a,z,x)

	float* amat_vec;
	float* zmat_vec;
	float* zmat_vec_exp;

	cudaMallocManaged(&amat_vec, dima * dimz * sizeof(float));
	cudaMallocManaged(&zmat_vec, dima * dimz * sizeof(float));
	cudaMallocManaged(&zmat_vec_exp, dima * dimz * sizeof(float));

	// Allocate memory for the cumulative matrix

	float* cum_sum_pi_z;

	cudaMallocManaged(&cum_sum_pi_z, (dimz + 1) * dimz * sizeof(float));

	/*--- Set up the parameters structure ---*/

	parameters p; // initialize parameters

	/* Main model parameters */

	float* beta;
	float* sigma_hh;
	float* delta;
	float* alpha_c;
	float* iter_r;
	float* relaxation;

	cudaMallocManaged(&beta, sizeof(float));
	cudaMallocManaged(&sigma_hh, sizeof(float));
	cudaMallocManaged(&delta, sizeof(float));
	cudaMallocManaged(&alpha_c, sizeof(float));
	cudaMallocManaged(&iter_r, sizeof(float));
	cudaMallocManaged(&relaxation, sizeof(float));

	beta[0] = 0.96; // discount factor 
	sigma_hh[0] = 3; // risk aversion 
	delta[0] = 0.08; // depreciation rate
	alpha_c[0] = 0.36; // intensity of capital in production function
	relaxation[0] = 0.1; // relaxation parameter

	p.beta = beta; // discount factor
	p.sigma_hh = sigma_hh; // risk aversion
	p.delta = delta; // depreciation rate
	p.alpha_c = alpha_c; // corporate sector weight capital
	p.relaxation = relaxation;

	/* Parameters for panel simulation */

	int* number_people; // number of agents in simulation
	int* number_periods; // number of periods in simulation
	int* burn_in; // burn_in periods

	cudaMallocManaged(&number_people, sizeof(int));
	cudaMallocManaged(&number_periods, sizeof(int));
	cudaMallocManaged(&burn_in, sizeof(int));

	number_people[0] = 3000;
	number_periods[0] = 2000;
	burn_in[0] = 500;

	p.number_people = number_people;
	p.number_periods = number_periods;
	p.burn_in = burn_in;

	/* Grid Parameters*/

	int* dima_cuda;
	int* dimz_cuda;
	int* dim_total_cuda;

	cudaMallocManaged(&dima_cuda, sizeof(int));
	cudaMallocManaged(&dimz_cuda, sizeof(int));
	cudaMallocManaged(&dim_total_cuda, sizeof(int));

	dima_cuda[0] = dima;
	dimz_cuda[0] = dimz;
	dim_total_cuda[0] = dim_total;

	float* min_a_cuda;
	float* min_z_cuda;

	cudaMallocManaged(&min_a_cuda, sizeof(float));
	cudaMallocManaged(&min_z_cuda, sizeof(float));

	min_a_cuda[0] = agrid[0];
	min_z_cuda[0] = zgrid[0];

	float* dx_a_cuda;
	float* dx_z_cuda;

	cudaMallocManaged(&dx_a_cuda, sizeof(float));
	cudaMallocManaged(&dx_z_cuda, sizeof(float));

	dx_a_cuda[0] = agrid[1] - agrid[0];
	dx_z_cuda[0] = zgrid[1] - zgrid[0];

	p.dima = dima_cuda;
	p.dimz = dimz_cuda;
	p.dim_total = dim_total_cuda;

	p.min_a = min_a_cuda;
	p.min_z = min_z_cuda;

	p.dx_a = dx_a_cuda;
	p.dx_z = dx_a_cuda;

	/* Convergence criteria parameters */

	int* outer_iter;

	cudaMallocManaged(&outer_iter, sizeof(int));

	p.outer_iter = outer_iter;

	/*--- Set up the prices structure  ---*/ 

	prices p_prices; // initialize vector of pricse

	float* r;
	float* R;
	float* w;

	cudaMallocManaged(&r, sizeof(float));
	cudaMallocManaged(&R, sizeof(float));
	cudaMallocManaged(&w, sizeof(float));

	r[0] = 1;
	R[0] = 1;
	w[0] = 1;

	p_prices.r = r;
	p_prices.R = R;
	p_prices.w = w;

	/*--- Set up the grids structure  ---*/

	grids Grids; 

	/* Value and Policy functions*/

	// initialize value function

	float* value;

	cudaMallocManaged(&value, dim_total * sizeof(float));

	Grids.ptr_value = value;

	// initialize policies savings

	float* policy;
	float* check_policy;
	
	cudaMallocManaged(&policy, *p.dim_total * sizeof(float));
	cudaMallocManaged(&check_policy, *p.dim_total * sizeof(float));

	Grids.ptr_policy = policy;
	Grids.ptr_check_policy = check_policy;

	// initialize expected value

	float* ev;

	cudaMallocManaged(&ev, dim_total * sizeof(float));

	Grids.ptr_ev = ev;

	// initialize golden section search upper bound

	float* limit;

	cudaMallocManaged(&limit, dim_total * sizeof(float));

	Grids.ptr_limit = limit;

	/* Panel Simulation grids */

	// input to CUDA kernel generating random numbers

	curandState_t* states;
	cudaMallocManaged(&states, *p.number_people** p.number_periods * sizeof(curandState_t));

	Grids.ptr_states = states;

	// obtained random numbers

	float* random_numbers;
	cudaMallocManaged(&random_numbers, *p.number_people** p.number_periods * sizeof(float));

	Grids.ptr_random_numbers = random_numbers;

	// this stores the index of the idiosyncratic productivity shock in the panel simulation

	int* random_draws_z;
	cudaMallocManaged(&random_draws_z, *p.number_people** p.number_periods * sizeof(int));

	Grids.ptr_random_draws_z = random_draws_z;

	// assets panel simulation

	float* assets_simulation;
	
	cudaMallocManaged(&assets_simulation, *p.number_people** p.number_periods * sizeof(float));

	Grids.ptr_assets_simulation = assets_simulation;

	// labour supply panel simulation

	float* labour_supply_simulation;

	cudaMallocManaged(&labour_supply_simulation, *p.number_periods** p.number_people * sizeof(float));

	Grids.ptr_labour_supply_simulation = labour_supply_simulation;

	// consumption policy and consumption panel simulation

	float* consumption;
	float* consumption_simulation;

	cudaMallocManaged(&consumption, *p.dim_total * sizeof(float));
	cudaMallocManaged(&consumption_simulation, *p.number_people** p.number_periods * sizeof(float));

	Grids.ptr_consumption = consumption;
	Grids.ptr_consumption_simulation = consumption_simulation;

	// income policies 

	float* income_labour;
	float* income_capital;

	cudaMallocManaged(&income_labour, *p.dim_total * sizeof(float));
	cudaMallocManaged(&income_capital, *p.dim_total * sizeof(float));

	Grids.ptr_income_labour = income_labour;
	Grids.ptr_income_capital = income_capital;

	// income panel simulation

	float* income_simulation_labour;
	float* income_simulation_capital;
	
	cudaMallocManaged(&income_simulation_labour, *p.number_people** p.number_periods * sizeof(float));
	cudaMallocManaged(&income_simulation_capital, *p.number_people** p.number_periods * sizeof(float));

	Grids.ptr_income_simulation_capital = income_simulation_capital;
	Grids.ptr_income_simulation_labour = income_simulation_labour;


	// aggregates simulation

	float* total_assets;
	float* total_consumption_simulation;
	float* total_income_simulation_capital;
	float* total_income_simulation_labour;
	float* SWF;
	float* moment_zero_wealth;
	float* total_labour_supply;

	cudaMallocManaged(&total_assets, sizeof(float));
	cudaMallocManaged(&total_consumption_simulation, sizeof(float));
	cudaMallocManaged(&total_income_simulation_capital, sizeof(float));
	cudaMallocManaged(&total_income_simulation_labour, sizeof(float));
	cudaMallocManaged(&SWF, sizeof(float));
	cudaMallocManaged(&moment_zero_wealth, sizeof(float));
	cudaMallocManaged(&total_labour_supply, sizeof(float));

	Grids.ptr_total_assets = total_assets;
	Grids.ptr_SWF = SWF;
	Grids.ptr_total_consumption_simulation = total_consumption_simulation;
	Grids.ptr_total_income_simulation_capital = total_income_simulation_capital;
	Grids.ptr_total_income_simulation_labour = total_income_simulation_labour;
	Grids.ptr_moment_zero_wealth = moment_zero_wealth;
	Grids.ptr_total_labour_supply = total_labour_supply;

	// SWF Value Function 

	float* SWF_value_function_simulation;

	cudaMallocManaged(&SWF_value_function_simulation, *p.number_people** p.number_periods * sizeof(float));

	Grids.ptr_SWF_value_function_simulation = SWF_value_function_simulation;

	// Gini

	float* gini_coefficient;

	cudaMallocManaged(&gini_coefficient, sizeof(float));

	Grids.ptr_gini_coefficient = gini_coefficient;

	// indicator zero wealth

	float* indicator_zero_wealth;

	cudaMallocManaged(&indicator_zero_wealth, *p.number_people** p.number_periods * sizeof(float));
	
	Grids.ptr_indicator_zero_wealth = indicator_zero_wealth;

	// Auxiliary Vector for Sums 

	float* aux_vector_sums;

	cudaMallocManaged(&aux_vector_sums, *p.number_people** p.number_periods * sizeof(float));

	Grids.ptr_aux_vector_sums = aux_vector_sums;

	/* State-space arrays*/

	// unidimensional grids

	Grids.ptr_agrid = agrid;
	Grids.ptr_zgrid = zgrid;

	// idiosyncratic shock transition matrix

	Grids.ptr_pi_z = Pi_z;
	Grids.ptr_cum_sum_pi_z = cum_sum_pi_z;

	// vectorized grids

	Grids.ptr_amat_vec = amat_vec;
	Grids.ptr_zmat_vec = zmat_vec;
	Grids.ptr_zmat_vec_exp = zmat_vec_exp;


	// Vector of pointers for functions that use the Cooperative Groups library

	void* KernelArgs[] = {

		(void*)&p,
		(void*)&Grids,
		(void*)&p_prices,

	};

	// First we need to generator random numbers for panel simulation (pre-allocate them)
	// Takes a while in debug mode
	// Blocks and Threads launch parameters for CUDA kernels

	int Blocks = 0;
	int Threads = 0;

	// Generate Random Numbers

	cudaOccupancyMaxPotentialBlockSize(&Blocks, &Threads, generate_random_gpu, 0, 0);
	generate_random_gpu << < Blocks, Threads >> > (p, Grids, p_prices);
	CHECK(cudaDeviceSynchronize());

	
	/*-------------------------------------
		This section solves for the SS GE 
	--------------------------------------*/

	// Discretize AR(1) process for idiosyncratic productivity following Tauchen (1986)

	tauchen(dimz, mean_z, rho_z, sigma_z, zgrid, Pi_z);

	*p.min_z = zgrid[0];

	// Obtain ergodic matrix for the idiosyncratic shocks

	ergodic_matrix(dimz, aux_vector, Pi_z, Pi_z_ergodic);
	
	// Obtain cumulative matrix across columns to generate 
	// the random shock index in the panel simulation

	get_cum_sum_matrices(cum_sum_pi_z, Pi_z, dimz);

	// vectorize the assets and idiosyncratic shock unidimensional grids

	vectorize_grids(amat_vec, zmat_vec, agrid, zgrid, dima, dimz);

	// exponentiate the idiosyncratic shocks vectorized grid to compute labour income

	vectorize_exp_grids_1d(zmat_vec, zmat_vec_exp, dim_total);

	// specify initial guess for the interest rate

	*p_prices.r = 0.0;

	// call the function evaluating the model for a given guess on the interest rate

	auto start = std::chrono::steady_clock::now();

	// Print model parameterization

	std::cout << " Solving the Aiyagari (1994) model with the following parameterizaton: " << "\n" <<
		"\n" << "			Assets dimension: " << *p.dima <<
		"\n" << "			Productivity shock dimension: " << *p.dimz <<
		"\n" << "			Beta: " << *p.beta <<
		"\n" << "			Sigma: " << *p.sigma_hh <<
		"\n" << "			Alpha: " << *p.alpha_c <<
		"\n" << "			Delta: " << *p.delta <<
		"\n" << "			Persistence shocks: " << rho_z <<
		"\n" << "			Std. shocks: " << sigma_z <<
		"\n";


	update_r(p, Grids, p_prices,KernelArgs);

	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<float> elapsed_seconds = end - start;
	std::cout << "\n Model Computed in: " << elapsed_seconds.count() << " seconds. \n";

	// Free allocated memory

	cudaFree(agrid);
	cudaFree(zgrid);
	cudaFree(Pi_z);
	cudaFree(Pi_z_ergodic);
	cudaFree(aux_vector);
	cudaFree(amat_vec);
	cudaFree(zmat_vec);
	cudaFree(zmat_vec_exp);
	cudaFree(cum_sum_pi_z);
	cudaFree(beta);
	cudaFree(sigma_hh);
	cudaFree(delta);
	cudaFree(alpha_c);
	cudaFree(iter_r);
	cudaFree(relaxation);
	cudaFree(number_people);
	cudaFree(number_periods);
	cudaFree(burn_in);
	cudaFree(dima_cuda);
	cudaFree(dimz_cuda);
	cudaFree(dim_total_cuda);
	cudaFree(min_a_cuda);
	cudaFree(min_z_cuda);
	cudaFree(dx_a_cuda);
	cudaFree(dx_z_cuda);
	cudaFree(outer_iter);
	cudaFree(r);
	cudaFree(R);
	cudaFree(w);
	cudaFree(value);
	cudaFree(policy);
	cudaFree(check_policy);
	cudaFree(ev);
	cudaFree(limit);
	cudaFree(states);
	cudaFree(random_numbers);
	cudaFree(random_draws_z);
	cudaFree(assets_simulation);
	cudaFree(labour_supply_simulation);
	cudaFree(consumption);
	cudaFree(consumption_simulation);
	cudaFree(income_labour);
	cudaFree(income_capital);
	cudaFree(income_simulation_labour);
	cudaFree(income_simulation_capital);
	cudaFree(total_assets);
	cudaFree(total_consumption_simulation);
	cudaFree(total_income_simulation_capital);
	cudaFree(total_income_simulation_labour);
	cudaFree(SWF);
	cudaFree(moment_zero_wealth);
	cudaFree(total_labour_supply);
	cudaFree(SWF_value_function_simulation);
	cudaFree(gini_coefficient);
	cudaFree(indicator_zero_wealth);
	cudaFree(aux_vector_sums);


}

