#pragma once


// CUDA headers

#include<cuda.h>
#include<curand.h>
#include<curand_kernel.h>

const int dimz_global = 7;


struct prices {

	/*--- This structure contains the prices required for computing the Aiyagari model ---*/

	float* r; // interest rate
	float* R; 
	float* w; // wage

};

struct parameters {

	/*--- This structure contains the main parameters required for computing the Aiyagari model ---*/

	// model parameters

	float* beta; // discount factor
	float* sigma_hh; // risk aversion
	float* delta; // depreciation rate
	float* alpha_c; // corporate sector weight capital
	float* relaxation;

	// panel simulation parameters

	int* number_people; // number of people in simulation
	int* number_periods; // number of periods in simulation

	// state space parameters

	const int* dima; // assets dimension
	const int* dimz; // idiosyncratic shocks dimension
	int* dim_total; // total points on the state space

	float* min_a; // min assets
	float* min_z; // min idiosyncratic productivity

	float* dx_a; // step size assets
	float* dx_z; // step size idiosyncratic productivity 

	// GE iterations

	int* outer_iter; // GE iterations on prices

	// burn in period

	int* burn_in;

	// tolerance convergence prices

	float* tol_r;

};

struct grids { 

	/*--- This structure contains the main grids required for computing the Aiyagari model ---*/

	/* Value and Policy functions*/

	// value function 

	float* ptr_value;

	// assets policies

	float* ptr_policy;
	float* ptr_check_policy;

	// expected value 

	float* ptr_ev;

	// upper bound for golden section search

	float* ptr_limit;

	/* Panel Simulation grids */

	// input to CUDA kernel generating random numbers

	curandState_t* ptr_states;

	// obtained random numbers

	float* ptr_random_numbers;

	// this stores the index of the idiosyncratic productivity shock in the panel simulation

	int* ptr_random_draws_z;

	// assets panel simulation

	float* ptr_assets_simulation;

	// labour supply panel simulation

	float* ptr_labour_supply_simulation;

	// consumption policy and consumption panel simulation

	float* ptr_consumption;
	float* ptr_consumption_simulation;

	// income policies 

	float* ptr_income_labour;
	float* ptr_income_capital;

	// income panel simulation

	float* ptr_income_simulation_labour;
	float* ptr_income_simulation_capital;

	// aggregates simulation

	float* ptr_total_assets;
	float* ptr_total_consumption_simulation;
	float* ptr_total_income_simulation_capital;
	float* ptr_total_income_simulation_labour;
	float* ptr_SWF;
	float* ptr_moment_zero_wealth;
	float* ptr_total_labour_supply;

	// SWF Value Function 

	float* ptr_SWF_value_function_simulation;

	// Gini

	float* ptr_gini_coefficient;

	// indicator zero wealth

	float* ptr_indicator_zero_wealth;

	// Auxiliary Vector for Sums 

	float* ptr_aux_vector_sums;

	/* State-space arrays*/

	// unidimensional grids

	float* ptr_agrid;
	float* ptr_zgrid;

	// idiosyncratic shock transition matrix

	float* ptr_pi_z;
	float* ptr_cum_sum_pi_z;

	// vectorized grids

	float* ptr_amat_vec;
	float* ptr_zmat_vec;
	float* ptr_zmat_vec_exp;

};