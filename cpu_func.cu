// C++ headers

#include <iostream>
#include <chrono>
#include <random>

// Project headers

#include "cpu_functions.cuh"

inline float CDFSTDNormal(float x)
{
	//Function CDFSTDNormal: computes the standard normal CDF using Abramowiz and Stegun (1964) approximation

	// constants
	float a1 = 0.2548295;
	float a2 = -0.2844967;
	float a3 = 1.4214137;
	float a4 = -1.4531520;
	float a5 = 1.0614054;
	float p = 0.32759;

	// Save the sign of x
	int sign = 1;
	if (x < 0)
		sign = -1;
	x = fabs(x) / sqrt(2.0);

	// A&S formula 7.1.26
	float t = 1.0f / (1.0f + p * x);
	float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * expf(-x * x);

	return 0.5 * (1.0f + sign * y);


};

void tauchen(const int dim, const float mean, const float rho, const float sigma, float* grid, float* Pi) {

	/*---------------------------------------------------------------------------
	
	This function discretizes an AR(1) process following Tauchen (1986)'s method

	dim (input) = integer specifying dimension of grid for productivity
	mean (input) = mean of the AR(1) process
	rho (input) = autocorrelation of the AR(1) process
	sigma (input) = std of the shock
	grid (output) = grid containing the discretized values for z 
	Pi(output) = transition matrix (z,z')

	-----------------------------------------------------------------------------*/

	float* brackets;
	cudaMallocManaged(&brackets, (dim + 1) * sizeof(float));
	float* mat_tauchen;
	cudaMallocManaged(&mat_tauchen, dim * (dim + 1) * sizeof(float));
	float* brackets_mat;
	cudaMallocManaged(&brackets_mat, dim * (dim + 1) * sizeof(float));

	const int bandwith = 3;

	// Obtain volatility of the AR1 process

	const float sigma_productivity = sigma / sqrt(1.0f - rho * rho);

	// Create the grid for the process

	const float min = mean - bandwith * sigma_productivity;
	const float max = mean + bandwith * sigma_productivity;
	const float dx_z = (max - min) / (dim - 1);


	for (int i = 0; i < dim; i++) {

		grid[i] = min + dx_z * i;

	}

	// create transition matrix

	brackets[dim] = +99999;
	brackets[0] = -99999;

	for (int i = 1; i < dim; i++) {

		brackets[i] = 0.5 * (grid[i] + grid[i - 1]);

	}

	for (int j = 0; j < dim + 1; j++) {
		for (int i = 0; i < dim; i++) {

			mat_tauchen[i * (dim + 1) + j] = grid[i];
			brackets_mat[i * (dim + 1) + j] = brackets[j];


		}
	}

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			Pi[i * dim + j] = CDFSTDNormal((brackets_mat[i * (dim + 1) + j + 1] - rho * mat_tauchen[i * (dim + 1) + j + 1] - (1.0f - rho) * mean) / (sigma)) - CDFSTDNormal((brackets_mat[i * (dim + 1) + j] - rho * mat_tauchen[i * (dim + 1) + j] - (1.0f - rho) * mean) / (sigma));

		}

	}

	cudaFree(brackets);
	cudaFree(brackets_mat);
	cudaFree(mat_tauchen);

};

void ergodic_matrix(const int dim, float* aux_vector, float* Pi, float* Pi_ergodic) {

	// initial guess (uniform)

	for (int i = 0; i < dim; i++) {

		Pi_ergodic[i] = float(1.0f / float(dim));
		aux_vector[i] = 0;
	}

	// Here main loop to construct the matrices

	for (int i = 0; i < 200; i++) {

		// outer loop

		for (int h = 0; h < dim; h++) {

			// inner loop

			for (int hh = 0; hh < dim; hh++) {

				aux_vector[h] += Pi[hh * dim + h] * Pi_ergodic[hh];

			}


		}

		// update Pi_ergodic

		for (int h = 0; h < dim; h++) {

			Pi_ergodic[h] = aux_vector[h];
			aux_vector[h] = 0;
		}

	}

}

void get_cum_sum_matrices(float* cum_sum_pi_z, float* pi_z,  const int dimz) {

	for (int i = 0; i < dimz; i++) {
		for (int j = 0; j < dimz + 1; j++) {

			if (j == 0) {

				cum_sum_pi_z[j] = 0;

			}

			else if (j == 1) {

				cum_sum_pi_z[j + (dimz + 1) * i] = pi_z[j - 1 + dimz * i];

			}
			else {

				cum_sum_pi_z[j + (dimz + 1) * i] = pi_z[j - 1 + dimz * i] + cum_sum_pi_z[(j - 1) + (dimz + 1) * i];


			}
		}
	}

}

void vectorize_grids(float* row_vec, float* column_vec, float* row_grid, float* column_grid, const int dim_row, const int dim_column) {

	// row-major convention

		for (int j = 0; j < dim_column; j++) {
			for (int i = 0; i < dim_row; i++) {

				row_vec[i + dim_row * j] = row_grid[i];
				column_vec[i + dim_row * j] = column_grid[j];
			}
		}

};

void vectorize_exp_grids(float* vec_1, float* vec_2, float* vec_1_exp, float* vec_2_exp, const int dim_grid) {

	for (int i = 0; i < dim_grid; i++) {

		vec_1_exp[i] = expf(vec_1[i]);
		vec_2_exp[i] = expf(vec_2[i]);

	}

}

void vectorize_exp_grids_1d(float* vec_1,float* vec_1_exp, const int dim_grid) {

	for (int i = 0; i < dim_grid; i++) {

		vec_1_exp[i] = expf(vec_1[i]);

	}

}
