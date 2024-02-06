#pragma once

// C++ headers

#include <iostream>

// Project headers

#include "model_inputs.cuh"

// Function declarations 

inline float CDFSTDNormal(float x);

void tauchen(const int dim, const float mean, const float rho, const float sigma, float* zgrid, float* Pi);

void ergodic_matrix(const int dim, float* aux_vector, float* Pi, float* Pi_ergodic);

void get_cum_sum_matrices(float* cum_sum_pi_z,  float* pi_z,  const int dimz);

void vectorize_grids(float* row_vec, float* column_vec, float* row_grid, float* column_grid, const int dim_row, const int dim_column);

void vectorize_exp_grids(float* vec_1, float* vec_2, float* vec_1_exp, float* vec_2_exp, const int);

void vectorize_exp_grids_1d(float* vec_1, float* vec_1_exp, const int dim_grid);

void compute_gini(parameters p, grids Grids, prices prices);

void merge(float* arr, int p, int q, int r);

void mergeSort(float* arr, int l, int r);

void update_r(parameters p, grids Grids, prices p_prices, void* KernelArgs[]);

void model_solver(parameters p, grids Grids, prices p_prices, void* KernelArgs[]);
