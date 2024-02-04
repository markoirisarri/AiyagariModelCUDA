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

__global__ void VFI_optimized(const parameters p, grids Grids, prices prices) {

	/*----------------------------------------------------------------------
	
	This function takes as inputs the structures of pointers in model_inputs.cuh
	and computes the value and policy functions 

	-----------------------------------------------------------------------*/

	// Declare groups of threads using cooperative_groups library

	cg::grid_group g = cg::this_grid();

	cg::thread_block b = cg::this_thread_block();

	// Initialiaze shared memory 
	// Note: dimz_global is global so that 
	// the shared memory has a defined size at compile time

	 __shared__ float pi_z[dimz_global * dimz_global];

	for (int i = 0; i < dimz_global * dimz_global ; i++) {

		pi_z[i] =  Grids.ptr_pi_z[i];

	}
	
	// Synchronize blocks after initializing shared memory

	b.sync();

	// Initialize registers // 

	// Prices

	float r_interest_rate = *prices.r;
	float r_wage = *prices.w;

	// Interpolation

	float interpolant;

	// Convergence 

	float iter = 0;
	float tol = 1e-7f;

	// Linear indexes

	int counter;
	int index_a;
	int index_z;
	int linear_index;

	g.sync();

	// Pre-compute the maximum level of a' for each point on the state space

	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < *p.dim_total; i += gridDim.x * blockDim.x) {

		// Pre-compute max assets for golden section maximization 

		Grids.ptr_limit[i] = fminf((1 + r_interest_rate) * ((Grids.ptr_amat_vec[i])) 
			+ r_wage * Grids.ptr_zmat_vec_exp[i], Grids.ptr_agrid[*p.dima-1] - 0.001f);

		// Initialize the expected continuation value to some random values

		Grids.ptr_ev[i] = Grids.ptr_amat_vec[i] / 100.0f ;

	}

	// Synchronize all threads

	g.sync();
	
	/*--- VFI ---*/ 

	// h is max number of iterations (1000)

	for (int h = 0; h < 1000; h++) {

		g.sync();

		// This stores the policy function at the 980th iteration to check convergence

		if (h == 980) {

			for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < *p.dim_total; i += gridDim.x * blockDim.x) {

				Grids.ptr_check_policy[i] = Grids.ptr_policy[i];

			}

		}

		g.sync();

		// This implements a Howard Improvement, maximization is done every 15 iterations
		// and the first 16

		if (h <= 15 || (h > 15 && h % 15 == 0)) {

			// This solves the Bellman equation for the agents in the economy

			for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < *p.dim_total; i += gridDim.x * blockDim.x) {

				// Initialize registers for the golden section search

				int dima = *p.dima;
				int dimz = *p.dimz;
				float min = (*p.min_a); 
				float d = 1; 
				float alpha_1 = 1; 
				float alpha_2 = 1;
				float x1 = 1;
				float x2 = 1;
				float f1 = 1;
				float f2 = 1;
				float x2_old = 1,
				float x1_old = 1;
				float f1_old = 1;
				float f2_old = 2;
				float dx_a = 1 / (Grids.ptr_agrid[2] - Grids.ptr_agrid[1]);
				float sigma = *p.sigma_hh;
				float beta = *p.beta;
				float min_a = (*p.min_a);
				float max = Grids.ptr_limit[i];

				// check that max is always geq than min

				max = max * (max > min) + min * (max <= min);

				// perform first iteration of the golden section search

				d = max - min;
				alpha_1 = (3.0f - sqrtf(5.0f)) / 2.0f;
				alpha_2 = 1.0f - alpha_1;

				x1 = min + alpha_1 * d;
				x2 = min + alpha_2 * d;

				// histogram method to get linear index of policy a'

				index_a = int(((x1)-*p.min_a) * dx_a); // index on assets 
				index_z = int((Grids.ptr_zmat_vec[i] + 0.001f - *p.min_z) / (Grids.ptr_zgrid[1] - Grids.ptr_zgrid[0])); // index on z productivity
				linear_index = index_a + dima * index_z; // index on (A,Z) state space (row-major)

				// precompute the componenet of the Bellman equation independent of a'

				float precompute = (1 + r_interest_rate) * ((Grids.ptr_amat_vec[i]))
					+  r_wage * Grids.ptr_zmat_vec_exp[i];

				// perform linear interpolation on expected value

				interpolant = (Grids.ptr_amat_vec[index_a + 1] - (x1)) * dx_a * Grids.ptr_ev[linear_index]
					+ (1.0f - (Grids.ptr_amat_vec[index_a + 1] - (x1)) * dx_a) * Grids.ptr_ev[linear_index + 1];

				// compute f1 (function evaluation at candidate x1)

				f1 = (powf((precompute - x1), 1.0f - sigma) / (1.0f - sigma)
					+ beta * interpolant);

				// perform same steps to obtain f2

				index_a = int(((x2)-*p.min_a) * dx_a);

				linear_index = index_a + dima * index_z;

				interpolant = (Grids.ptr_amat_vec[index_a + 1] - (x2)) * dx_a * Grids.ptr_ev[linear_index]
					+ (1.0f - (Grids.ptr_amat_vec[index_a + 1] - (x2)) * dx_a) * Grids.ptr_ev[linear_index + 1];

				precompute = (1 + r_interest_rate) * ((Grids.ptr_amat_vec[i]))
					+  r_wage * Grids.ptr_zmat_vec_exp[i];

				f2 = (powf((precompute - x2), 1.0f - sigma) / (1.0f - sigma)
					+ beta * interpolant);

				// now inner loop golden section search

				d = alpha_1 * alpha_2 * d;

				float ev;
				float ev_next;

				float global_a = 1;
				float global_a_next = 1;

				int counter = 0;

				while (fabsf(x1 - x2) > tol && counter < 100) {


					x2_old = x2;
					x1_old = x1;
					f1_old = f1;
					f2_old = f2;

					// Update values based on which (f2, f1) is greater

					max = max * (f2_old > f1_old) + x2_old * (f2_old <= f1_old);
					min = x1_old * (f2_old > f1_old) + min * (f2_old <= f1_old);
					x1 = x2_old * (f2_old > f1_old) + (max - alpha_2 * (max - min)) * (f2_old <= f1_old);
					x2 = (min + alpha_2 * (max - min)) * (f2_old > f1_old) + x1_old * (f2_old <= f1_old);

					index_a = int(((x2)-min_a) * dx_a) * (f2_old > f1_old) + int(((x1)-min_a) * dx_a) * (f2_old <= f1_old);

					global_a = Grids.ptr_amat_vec[index_a];
					global_a_next = Grids.ptr_amat_vec[index_a + 1];

					linear_index = index_a + dima * index_z;

					ev = Grids.ptr_ev[linear_index];
					ev_next = Grids.ptr_ev[linear_index + 1];

					// Construct linear interpolants

					interpolant = ((global_a_next - (x2)) * dx_a * ev
						+ (1.0f - (global_a_next - (x2)) * dx_a) * ev_next) * (f2_old > f1_old) + ((global_a_next - (x1)) * dx_a * ev
							+ (1.0f - (global_a_next - (x1)) * dx_a) * ev_next) * (f2_old <= f1_old);

					precompute = (1 + r_interest_rate) * ((Grids.ptr_amat_vec[i]))
						+  r_wage * Grids.ptr_zmat_vec_exp[i];

					// evaluate Bellman equation at each candidate x1,x2

					f2 = (powf((precompute - x2), 1.0f - sigma) / (1.0f - sigma)
						+ beta * interpolant) * (f2_old > f1_old) + f1_old * (f2_old <= f1_old);
					f1 = (powf((precompute - x1), 1.0f - sigma) / (1.0f - sigma)
						+ beta * interpolant) * (f2_old <= f1_old) + f2_old * (f2_old > f1_old);

					counter++;

				}

				Grids.ptr_policy[i] = x1;
				Grids.ptr_value[i] = f1;

			}

			// synchronize the entire grid of threads after performing the maximizaiton step 

			g.sync();

			// this block of code computes the expected continuation value at each point on the state space

			for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < *p.dim_total; i += gridDim.x * blockDim.x) {

				int dima = *p.dima;
				int dimz = *p.dimz;

				float aux_expectation = 0.0f;

				float value = 0.0f;
				float trans = 0.0f;

				// histogram method to obtain indexes on each dimension

				index_a = int((Grids.ptr_amat_vec[i] + 0.001f - *p.min_a) / (Grids.ptr_agrid[1] - Grids.ptr_agrid[0]));
				index_z = int((Grids.ptr_zmat_vec[i] + 0.001f - *p.min_z) / (Grids.ptr_zgrid[1] - Grids.ptr_zgrid[0]));

				// for each point on the state space, cycle through each possible z' realization

				for (int zz = 0; zz < *p.dimz; zz++) {

					value = Grids.ptr_value[index_a + dima * zz];

					trans = pi_z[dimz * index_z + zz];

					aux_expectation += trans * value;

				}

				Grids.ptr_ev[i] = 0.5f * (aux_expectation) + 0.5f * Grids.ptr_ev[i]; // perform an update with a relaxation parameter (0.5 here)

			}

			g.sync();

		}else{

			// here update values given policies (save max operator)

			for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < *p.dim_total; i += gridDim.x * blockDim.x) {

				// initialize registers

				int dima = *p.dima;
				int dimz = *p.dimz;
				float dx_a = 1 / (Grids.ptr_agrid[2] - Grids.ptr_agrid[1]);
				float sigma = *p.sigma_hh;
				float beta = *p.beta;
				float min_a = *p.min_a;
			
				// get linear index of policy a'

				index_a = int(((Grids.ptr_policy[i]) - *p.min_a) * dx_a);
				index_z = int((Grids.ptr_zmat_vec[i] + 0.001f - *p.min_z) / (Grids.ptr_zgrid[1] - Grids.ptr_zgrid[0]));
				linear_index = index_a + dima * index_z;

				float precompute = (1 + r_interest_rate) * ((Grids.ptr_amat_vec[i])) 
					+  r_wage * Grids.ptr_zmat_vec_exp[i];

				// get interpolant

				interpolant = (Grids.ptr_amat_vec[index_a + 1] - (Grids.ptr_policy[i])) * dx_a * Grids.ptr_ev[linear_index]
					+ (1.0f - (Grids.ptr_amat_vec[index_a + 1] - (Grids.ptr_policy[i])) * dx_a) * Grids.ptr_ev[linear_index + 1];

				// compute the value with the given policy a'(a,z)

				Grids.ptr_value[i] = (powf((precompute - Grids.ptr_policy[i]), 1.0f - sigma) / (1.0f - sigma)
					+ beta * interpolant);

			}
		
			// synchronize all threads after updating the value function with the given policy function

			g.sync();


			// this block of code computes the expected continuation value at each point on the state space

			for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < *p.dim_total; i += gridDim.x * blockDim.x) {

				int dima = *p.dima;
				int dimz = *p.dimz;

				float aux_expectation = 0.0f;

				float value = 0.0f;
				float trans = 0.0f;

				// histogram method to obtain indexes on each dimension

				index_a = int((Grids.ptr_amat_vec[i] + 0.001f - *p.min_a) / (Grids.ptr_agrid[1] - Grids.ptr_agrid[0]));
				index_z = int((Grids.ptr_zmat_vec[i] + 0.001f - *p.min_z) / (Grids.ptr_zgrid[1] - Grids.ptr_zgrid[0]));

				// for each point on the state space, cycle through each possible z' realization

				for (int zz = 0; zz < *p.dimz; zz++) {

					value = Grids.ptr_value[index_a + dima * zz];

					trans = pi_z[dimz * index_z + zz];

					aux_expectation += trans * value;

				}

				Grids.ptr_ev[i] = 0.5f * (aux_expectation) + 0.5f * Grids.ptr_ev[i]; // perform an update with a relaxation parameter (0.5 here)

			}

			g.sync();


		}

		iter++;
	}

	g.sync();


	// VFI is done, now we compute consumption and labour and capital income at each point on the state space

	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < *p.dim_total; i += gridDim.x * blockDim.x) {

		int dima = *p.dima;
		int dimz = *p.dimz;
		float dx_a = 1 / (Grids.ptr_agrid[2] - Grids.ptr_agrid[1]);
		float sigma = *p.sigma_hh;
		float beta = *p.beta;
		float min_a = *p.min_a;

		// get linear index of policy a'

		index_a = int(((Grids.ptr_policy[i]) - *p.min_a) * dx_a);
		index_z = int((Grids.ptr_zmat_vec[i] + 0.001f - *p.min_z) / (Grids.ptr_zgrid[1] - Grids.ptr_zgrid[0]));
		linear_index = index_a + dima * index_z;

		float precompute = (1 + r_interest_rate) * ((Grids.ptr_amat_vec[i]))
			+ r_wage * Grids.ptr_zmat_vec_exp[i];

		// obtain interpolant

		interpolant = (Grids.ptr_amat_vec[index_a + 1] - (Grids.ptr_policy[i])) * dx_a * Grids.ptr_ev[linear_index]
			+ (1.0f - (Grids.ptr_amat_vec[index_a + 1] - (Grids.ptr_policy[i])) * dx_a) * Grids.ptr_ev[linear_index + 1];

		// compute value function

		Grids.ptr_value[i] = (powf((precompute - Grids.ptr_policy[i]), 1.0f - sigma) / (1.0f - sigma)
			+ beta * interpolant);

		// obtain consumption, labour income and capital income at each point on the state space

		Grids.ptr_consumption[i] = (precompute - Grids.ptr_policy[i]);
		Grids.ptr_income_labour[i] = r_wage * Grids.ptr_zmat_vec_exp[i] ;
		Grids.ptr_income_capital[i] = r_interest_rate * (Grids.ptr_amat_vec[i]);

	}

}


