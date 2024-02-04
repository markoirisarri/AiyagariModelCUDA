// C++ headers

#include <iostream>

// Project headers

#include "model_inputs.cuh"
#include "cpu_functions.cuh"

void compute_gini(parameters p, grids Grids, prices prices) {

	// load a random chunk of assets simulation data

	float* assets_temp = new float[10000];
	int h = 0;
	for (int i = 0; i < 10000; i++) {

		h = i + (*p.number_periods - 500.0f) * *p.number_people;
		assets_temp[i] = Grids.ptr_assets_simulation[h];

	}

	// now we need to sort the assets_temp;

	mergeSort(assets_temp, 0, 10000 - 1);

	// construct the Gini coefficient

	float total_assets = 0;

	for (int i = 0; i < 10000; i++) {

		total_assets += assets_temp[i];

	}

	// check whether we should increase the maximum value for assets on the grid

	if (assets_temp[9800] >= expf(Grids.ptr_agrid[*p.dima - 1]) - 0.1f) {

		for (int i = 0; i < 5; i++) {
			std::cout << "\n\n\n" << "ERROR: Maximum Level of Assets Too Low, increase max_a" << "\n \n \n ";
		}
	}

	float A = 0;
	float cum_people = 0;
	float cum_assets = 0;

	for (int i = 0; i < 10000; i++) {

		cum_people += 1.0f / 10000.0f;
		cum_assets += assets_temp[i] / total_assets;
		A += (cum_people - cum_assets);

	}

	Grids.ptr_gini_coefficient[0] = A / 10000.0f / 0.5;

	delete[] assets_temp;

}

void merge(float* arr, int p, int q, int r) {

	// Create L ← A[p..q] and M ← A[q..r]
	
	int n1 = q - p + 1;
	int n2 = r - q;

	//float L[n1], M[n2];

	float* L = new float[n1];
	float* M = new float[n2];

	for (int i = 0; i < n1; i++)
		L[i] = arr[p + i];
	for (int j = 0; j < n2; j++)
		M[j] = arr[q + 1 + j];

	// Maintain current index of sub-arrays and main array
	int i, j, k;
	i = 0;
	j = 0;
	k = p;

	// Until we reach either end of either L or M, pick larger among
	// elements L and M and place them in the correct position at A[p..r]
	while (i < n1 && j < n2) {
		if (L[i] <= M[j]) {
			arr[k] = L[i];
			i++;
		}
		else {
			arr[k] = M[j];
			j++;
		}
		k++;
	}

	// When we run out of elements in either L or M,
	// pick up the remaining elements and put in A[p..r]
	while (i < n1) {
		arr[k] = L[i];
		i++;
		k++;
	}

	while (j < n2) {
		arr[k] = M[j];
		j++;
		k++;
	}

	delete[] L;
	delete[] M;

}

// Divide the array into two subarrays, sort them and merge them
void mergeSort(float* arr, int l, int r) {
	if (l < r) {
		// m is the point where the array is divided into two subarrays
		int m = l + (r - l) / 2;

		mergeSort(arr, l, m);
		mergeSort(arr, m + 1, r);

		// Merge the sorted subarrays
		int p = l,
			int q = m;
		merge(arr, l, m, r);
	}
}