#include "cuda_runtime.h"
#include <iostream>
#include "device_launch_parameters.h"
#include "kernel.cuh"

using namespace std;

// each kernel function thread has copy of global variables
__global__ void kernel_1t1e(float* A_d, float* B_d, float* C_d, int width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < width) && (col < width)) {
		// each thread computes one element
		A_d[row * width + col] = B_d[row * width + col] + C_d[row * width + col];
	}
}

__global__ void kernel_1t1r(float* A_d, float* B_d, float* C_d, int width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if ((row < width)) {
		// each thread computes row
		for (int col = 0;col < width;col++) {
			A_d[row * width + col] = B_d[row * width + col] + C_d[row * width + col];
		}
	}
}

__global__ void kernel_1t1c(float* A_d, float* B_d, float* C_d, int width) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((col < width)) {
		// each thread computes col
		for (int row = 0;row < width;row++) {
			A_d[row * width + col] = B_d[row * width + col] + C_d[row * width + col];
		}
	}
}