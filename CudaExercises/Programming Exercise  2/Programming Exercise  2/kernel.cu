
#include "kernel.cuh"
#include <stdio.h>

const int TILE_WIDTH = 32; 
// a tile width of 32 will invoke 2 global memory load of size 32x32x4 bytes totalling to 
// 8kB < 48 KB (valid)
// Ideal runtime scenario will have 64Kb/8Kb = 8 blocks running in paralle per SM


__global__ void matmul_rec_glob(float* A, float* B, float* C, int n, int k, int m) {
	// this thread computs C[x][y];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	if (row < n && col < m) {
		float cVal = 0;
		for (int _k = 0; _k < k;_k++) {
			cVal += A[row * k + _k] * B[_k * m + col];
		}
		C[row * m + col] = cVal;
	}
}

__global__ void matmul_rec_shar(float* A, float* B, float* C, int n, int k, int m) {
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	if (row < n && col < m) {
		float pValue = 0; // partial sum
		for (int q = 0; q < k / TILE_WIDTH; q++) {
			// Load collaboratively a row/col on a submatrix
			ds_A[ty][tx] = A[(row * k) + (q * TILE_WIDTH + tx)];
			ds_B[ty][tx] = B[col + (q * TILE_WIDTH + ty) * k];

			__syncthreads();
			for (int r = 0; r < TILE_WIDTH;r++) {
				pValue += ds_A[ty][r] * ds_B[r][tx];
			}
			__syncthreads();
		}
		C[row * m + col] = pValue;
	}
}