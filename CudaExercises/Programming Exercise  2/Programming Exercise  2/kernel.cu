
#include "kernel.cuh"
#include <stdio.h>

using namespace std;

const int TILE_WIDTH = 32; 
// a tile width of 32 will invoke 2 global memory load of size 32x32x4 bytes totalling to 
// 8kB < 48 KB (valid)
// Ideal runtime scenario will have 64Kb/8Kb = 8 blocks running in parallel per SM, 
//   which is valid since Max Blocks / Multiprocessor:	16




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

// Theoretical GMA Computations
// In a block (which in this case has size TILE_WIDTH, TILE_WIDTH for maximum shared resource allocation):
//   each thread / per single pass
//     - load 1 element of each submatrix of A and B (size TILE_WIDTH x TILE_WIDTH) totalling to 2 memory loads
//     - will perform 1 (addition) + 1 (oplication) = 2 single point(sp) arithmetic operations per row(col) element
//         of A(B) totalling to 2xTILE_WIDTH=64 sp ops.
//     - CGMA: 64/2 = 32
// 
// Note:  
__global__ void matmul_rec_shar(float* A, float* B, float* C, int n, int k, int m) {
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	if (row < n && col < m) {
		float pValue = 0; // partial sum
		for (int q = 0; q < (k + TILE_WIDTH - 1) / TILE_WIDTH; q++) {
			// Load collaboratively a row/col on a submatrix
			ds_A[ty][tx] = A[(row * k) + (q * TILE_WIDTH + tx)];
			ds_B[ty][tx] = B[col + (q * TILE_WIDTH + ty) * k];
			// ds_B[ty][tx] = B[(q * TILE_WIDTH + ty)*m + col]; // more coalesced??

			__syncthreads();
			for (int r = 0; r < TILE_WIDTH;r++) {
				pValue += ds_A[ty][r] * ds_B[r][tx];
				// pValue += ds_A[ty][r] * ds_B[tx][r];
			}
			__syncthreads();
		}
		C[row * m + col] = pValue;
	}
}