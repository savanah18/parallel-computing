#include <iostream>
#include "utils.cuh"
#include "kernel.cuh"
#include <chrono>
#include <math.h>

using namespace std;
const pair<int, int> randRange = { 0,100 };

const int DEFAULT_BLOCK_WIDTH = 4;
const int MAX_BLOCKS_PER_SM = 16;

class BaseRunner {
public:	
	// ABSTRACT method
	virtual void runKernel(dim3 dimGrid, dim3 dimBlock, float* A_d, float* B_d, float* C_d, int width) = 0;

	// COMMON Method
	void run(float* B, float* C, int width, int block_width=DEFAULT_BLOCK_WIDTH, int log=0) {
		auto start = chrono::high_resolution_clock::now();
		size_t memSize = width * width * sizeof(float);
		float* A_d; float* B_d; float* C_d;
		cudaMalloc((void**)&A_d, memSize);
		cudaMalloc((void**)&B_d, memSize);
		cudaMalloc((void**)&C_d, memSize);

		cudaMemcpy(B_d, B, memSize, cudaMemcpyHostToDevice);
		cudaMemcpy(C_d, C, memSize, cudaMemcpyHostToDevice);

		auto load_end = chrono::high_resolution_clock::now();
		chrono::duration<double> duration = load_end - start;
		cout << "Host->Device Load Time: " << duration.count() << " seconds" << endl;

		// kernel setup
		int gridSize = (width + block_width - 1) / block_width; // number of blocks per dim in grid
		
		dim3 dimGrid(gridSize, gridSize); // maximize number
		dim3 dimBlock(block_width, block_width);

		cout << "Grid Dimension: " << gridSize << " x " << gridSize << endl;
		cout << "Block Dimension: " << block_width << " x " << block_width << endl;

		auto kernel_setup = chrono::high_resolution_clock::now();
		duration = kernel_setup - load_end;
		cout << "Kernel Setup Time: " << duration.count() << " seconds" << endl;

		// kernel call
		runKernel(dimGrid,dimBlock, A_d, B_d, C_d, width);
		// Error checking
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
		}

		auto kernel_call = chrono::high_resolution_clock::now();
		duration = kernel_call - kernel_setup;
		cout << "Kernel Function Call Time: " << duration.count() << " seconds" << endl;
		cout << duration.count() << endl;

		// copy result to host (flat dimension)
		float* A = new float[width * width];
		cudaMemcpy(A, A_d, memSize, cudaMemcpyDeviceToHost);
		printf("Matrix A[%d,%d]\n", width, width);
		if(log) printMatrix(A, width);

		auto load_host_end = chrono::high_resolution_clock::now();
		duration = load_host_end - kernel_call;
		cout << "Device->Host Load Time: " << duration.count() << " seconds" << endl;

		// free memory
		cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
		free(A); free(B); free(C);
		
		auto run_end = chrono::high_resolution_clock::now();
		duration = run_end - start;
		cout << "Total Execution Time: " << duration.count() << " seconds" << endl;
	}
};

class K1Runner : public BaseRunner {
public:
	void runKernel(dim3 dimGrid, dim3 dimBlock, float* A_d, float* B_d, float* C_d, int width) {
		kernel_1t1e << <dimGrid, dimBlock >> > (A_d, B_d, C_d, width);
	}
};

class K2Runner : public BaseRunner {
public:
	void runKernel(dim3 dimGrid, dim3 dimBlock, float* A_d, float* B_d, float* C_d, int width) {
		kernel_1t1r << <dimGrid, dimBlock >> > (A_d, B_d, C_d, width);
	}
};

class K3Runner : public BaseRunner {
public:
	void runKernel(dim3 dimGrid, dim3 dimBlock, float* A_d, float* B_d, float* C_d, int width) {
		kernel_1t1c << <dimGrid, dimBlock >> > (A_d, B_d, C_d, width);
	}
};

int main() {
	int deviceCount;
	cudaError_t err;

	cudaGetDeviceCount(&deviceCount);
	cout << "Device Count\t" << deviceCount << endl;

	for (int device = 0; device < deviceCount ;device++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printCudaDeviceProperties(deviceProp);
	}

	// Initialize runners
	K1Runner runner1 = K1Runner();
	K2Runner runner2 = K2Runner();
	K3Runner runner3 = K3Runner();


	int tcs; cin >> tcs;
	for (int tc = 0;tc < tcs; tc++) {
		cout << "==================================" << endl;
		cout << "Test Case: " << tc << endl;
		int runner,width, block_width, log; 
		cin >> runner >> width >> block_width >> log;

		float* B = generateRandomMatrix(width, randRange);
		printf("Matrix B[%d,%d]\n", width, width);
		if(log) printMatrix(B, width);

		float* C = generateRandomMatrix(width, randRange);
		printf("Matrix C[%d,%d]\n", width, width);
		if(log) printMatrix(C, width);

		switch (runner)
		{
		case 1:
			runner1.run(B, C, width, block_width, log);
			break;
		case 2:
			runner2.run(B, C, width, block_width, log);
			break;
		case 3:
			runner3.run(B, C, width, block_width, log);
			break;
		default:
			break;
		}
		cout << "==================================" << endl;
	}
}