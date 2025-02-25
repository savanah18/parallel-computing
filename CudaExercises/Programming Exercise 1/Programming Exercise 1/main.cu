#include <iostream>
#include "utils.cuh"
#include "kernel.cuh"
#include <functional>

using namespace std;
const pair<int, int> randRange = { 0,100 };

const int DEFAULT_BLOCK_WIDTH = 4;

class BaseRunner {
public:	
	// ABSTRACT method
	virtual void runKernel(dim3 dimGrid, dim3 dimBlock, float* A_d, float* B_d, float* C_d, int width) = 0;

	// COMMON Method
	void run(float* B, float* C, int width, int block_width=DEFAULT_BLOCK_WIDTH, int log=0) {
		size_t memSize = width * width * sizeof(float);
		float* A_d; float* B_d; float* C_d;
		cudaMalloc((void**)&A_d, memSize);
		cudaMalloc((void**)&B_d, memSize);
		cudaMalloc((void**)&C_d, memSize);

		cudaMemcpy(B_d, B, memSize, cudaMemcpyHostToDevice);
		cudaMemcpy(C_d, C, memSize, cudaMemcpyHostToDevice);

		// kernel setup
		int numBlocks = width / block_width; // number of blocks per dim in grid
		if (width % block_width) numBlocks++;
		dim3 dimGrid(numBlocks, numBlocks);
		dim3 dimBlock(block_width, block_width);

		// kernel call
		runKernel(dimGrid,dimBlock, A_d, B_d, C_d, width);

		// copy result to host (flat dimension)
		float* A = new float[width * width];
		cudaMemcpy(A, A_d, memSize, cudaMemcpyDeviceToHost);
		printf("Matrix A[%d,%d]\n", width, width);
		if(log) printMatrix(A, width);

		// free memory
		cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
		free(A); free(B); free(C);
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