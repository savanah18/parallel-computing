#include <iostream>
#include "utils.cuh"
#include "kernel.cuh"
#include <chrono>
#include <math.h>

using namespace std;
const pair<int, int> randRange = { 0,100 };

const int DEFAULT_BLOCK_WIDTH = 32;
const int MAX_BLOCKS_PER_SM = 16;

class BaseRunner {
public:
	// ABSTRACT method
	virtual void runKernel(dim3 dimGrid, dim3 dimBlock, float* A_d, float* B_d, float* C_d, int n, int k, int m) = 0;

	// COMMON Method
	void run(float* A, float* B, int n, int k, int m, int block_width = DEFAULT_BLOCK_WIDTH, int log = 0) {

		auto start = chrono::high_resolution_clock::now();
		size_t memSizeA = n * k * sizeof(float);
		size_t memSizeB = k * m * sizeof(float);
		size_t memSizeC = n * m * sizeof(float);

		float* A_d; float* B_d; float* C_d;
		cudaMalloc((void**)&A_d, memSizeA);
		cudaMalloc((void**)&B_d, memSizeB);
		cudaMalloc((void**)&C_d, memSizeC);

		/*printf("Matrix A[%d,%d]\n", n, k);printMatrix(A, n, k); 
		printf("Matrix B[%d,%d]\n", k, m);printMatrix(B, k, m);*/
		cudaMemcpy(A_d, A, memSizeA, cudaMemcpyHostToDevice);
		cudaMemcpy(B_d, B, memSizeB, cudaMemcpyHostToDevice);

		/*auto load_end = chrono::high_resolution_clock::now();
		chrono::duration<double> duration = load_end - start;
		cout << "Host->Device Load Time: " << duration.count() << " seconds" << endl;*/

		// kernel setup
		int gridSizeY = (n + block_width - 1) / block_width; // number of blocks per dim.y in grid
		int gridSizeX = (m + block_width - 1) / block_width; // number of blocks per dim.y in grid


		dim3 dimGrid(gridSizeX, gridSizeY); // maximize number
		dim3 dimBlock(block_width, block_width);

		/*cout << "Grid Dimension: " << gridSizeX << " x " << gridSizeY << endl;
		cout << "Block Dimension: " << block_width << " x " << block_width << endl;*/

		// warmup
		matmul_rec_glob << <dimGrid, dimBlock >> > (A_d, B_d, C_d, n, k, m);

		/*auto kernel_setup = chrono::high_resolution_clock::now();
		duration = kernel_setup - load_end;
		cout << "Kernel Setup Time: " << duration.count() << " seconds" << endl;*/

		// kernel call
		auto kernel_call_start = chrono::high_resolution_clock::now();
		runKernel(dimGrid, dimBlock, A_d, B_d, C_d, n, k, m);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
		}
		auto kernel_call_end = chrono::high_resolution_clock::now();
		chrono::duration<double> duration = kernel_call_end - kernel_call_start;
		//cout << "Kernel Function Call Time: " << duration.count() << " seconds" << endl;
		cout << duration.count() << endl;

		// copy result to host (flat dimension)
		float* C = new float[n * m];
		cudaMemcpy(C, C_d, memSizeC, cudaMemcpyDeviceToHost);
		cudaMemcpy(A, A_d, memSizeA, cudaMemcpyDeviceToHost);
		cudaMemcpy(B, B_d, memSizeB, cudaMemcpyDeviceToHost);

		/*printf("matrix c[%d,%d]\n", n, m);
		if (log) printmatrix(c, n, m);*/

		//auto load_host_end = chrono::high_resolution_clock::now();
		//duration = load_host_end - kernel_call;
		//cout << "Device->Host Load Time: " << duration.count() << " seconds" << endl;

		// free memory
		cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
		cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
		free(A); free(B); // free(C);

		/*auto run_end = chrono::high_resolution_clock::now();
		duration = run_end - start;
		cout << "Total Execution Time: " << duration.count() << " seconds" << endl;*/
	}
};

class K1Runner : public BaseRunner {
public:
	void runKernel(dim3 dimGrid, dim3 dimBlock, float* A_d, float* B_d, float* C_d, int n, int k, int m) {
		// Get size of A_d, B_d and C_d
		matmul_rec_glob << <dimGrid, dimBlock >> > (A_d, B_d, C_d, n,k,m);
	}
};

class K2Runner : public BaseRunner {
public:
	void runKernel(dim3 dimGrid, dim3 dimBlock, float* A_d, float* B_d, float* C_d, int n, int k, int m) {
		// Get size of A_d, B_d and C_d
		matmul_rec_shar << <dimGrid, dimBlock >> > (A_d, B_d, C_d, n, k, m);
	}
};

int main() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cout << "Device Count\t" << deviceCount << endl;

	for (int device = 0; device < deviceCount;device++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printCudaDeviceProperties(deviceProp);
	}

	// initialize runner
	K1Runner runner1 = K1Runner();
	K2Runner runner2 = K2Runner();
	//int n = 32; int k = 64; int m = 32;
	//float* A = generateRandomMatrix(n, k, make_pair(0,10));
	//float* B = generateRandomMatrix(k, m, make_pair(0, 10));
	////float A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	//// float B[8] = { 1, 2, 3, 4, 5, 6, 7, 8};
	//runner1.run(A, B, n, k, m, 32, 1);

	int tcs; cin >> tcs;
	for (int tc = 0;tc < tcs; tc++) {
		cout << "==================================" << endl;
		cout << "Test Case: " << tc << endl;
		int n, k, m;

		cout << "==================================" << endl;
	}

	////free(A), free(B);
}
