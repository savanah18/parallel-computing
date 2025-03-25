#include <iostream>
#include "utils.cuh"
#include "kernel.cuh"
#include <chrono>
#include <math.h>

using namespace std;
const pair<int, int> randRange = { 0,100 };

const int DEFAULT_BLOCK_WIDTH = 32;
const int MAX_BLOCKS_PER_SM = 16;
const pair<int, int> DEFAULT_RANGE_VALUES = { 0,10 };

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
		/*matmul_rec_glob << <dimGrid, dimBlock >> > (A_d, B_d, C_d, n, k, m);*/

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
		cout << duration.count() << " ";

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
		free(C);

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

	int tcs; cin >> tcs;
	for (int tc = 0;tc < tcs; tc++) {
		cout << "==================================" << endl;
		cout << "Test Case: " << tc << endl;
		int bSize, nSamples, log; float offset;
		cin >> bSize >> offset >> nSamples >> log;
		tuple<int, int, int> shapes = generateShape(bSize, offset, tc * 12345 + bSize + offset + nSamples);
		int n, k, m; n = get<0>(shapes);k = get<1>(shapes);m = get<2>(shapes);
		cout << "shapes: " << n << "," << k << "," << m << "," << endl;
		for (int sample = 0;sample <= nSamples;sample++) {
			/*cout << "shapes: " << n << "," << k << "," << m << "," << endl;
			cout << "Generating random matrices ..." << endl;*/
			float* A = generateRandomMatrix(n, k, DEFAULT_RANGE_VALUES);
			float* B = generateRandomMatrix(k, m, DEFAULT_RANGE_VALUES);
			runner1.run(A, B, n, k, m, DEFAULT_BLOCK_WIDTH, 0);
			runner2.run(A, B, n, k, m, DEFAULT_BLOCK_WIDTH, 0);
			cout << endl;
			free(A); free(B);
		}

		/*runner1.run(A, B, n, k, m, 32, 0);
		runner2.run(A, B, n, k, m, 32, 0);*/
		cout << "==================================" << endl;
	}

	////free(A), free(B);
}
