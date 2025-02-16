#include<iostream>
#include<vector>
#include "vectork.cuh"

using namespace std;

// global functions (GOOD FOR both GPU and CPu executions)
__global__ void addVectorK(float* a_d, float* b_d, float* c_d, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n) c_d[i] = a_d[i] + b_d[i];
}

__global__ void multMatrixK(float* A_d, float* B_d, float* C_d, int m, int n, int o) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;

	// Sum_j(A_ij*B_jk)
	if (i < m && k < o) {
		float sum = 0;
		for (int j = 0;j < n;j++) {
			sum += A_d[i * n + j] * B_d[j * o + k]; 
		}
		C_d[i * o + k] = sum;
	}
}

// utility
vf addVector(vf& a, vf& b) {
	float* a_d; float* b_d; float* c_d;
	vf c(a.size());
	// allocate cuda memory for a,b, and c
	int n = (int)a.size(); int memSize = n * sizeof(float);
	cudaMalloc((void**)&a_d, memSize); cudaMalloc((void**)&b_d, memSize); cudaMalloc((void**)&c_d, memSize);
	// copy contents of a->a_d, b->b_d;
	cudaMemcpy(a_d, a.data(), memSize, cudaMemcpyHostToDevice);cudaMemcpy(b_d, b.data(), memSize, cudaMemcpyHostToDevice);

	// call kernel functions
	//int allowedBlocks = min({ 1,MAX_BLOCKS });
	int allowedThreadsPerBlock = min({ n,MAX_THREADS });
	addVectorK << < n / allowedThreadsPerBlock, allowedThreadsPerBlock >> > (a_d, b_d, c_d, n);

	// load result back to host
	cudaMemcpy(c.data(), c_d, memSize, cudaMemcpyDeviceToHost);

	// print result
	for (float e : a) { cout << e << " "; } cout << endl;
	for (float e : b) { cout << e << " "; } cout << endl;
	for (float e : c) { cout << e << " "; } cout << endl;
	// clear memory
	cudaFree(a_d);cudaFree(b_d);cudaFree(c_d);

	// return results
	return c;
}

void printMyMatrix(mf& M) {
	int m = (int)M.size(); int n = (int)M[0].size();
	for (int i = 0;i < m;i++) { for (int j = 0;j < n;j++) { cout << M[i][j] << " "; } cout << endl; }
}

mf multMatrix(mf& A, mf& B) {
	float* A_d; float* B_d; float* C_d;
	int m = (int)A.size(); int n = (int)A[0].size(); int o = (int)B[0].size();

	mf C(m, vf(o));

	cout << "allocate cuda memory for A,B and C" << endl;
	// allocate cuda memory for A,B and C
	cudaMalloc((void**)&A_d, m * n * sizeof(float));
	cudaMalloc((void**)&B_d, n * o * sizeof(float));
	cudaMalloc((void**)&C_d, m * o * sizeof(float));

	cout << "Load to device" << endl;
	// Load to device
	cudaMemcpy2D(A_d, n * sizeof(float), A[0].data(), n * sizeof(float), n, m, cudaMemcpyHostToDevice);
	cudaMemcpy2D(B_d, o * sizeof(float), B[0].data(), o * sizeof(float), o, n, cudaMemcpyHostToDevice);

	cout << "call kernel functions" << endl;
	// call kernel functions
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(n / threadsPerBlock.x, o / threadsPerBlock.y);
	multMatrixK << <numBlocks, threadsPerBlock >> > (A_d, B_d, C_d, m,n,o);

	cout << "Load result to host" << endl;
	// Load result to host
	cudaMemcpy2D(C[0].data(), o * sizeof(float), C_d, o * sizeof(float), o, m, cudaMemcpyDeviceToHost);

	cout << "print result" << endl;
	// print result
	//printMatrix(C);

	//cout << "Free memory" << endl;
	// Free memory
	// cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);

	return C;
}