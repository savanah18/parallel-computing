#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cuh"
#include <iostream>
#include <random>


using namespace std;

void printCudaDeviceProperties(cudaDeviceProp  &deviceProp) {
	cout << "===========================================================" << endl;
	cout << "Device " << deviceProp.name << " Properties:" << endl;
	cout << "Clock Rate (KHz) " << deviceProp.clockRate << endl;
	cout << "Memory Clock Rate (KHz) " << deviceProp.memoryClockRate << endl;
	cout << "Total Global Memory (MB):\t" << (deviceProp.totalGlobalMem / MB) << endl;
	cout << "Shared Memory / Block (KB):\t" << (deviceProp.sharedMemPerBlock / KB) << endl;
	// Warp: groups of threads that executes the same instruction
	cout << "Warp Size:\t" << deviceProp.warpSize << endl;
	// Pitch: Padded size of each row in an array?
	cout << "Pitch (MB):\t" << (deviceProp.memPitch / MB) << endl;
	cout << "Max Threads / Block:\t" << deviceProp.maxThreadsPerBlock << endl;
	cout << "Max Dimension Size of Block :\t" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << endl;
	cout << "Max Dimension Size of Grid :\t" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << endl;
	cout << "Multiprocess Count:\t" << deviceProp.multiProcessorCount << endl;
	cout << "Max Blocks / Multiprocessor:\t" << deviceProp.maxBlocksPerMultiProcessor << endl;
	cout << "Concurrent Kernels:\t" << deviceProp.concurrentKernels << endl;
	cout << "Max Threads / Multiprocessor:\t" << deviceProp.maxThreadsPerMultiProcessor << endl;
	cout << "Shared Memory (KB) / Multiprocessor:\t" << (deviceProp.sharedMemPerMultiprocessor / KB) << endl;
	cout << "===========================================================" << endl;
}

float* generateRandomMatrix(int size, pair<int, int> range) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dis(range.first, range.second);
	
	float* matrix = new float [size*size];

	for (int i = 0;i < size;i++) {
		for (int j = 0;j < size;j++) {
			matrix[i*size+j] = dis(gen);
		}
	}
	return matrix;
}

void printMatrix(float* matrix, int size) {
	for (int i = 0;i < size;i++) {
		for (int j = 0;j < size;j++) {
			printf("%08.4f\t", matrix[i*size+j]);
		}
		cout << endl;
	}
}

void printVectorizedMatrix(float* matrix, pair<int, int> dim) {
	for (int i = 0;i < dim.first;i++) {
		for (int j = 0;j < dim.second;j++) {
			printf("%08.4f\t",matrix[i*dim.first + j]);
		}
		cout << endl;
	}
}