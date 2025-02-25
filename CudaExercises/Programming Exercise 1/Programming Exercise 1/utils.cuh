#ifndef UTILS_H
#define UTILS_H

#include "cuda_runtime.h"
#include <iostream>
#include <vector>

using namespace std;

const int MB = 1048576; // number of bytes in 1 Megabytes
const int KB = 1024;

void printCudaDeviceProperties(cudaDeviceProp& deviceProp);
float* generateRandomMatrix(int size, pair<int, int> range);
void printMatrix(float* matrix, int size);
void printVectorizedMatrix(float* matrix, pair<int, int> dim);
#endif 
