#ifndef UTILS_H
#define UTILS_H

#include "cuda_runtime.h"
#include <iostream>
#include <tuple>

using namespace std;

const int MB = 1048576; // number of bytes in 1 Megabytes
const int KB = 1024; // numbero of bytes in 1 Kilobytes

void printCudaDeviceProperties(cudaDeviceProp& deviceProp);
float* generateRandomMatrix(int n, int m, pair<int, int> range);
void printMatrix(float* matrix, int n, int m);
tuple<int,int,int> generateShape(int bSize, float offset, int seed);

#endif 