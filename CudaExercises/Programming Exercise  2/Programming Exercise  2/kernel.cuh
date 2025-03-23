#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include <iostream>
#include "device_launch_parameters.h"

using namespace std;

__global__ void matmul_rec_glob(float* A, float* B, float* C, int n, int k, int m);
__global__ void matmul_rec_shar(float* A, float* B, float* C, int n, int k, int m);

#endif 