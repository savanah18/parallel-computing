#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include <iostream>
#include "device_launch_parameters.h"

using namespace std;

__global__ void kernel_1t1e(float* A_d, float* B_d, float* C_d, int width);
__global__ void kernel_1t1r(float* A_d, float* B_d, float* C_d, int width);
__global__ void kernel_1t1c(float* A_d, float* B_d, float* C_d, int width);

#endif 
