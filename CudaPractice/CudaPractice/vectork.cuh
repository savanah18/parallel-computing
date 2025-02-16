#ifndef VECTORK_CUH
#define VECTORK_CUH

#include<vector>
#include "globals.cuh"
using namespace std;
// consts
const int MAX_THREADS = 512;
const int MAX_BLOCKS = 4;

__global__ void addVectorK(float* a_d, float* b_d, float* c_d);
vf addVector(vf& a, vf& b);
mf multMatrix(mf& A, mf& B);

#endif // VECTORK_CUH