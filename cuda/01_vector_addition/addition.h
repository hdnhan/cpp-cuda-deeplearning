#ifndef ADDITION_H
#define ADDITION_H

#include <cassert>  // assert
#include <stdio.h>

namespace addition {
// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
/*
Step:
    1 Calculate global thread ID
    2. Boundary check
*/

template <typename T>
__global__ void add(const T *a, const T *b, T *c, unsigned int N) {
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) {
        c[tid] = a[tid] + b[tid];
        // printf("%d, %f, %f, %f\n", tid, a[tid], b[tid], c[tid]);
    }
}

// Initialize random numbers in each array
template <typename T>
void init(T *a, T *b, int N, int low = 0, int high = 1) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = (T)rand() / (float)RAND_MAX * (high - low) + low;
        b[i] = (T)rand() / (float)RAND_MAX * (high - low) + low;
    }
}

// Check result
template <typename T>
void verify(const T *a, const T *b, const T *c, int N) {
    for (int i = 0; i < N; i++) {
        // std::cout << i << ": " << a[i] << " " << b[i] << " " << c[i] << std::endl;
        assert(c[i] == a[i] + b[i]);
    }
}

}  // namespace addition

#endif