#ifndef MULTIPLICATION_H
#define MULTIPLICATION_H

#include <stdio.h>  // printf

#include <cassert>  // assert

namespace multiplication {

/*
Step:
    1. Compute each thread's global row and column index
    2. Iterate over row, and down column
*/

template <typename T>
__global__ void mul(const T *a, const T *b, T *c, const unsigned int arows, const unsigned int N, const unsigned int bcols) {
    unsigned int rid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cid = blockIdx.y * blockDim.y + threadIdx.y;

    if (rid >= arows || cid >= bcols) return;
    c[rid * bcols + cid] = 0;
    for (int k = 0; k < N; k++) {
        c[rid * bcols + cid] += a[rid * N + k] * b[k * bcols + cid];
    }
}

// Shared memory is used to store partial results
template <typename T>
__global__ void mul_shared_mem(const T *a, const T *b, T *c, const unsigned int arows, const unsigned int N, const unsigned int bcols) {
    unsigned int rid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cid = blockIdx.y * blockDim.y + threadIdx.y;

    // Statically allocated shared memory
    __shared__ T sa[1 << 10];  // = RTHREADS x CTHREADS
    __shared__ T sb[1 << 10];  // = RTHREADS x CTHREADS

    T tmp = 0;
    for (int i = 0; i < N; i += blockDim.y) {
        sa[threadIdx.x * blockDim.y + threadIdx.y] = 0;
        sb[threadIdx.x * blockDim.y + threadIdx.y] = 0;
        if (rid * N + i + threadIdx.y < arows * N)
            sa[threadIdx.x * blockDim.y + threadIdx.y] = a[rid * N + i + threadIdx.y];
        if (i * bcols + threadIdx.x * bcols + cid < N * bcols)
            sb[threadIdx.x * blockDim.y + threadIdx.y] = b[i * bcols + threadIdx.x * bcols + cid];

        __syncthreads();

        for (int j = 0; j < blockDim.x; j++) {
            tmp += sa[threadIdx.x * blockDim.y + j] * sb[j * blockDim.y + threadIdx.y];
        }
        __syncthreads();
    }
    if (rid < arows && cid < bcols)
        c[rid * bcols + cid] = tmp;
}

// Matrix multiplication on rows
template <typename T>
__global__ void mul_rows(const T *a, const T *b, T *c, const unsigned int arows, const unsigned int N, const unsigned int bcols) {
    unsigned int rid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cid = blockIdx.y * blockDim.y + threadIdx.y;

    if (rid >= arows || cid >= bcols) return;
    c[rid * bcols + cid] = 0;
    for (int k = 0; k < N; k++) {
        c[rid * bcols + cid] += a[rid * N + k] * b[cid * N + k];
    }
}

template <typename T>
__global__ void mul_cols(const T *a, const T *b, T *c, const unsigned int arows, const unsigned int N, const unsigned int bcols) {
    unsigned int rid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int cid = blockIdx.y * blockDim.y + threadIdx.y;

    if (rid >= arows || cid >= bcols) return;
    c[rid * bcols + cid] = 0;
    for (int k = 0; k < N; k++) {
        c[rid * bcols + cid] += a[k * arows + rid] * b[k * bcols + cid];
    }
}

// Initialize random numbers in each array
template <typename T>
void init(T *a, const unsigned int rows, const unsigned int cols, int low = 0, int high = 1) {
    srand((unsigned int)time(NULL));
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            a[i * rows + j] = (T)rand() / (float)RAND_MAX * (high - low) + low;
        }
    }
}

// Matrix transpose
template <typename T>
void transpose(const T *a, T *at, const unsigned int rows, const unsigned int cols) {
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            at[j * rows + i] = a[i * cols + j];
        }
    }
}

// Check result on the CPU
template <typename T>
void verify(const T *a, const T *b, const T *c, const unsigned int arows, const unsigned int N, const unsigned int bcols, double eps = 1e-12) {
    for (unsigned int i = 0; i < arows; i++) {
        for (unsigned int j = 0; j < bcols; j++) {
            T tmp = 0;
            for (unsigned int k = 0; k < N; k++) {
                tmp += a[i * N + k] * b[k * bcols + j];
            }
            // std::cout.precision(17);
            // std::cout << i << " " << j << " " << tmp << " " << c[i * bcols + j] << " " << abs(tmp - c[i * bcols + j]) << std::endl;
            assert(abs(tmp - c[i * bcols + j]) <= (T)eps);
        }
    }
}

}  // namespace multiplication

#endif