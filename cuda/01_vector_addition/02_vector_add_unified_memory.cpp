#include <chrono>
#include <iostream>

#include "addition.h"
typedef float T;
#define MINV 0
#define MAXV 100

int main() {
    // Array size of N
    const int N = 1 << 29;

    // Declare unified memory pointers
    T *a, *b, *c;

    // Allocation memory for these pointers
    cudaMallocManaged(&a, N * sizeof(T));
    cudaMallocManaged(&b, N * sizeof(T));
    cudaMallocManaged(&c, N * sizeof(T));

    // Initialize vectors
    addition::init(a, b, N, MINV, MAXV);

    // Threads per CTA (1024 threads per CTA)
    int BLOCK_SIZE = 1 << 10;

    // CTAs per Grid
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch CUDA kernel
    addition::add<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, c, N);

    // Wait for all previous operations before using values
    // We need this because we don't get the implicit synchronization of
    // cudaMemcpy like in the original example
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Time Execution: %0.6f\n", diff.count());

    // Verify the result on the CPU
    start = std::chrono::high_resolution_clock::now();
    addition::verify(a, b, c, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("Time Verification: %0.6f\n", diff.count());

    // Free unified memory (same as memory allocated with cudaMalloc)
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    std::cout << "COMPLETED SUCCESSFULLY!\n";

    return 0;
}
