#include <chrono>
#include <iostream>

#include "multiplication.h"

typedef double T;
#define MINV 0
#define MAXV 1  // float/double => should be <= 1

// a: arows x N
// b: N x bcols
int main() {
    // Matrix size of 1024 x 1024;
    const unsigned int arows = 1 << 10;
    const unsigned int N = 1 << 11;
    const unsigned int bcols = 1 << 12;

    // Host arrays
    T *ha = (T *)malloc(arows * N * sizeof(T));
    T *hb = (T *)malloc(N * bcols * sizeof(T));
    T *hc = (T *)malloc(arows * bcols * sizeof(T));
    T *hbt = (T *)malloc(N * bcols * sizeof(T));  // tranpose

    // Initialize matrices
    multiplication::init(ha, arows, N);
    multiplication::init(hb, N, bcols);

    // Tranpose
    multiplication::transpose(hb, hbt, N, bcols);

    // Allocate device memory
    T *da, *dbt, *dc;
    cudaMalloc(&da, arows * N * sizeof(T));
    cudaMalloc(&dbt, N * bcols * sizeof(T));
    cudaMalloc(&dc, arows * bcols * sizeof(T));

    // Copy data to the device
    cudaMemcpy(da, ha, arows * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dbt, hbt, N * bcols * sizeof(T), cudaMemcpyHostToDevice);

    // Threads per CTA dimension
    const unsigned int RTHREADS = 32;
    const unsigned int CTHREADS = 32;

    // Blocks per grid dimension
    const unsigned int RBLOCKS = (max(arows, N) + RTHREADS - 1) / RTHREADS;
    const unsigned int CBLOCKS = (max(N, bcols) + CTHREADS - 1) / CTHREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(RTHREADS, CTHREADS);
    dim3 blocks(RBLOCKS, CBLOCKS);

    auto start = std::chrono::high_resolution_clock::now();
    // Launch kernel
    multiplication::mul_rows<<<blocks, threads>>>(da, dbt, dc, arows, N, bcols);

    // Copy back to the host
    cudaMemcpy(hc, dc, arows * bcols * sizeof(T), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Time Execution: %0.6f\n", diff.count());

    // Check result
    start = std::chrono::high_resolution_clock::now();
    multiplication::verify(ha, hb, hc, arows, N, bcols);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("Time Verification: %0.6f\n", diff.count());

    std::cout << "COMPLETED SUCCESSFULLY\n";

    // Free memory on device
    cudaFree(da);
    cudaFree(dbt);
    cudaFree(dc);

    return 0;
}