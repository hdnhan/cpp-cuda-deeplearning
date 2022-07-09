#include <chrono>
#include <iostream>

#include "addition.h"

typedef float T;
#define MINV 0
#define MAXV 100

int main() {
    // Array size of N
    unsigned int N = 1 << 29;

    // Vectors for holding the host-side (CPU-side) data
    T *ha = (T *)malloc(N * sizeof(T));
    T *hb = (T *)malloc(N * sizeof(T));
    T *hc = (T *)malloc(N * sizeof(T));

    // Initialize random numbers in each array
    addition::init(ha, hb, N, MINV, MAXV);

    // Allocate memory on the device-side (GPU-side)
    T *da, *db, *dc;
    cudaMalloc(&da, N * sizeof(T));
    cudaMalloc(&db, N * sizeof(T));
    cudaMalloc(&dc, N * sizeof(T));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(da, ha, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(T), cudaMemcpyHostToDevice);

    // Threads per CTA (1024)
    int NUM_THREADS = 1 << 10;

    // CTAs per Grid
    // We need to launch at LEAST as many threads as we have elements
    // This equation pads an extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel on the GPU
    // Kernel calls are asynchronous (the CPU program continues execution after
    // call, but no necessarily before the kernel finishes)
    addition::add<<<NUM_BLOCKS, NUM_THREADS>>>(da, db, dc, N);

    // Copy sum vector from device to host
    // cudaMemcpy is a synchronous operation, and waits for the prior kernel
    // launch to complete (both go to the default stream in this case).
    // Therefore, this cudaMemcpy acts as both a memcpy and synchronization
    // barrier.
    cudaMemcpy(hc, dc, N * sizeof(T), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Time Execution: %0.6f\n", diff.count());

    // Check result for errors
    start = std::chrono::high_resolution_clock::now();
    addition::verify(ha, hb, hc, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("Time Verification: %0.6f\n", diff.count());

    // Free memory on device
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    std::cout << "COMPLETED SUCCESSFULLY\n";
    return 0;
}
