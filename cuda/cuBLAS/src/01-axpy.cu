#include <cublas_v2.h>

#include <chrono>

#include "common.h"

typedef float T; // float vs double
const uint64_t N = 1 << 28;

// Threads per CTA (1024)
uint64_t NUM_THREADS = 1 << 10;
// CTAs per Grid
uint64_t NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

void addvv(const T *ha, const T *hb, T *hr, const uint64_t &N) {
    common::addvv(ha, hb, hr, N);
}

void cuaddvv(const T *ha, const T *hb, T *hr, const uint64_t &N) {
    // Allocate memory on the device-side (GPU-side)
    T *da, *db, *dc;
    cudaMalloc(&da, N * sizeof(T));
    cudaMalloc(&db, N * sizeof(T));
    cudaMalloc(&dc, N * sizeof(T));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(da, ha, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(T), cudaMemcpyHostToDevice);

    // Launch the kernel on the GPU
    common::cuaddvv<<<NUM_BLOCKS, NUM_THREADS>>>(da, db, dc, N);
    // cudaMemcpy acts as both a memcpy and synchronization barrier
    cudaMemcpy(hr, dc, N * sizeof(T), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void axpy(const double *ha, const double *hb, double *hr, const uint64_t &N) {
    // Allocate memory on the device-side (GPU-side)
    double *da, *db, *dc;
    cudaMalloc(&da, N * sizeof(double));
    cudaMalloc(&db, N * sizeof(double));
    cudaMalloc(&dc, N * sizeof(double));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(da, ha, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(double), cudaMemcpyHostToDevice);

    // Create and initialize a new context
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    const double scale = 1.0f;  // a * x + y

    // Launch the kernel on the GPU
    cublasDaxpy(handle, N, &scale, da, 1, db, 1);
    cudaMemcpy(hr, db, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cublasDestroy_v2(handle);
}

void axpy(const float *ha, const float *hb, float *hr, const uint64_t &N) {
    // Allocate memory on the device-side (GPU-side)
    float *da, *db, *dc;
    cudaMalloc(&da, N * sizeof(float));
    cudaMalloc(&db, N * sizeof(float));
    cudaMalloc(&dc, N * sizeof(float));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create and initialize a new context
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    const float scale = 1.0f;  // a * x + y

    // Launch the kernel on the GPU
    cublasSaxpy(handle, N, &scale, da, 1, db, 1);
    cudaMemcpy(hr, db, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cublasDestroy_v2(handle);
}

int main(int argc, char *argv[]) {
    // Vectors for holding the host-side (CPU-side) data
    T *ha = (T *)malloc(N * sizeof(T));
    T *hb = (T *)malloc(N * sizeof(T));
    T *hr0 = (T *)malloc(N * sizeof(T));
    T *hr1 = (T *)malloc(N * sizeof(T));
    T *hr2 = (T *)malloc(N * sizeof(T));

    // Initialize random numbers in each array
    auto start = std::chrono::high_resolution_clock::now();
    common::init(ha, N);
    common::init(hb, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Randomization   : %0.6fs\n", diff.count());

    // Normal vector addtion on CUDA
    start = std::chrono::high_resolution_clock::now();
    cuaddvv(ha, hb, hr1, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("GPU-side        : %0.6fs\n", diff.count());

    // cuBLAS axpy on CUDA
    start = std::chrono::high_resolution_clock::now();
    axpy(ha, hb, hr2, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("cublas<t>axpy   : %0.6fs\n", diff.count());

    // Normal vector addition on CPU
    start = std::chrono::high_resolution_clock::now();
    addvv(ha, hb, hr0, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("CPU-side        : %0.6fs\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    T diff01 = common::diff(hr0, hr1, N);
    T diff02 = common::diff(hr0, hr2, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("CPU Verification: %0.6fs\n", diff.count());
    printf("Difference: %0.6f %0.6f\n", diff01, diff02);

    // Free host memory
    free(ha);
    free(hb);
    free(hr0);
    free(hr1);
    free(hr2);
    return 0;
}