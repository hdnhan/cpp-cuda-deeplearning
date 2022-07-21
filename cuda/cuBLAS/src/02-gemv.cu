#include <cublas_v2.h>
#include <chrono>
#include "common.h"

typedef double T;

// Matrix size
const uint64_t M = 1 << 15;
const uint64_t N = 1 << 14;

// Threads per CTA (1024)
uint64_t NUM_THREADS = 1 << 10;
// CTAs per Grid
uint64_t NUM_BLOCKS = (M + NUM_THREADS - 1) / NUM_THREADS;

void mulmv(const T *hA, const T *hx, T *hy, const uint64_t &M, const uint64_t &N) {
    common::mulmv(hA, hx, hy, M, N);
}

void cumulmv(const T *hA, const T *hx, T *hy, const uint64_t &M, const uint64_t &N) {
    // Allocate device memory
    T *dA, *dx, *dy;
    cudaMalloc(&dA, M * N * sizeof(T));
    cudaMalloc(&dx, N * sizeof(T));
    cudaMalloc(&dy, M * sizeof(T));

    // Copy host memory to device memory
    cudaMemcpy(dA, hA, M * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, N * sizeof(T), cudaMemcpyHostToDevice);

    common::cumulmv<<<NUM_BLOCKS, NUM_THREADS>>>(dA, dx, dy, M, N);
    // Copy device memory to host memory
    cudaMemcpy(hy, dy, M * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dA);
    cudaFree(dx);
    cudaFree(dy);
}

void gemv(const float *hA, const float *hx, float *hy, const uint64_t &M, const uint64_t &N) {
    // Allocate device memory
    float *dA, *dx, *dy;
    cudaMalloc(&dA, M * N * sizeof(float));
    cudaMalloc(&dx, N * sizeof(float));
    cudaMalloc(&dy, M * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(dA, hA, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalaing factors
    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS gemv
    // Calculate: y = alpha * A * x + beta * y
    cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, dA, N, dx, 1, &beta, dy, 1);
    // Copy device memory to host memory
    cudaMemcpy(hy, dy, M * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dA);
    cudaFree(dx);
    cudaFree(dy);
    cublasDestroy_v2(handle);
}

void gemv(const double *hA, const double *hx, double *hy, const uint64_t &M, const uint64_t &N) {
    // Allocate device memory
    double *dA, *dx, *dy;
    cudaMalloc(&dA, M * N * sizeof(double));
    cudaMalloc(&dx, N * sizeof(double));
    cudaMalloc(&dy, M * sizeof(double));

    // Copy host memory to device memory
    cudaMemcpy(dA, hA, M * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, N * sizeof(double), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalaing factors
    double alpha = 1.0f;
    double beta = 0.0f;

    // cuBLAS gemv
    // Calculate: y = alpha * A * x + beta * y
    cublasDgemv(handle, CUBLAS_OP_T, N, M, &alpha, dA, N, dx, 1, &beta, dy, 1);
    // Copy device memory to host memory
    cudaMemcpy(hy, dy, M * sizeof(T), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dA);
    cudaFree(dx);
    cudaFree(dy);
    cublasDestroy_v2(handle);
}

int main(int argc, char *argv[]) {
    // Host arrays
    T *hA = (T *)malloc(M * N * sizeof(T));
    T *hx = (T *)malloc(N * sizeof(T));
    T *hy0 = (T *)malloc(M * sizeof(T));
    T *hy1 = (T *)malloc(M * sizeof(T));
    T *hy2 = (T *)malloc(M * sizeof(T));

    // Initialize matrices
    auto start = std::chrono::high_resolution_clock::now();
    common::init(hA, M, N);
    common::init(hx, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Randomization        : %0.6fs\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    mulmv(hA, hx, hy0, M, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("CPU-side             : %0.6fs\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    cumulmv(hA, hx, hy1, M, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("GPU-side             : %0.6fs\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    gemv(hA, hx, hy2, M, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("cublas<t>gemv        : %0.6fs\n", diff.count());

    // Check result
    start = std::chrono::high_resolution_clock::now();
    T diff01 = common::diff(hy0, hy1, M);
    T diff02 = common::diff(hy0, hy2, M);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("CPU Verification     : %0.6fs\n", diff.count());
    printf("Difference: %0.6f %0.6f\n", diff01, diff02);

    // Free host memory
    free(hA);
    free(hx);
    free(hy0);
    free(hy1);
    return 0;
}
