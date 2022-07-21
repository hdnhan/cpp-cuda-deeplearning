#include <cublas_v2.h>
#include <chrono>
#include "common.h"

typedef double T;

// Matrix size
const uint64_t arows = 1 << 10;
const uint64_t N     = 1 << 11;
const uint64_t bcols = 1 << 12;

// Threads per CTA dimension
const uint64_t RTHREADS = 32;
const uint64_t CTHREADS = 32;

// Blocks per grid dimension
const uint64_t RBLOCKS = (max(arows, N) + RTHREADS - 1) / RTHREADS;
const uint64_t CBLOCKS = (max(N, bcols) + CTHREADS - 1) / CTHREADS;

// Use dim3 structs for block  and grid dimensions
dim3 threads(RTHREADS, CTHREADS);
dim3 blocks(RBLOCKS, CBLOCKS);

void mulmm(const T *hA, const T *hB, T *hR, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols) {
    common::mulmm(hA, hB, hR, arows, N, bcols);
}

void cumulmm(const T *hA, const T *hB, T *hR, const uint64_t &arows, const uint64_t N, const uint64_t &bcols) {
    // Allocate device memory
    T *dA, *dB, *dC;
    cudaMalloc(&dA, arows * N * sizeof(T));
    cudaMalloc(&dB, N * bcols * sizeof(T));
    cudaMalloc(&dC, arows * bcols * sizeof(T));

    // Copy data to the device
    cudaMemcpy(dA, hA, arows * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * bcols * sizeof(T), cudaMemcpyHostToDevice);

    common::cumulmm<<<blocks, threads>>>(dA, dB, dC, arows, N, bcols);
    // Copy device memory to host memory
    cudaMemcpy(hR, dC, arows * bcols * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void cumulmm_shared_mem(const T *hA, const T *hB, T *hR, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols) {
    // Allocate device memory
    T *dA, *dB, *dC;
    cudaMalloc(&dA, arows * N * sizeof(T));
    cudaMalloc(&dB, N * bcols * sizeof(T));
    cudaMalloc(&dC, arows * bcols * sizeof(T));

    // Copy data to the device
    cudaMemcpy(dA, hA, arows * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * bcols * sizeof(T), cudaMemcpyHostToDevice);

    common::cumulmm_shared_mem<<<blocks, threads>>>(dA, dB, dC, arows, N, bcols);
    cudaMemcpy(hR, dC, arows * bcols * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void cumulmm_rows(const T *hA, const T *hB, T *hR, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols) {
    // Transpose matrices
    T *hBt = (T *)malloc(N * bcols * sizeof(T));
    common::transpose(hB, hBt, N, bcols);

    // Allocate device memory
    T *dA, *dB, *dC;
    cudaMalloc(&dA, arows * N * sizeof(T));
    cudaMalloc(&dB, N * bcols * sizeof(T));
    cudaMalloc(&dC, arows * bcols * sizeof(T));

    // Copy data to the device
    cudaMemcpy(dA, hA, arows * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hBt, N * bcols * sizeof(T), cudaMemcpyHostToDevice);

    common::cumulmm_rows<<<blocks, threads>>>(dA, dB, dC, arows, N, bcols);
    cudaMemcpy(hR, dC, arows * bcols * sizeof(T), cudaMemcpyDeviceToHost);

    free(hBt);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void cumulmm_cols(const T *hA, const T *hB, T *hR, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols) {
    // Transpose matrices
    T *hAt = (T *)malloc(arows * N * sizeof(T));
    common::transpose(hA, hAt, arows, N);

    // Allocate device memory
    T *dA, *dB, *dC;
    cudaMalloc(&dA, arows * N * sizeof(T));
    cudaMalloc(&dB, N * bcols * sizeof(T));
    cudaMalloc(&dC, arows * bcols * sizeof(T));

    // Copy data to the device
    cudaMemcpy(dA, hAt, arows * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * bcols * sizeof(T), cudaMemcpyHostToDevice);

    common::cumulmm_cols<<<blocks, threads>>>(dA, dB, dC, arows, N, bcols);
    cudaMemcpy(hR, dC, arows * bcols * sizeof(T), cudaMemcpyDeviceToHost);

    free(hAt);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void gemm_row(const float *hA, const float *hB, float *hR, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols) {
    // Allocate device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, arows * N * sizeof(float));
    cudaMalloc(&dB, N * bcols * sizeof(float));
    cudaMalloc(&dC, arows * bcols * sizeof(float));

    // Copy data to the device
    cudaMemcpy(dA, hA, arows * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * bcols * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalaing factors
    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS gemm
    // Calculate: C = (alpha * A) * B + (beta * C)
    // (arows x N) * (N x bcols) = (arows x bcols)
    // See A and B as row-major matrices and outputs a row-major matrix dC
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, bcols, arows, N, &alpha, dB, bcols, dA, N, &beta, dC, bcols);
    cudaMemcpy(hR, dC, arows * bcols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy_v2(handle);
}

void gemm_row(const double *hA, const double *hB, double *hR, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols) {
    // Allocate device memory
    double *dA, *dB, *dC;
    cudaMalloc(&dA, arows * N * sizeof(double));
    cudaMalloc(&dB, N * bcols * sizeof(double));
    cudaMalloc(&dC, arows * bcols * sizeof(double));

    // Copy data to the device
    cudaMemcpy(dA, hA, arows * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * bcols * sizeof(double), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalaing factors
    double alpha = 1.0f;
    double beta = 0.0f;

    // cuBLAS gemm
    // Calculate: C = (alpha * A) * B + (beta * C)
    // (arows x N) * (N x bcols) = (arows x bcols)
    // See A and B as row-major matrices and outputs a row-major matrix dC
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, bcols, arows, N, &alpha, dB, bcols, dA, N, &beta, dC, bcols);
    cudaMemcpy(hR, dC, arows * bcols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy_v2(handle);
}

void gemm_col(const float *hA, const float *hB, float *hR, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols) {
    float *hRt = (float *)malloc(arows * bcols * sizeof(float));
    // Allocate device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, arows * N * sizeof(float));
    cudaMalloc(&dB, N * bcols * sizeof(float));
    cudaMalloc(&dC, arows * bcols * sizeof(float));

    // Copy data to the device
    cudaMemcpy(dA, hA, arows * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * bcols * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalaing factors
    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS gemm
    // Calculate: C = (alpha * A) * B + (beta * C)
    // See A and B as column-major matrices and outputs a column-major matrix dC
    uint64_t lda = N;
    uint64_t ldb = bcols;
    uint64_t ldc = arows;  // fixed
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, arows, bcols, N, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
    cudaMemcpy(hRt, dC, arows * bcols * sizeof(float), cudaMemcpyDeviceToHost);
    common::translate(hRt, hR, arows, bcols);

    free(hRt);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy_v2(handle);
}

void gemm_col(const double *hA, const double *hB, double *hR, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols) {
    double *hRt = (double *)malloc(arows * bcols * sizeof(double));
    // Allocate device memory
    double *dA, *dB, *dC;
    cudaMalloc(&dA, arows * N * sizeof(double));
    cudaMalloc(&dB, N * bcols * sizeof(double));
    cudaMalloc(&dC, arows * bcols * sizeof(double));

    // Copy data to the device
    cudaMemcpy(dA, hA, arows * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * bcols * sizeof(double), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalaing factors
    double alpha = 1.0f;
    double beta = 0.0f;

    // cuBLAS gemm
    // Calculate: C = (alpha * A) * B + (beta * C)
    // See A and B as column-major matrices and outputs a column-major matrix dC
    uint64_t lda = N;
    uint64_t ldb = bcols;
    uint64_t ldc = arows;  // fixed
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, arows, bcols, N, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
    cudaMemcpy(hRt, dC, arows * bcols * sizeof(double), cudaMemcpyDeviceToHost);
    common::translate(hRt, hR, arows, bcols);

    free(hRt);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy_v2(handle);
}

// a: arows x N
// b: N x bcols
int main(int argc, char *argv[]) {
    // Host arrays
    T *hA = (T *)malloc(arows * N * sizeof(T));
    T *hB = (T *)malloc(N * bcols * sizeof(T));
    T *hR0 = (T *)malloc(arows * bcols * sizeof(T));
    T *hR1 = (T *)malloc(arows * bcols * sizeof(T));
    T *hR2 = (T *)malloc(arows * bcols * sizeof(T));
    T *hR3 = (T *)malloc(arows * bcols * sizeof(T));
    T *hR4 = (T *)malloc(arows * bcols * sizeof(T));
    T *hR5 = (T *)malloc(arows * bcols * sizeof(T));
    T *hR6 = (T *)malloc(arows * bcols * sizeof(T));

    // Initialize matrices
    auto start = std::chrono::high_resolution_clock::now();
    common::init(hA, arows, N);
    common::init(hB, N, bcols);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Randomization          : %0.6fs\n", diff.count());

    // Normal matrix multiplication
    start = std::chrono::high_resolution_clock::now();
    cumulmm(hA, hB, hR1, arows, N, bcols);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("cumulmm                : %0.6fs\n", diff.count());

    // Matrix multiplication with shared memory
    start = std::chrono::high_resolution_clock::now();
    cumulmm_shared_mem(hA, hB, hR2, arows, N, bcols);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("cumulmm_shared_mem     : %0.6fs\n", diff.count());

    // Matrix multiplication on rows
    start = std::chrono::high_resolution_clock::now();
    cumulmm_rows(hA, hB, hR3, arows, N, bcols);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("cumulmm_rows           : %0.6fs\n", diff.count());

    // Matrix multiplication on cols
    start = std::chrono::high_resolution_clock::now();
    cumulmm_cols(hA, hB, hR4, arows, N, bcols);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("cumulmm_cols           : %0.6fs\n", diff.count());

    // cuBLAS gemm row-major
    start = std::chrono::high_resolution_clock::now();
    gemm_row(hA, hB, hR5, arows, N, bcols);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("cublas<t>gemm row-major: %0.6fs\n", diff.count());

    // cuBLAS gemm col-major
    start = std::chrono::high_resolution_clock::now();
    gemm_col(hA, hB, hR6, arows, N, bcols);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("cublas<t>gemm col-major: %0.6fs\n", diff.count());

    // CPU-calculation
    start = std::chrono::high_resolution_clock::now();
    mulmm(hA, hB, hR0, arows, N, bcols);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("CPU-side               : %0.6fs\n", diff.count());

    // Check results
    start = std::chrono::high_resolution_clock::now();
    T diff01 = common::diff(hR0, hR1, arows, bcols);
    T diff02 = common::diff(hR0, hR2, arows, bcols);
    T diff03 = common::diff(hR0, hR3, arows, bcols);
    T diff04 = common::diff(hR0, hR4, arows, bcols);
    T diff05 = common::diff(hR0, hR5, arows, bcols);
    T diff06 = common::diff(hR0, hR6, arows, bcols);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("CPU Verification     : %0.6fs\n", diff.count());
    printf("Difference: %0.6f %0.6f %0.6f %0.6f %0.6f %0.6f\n", diff01, diff02, diff03, diff04, diff05, diff06);

    // Free host memory
    free(hA);
    free(hB);
    free(hR0);
    free(hR1);
    free(hR2);
    free(hR3);
    free(hR4);
    free(hR5);
    free(hR6);
    return 0;
}