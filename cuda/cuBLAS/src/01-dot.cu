#include <cublas_v2.h>
#include <chrono>
#include "common.h"

typedef double T;  // float vs double
const uint64_t N = 1 << 28;

T dotvv(const T *ha, const T *hb, const uint64_t &N) {
    return common::dotvv(ha, hb, N);
}

float dot(const float *ha, const float *hb, const uint64_t &N) {
    // Allocate memory on the device-side (GPU-side)
    float *da, *db;
    float res = 0;

    cudaMalloc(&da, N * sizeof(float));
    cudaMalloc(&db, N * sizeof(float));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // Perform the dot product
    cublasSdot(handle, N, da, 1, db, 1, &res);

    // Free the device memory
    cudaFree(da);
    cudaFree(db);
    cublasDestroy_v2(handle);
    return res;
}

double dot(const double *ha, const double *hb, const uint64_t &N) {
    // Allocate memory on the device-side (GPU-side)
    double *da, *db;
    double res = 0;

    cudaMalloc(&da, N * sizeof(double));
    cudaMalloc(&db, N * sizeof(double));

    // Copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(da, ha, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(double), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // Perform the dot product
    cublasDdot(handle, N, da, 1, db, 1, &res);

    // Free the device memory
    cudaFree(da);
    cudaFree(db);
    cublasDestroy_v2(handle);
    return res;
}

int main(int argc, char *argv[]) {
    // Vectors for holding the host-side (CPU-side) data
    T *ha = (T *)malloc(N * sizeof(T));
    T *hb = (T *)malloc(N * sizeof(T));
    T res0 = 0, res1 = 0;

    // Initialize random numbers in each array
    auto start = std::chrono::high_resolution_clock::now();
    common::init(ha, N);
    common::init(hb, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<T> diff = end - start;
    printf("Randomization: %0.6fs\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    res0 = dotvv(ha, hb, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("CPU-side     : %0.6fs\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    res1 = dot(ha, hb, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("cublasSdot   : %0.6fs\n", diff.count());
    std::cout << "Difference: " << res0 - res1 << std::endl;

    // Free host memory
    free(ha);
    free(hb);
    return 0;
}