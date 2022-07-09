#include <cublas_v2.h>

#include <cassert>

// Initialize a vector
void init(float *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (float)(rand() % 100);
    }
}

// Verify the result
void verify(float *a, float *b, float *c, float factor, int n) {
    for (int i = 0; i < n; i++) {
        assert(c[i] == factor * a[i] + b[i]);
    }
}

int main() {
    int n = 1 << 16;
    size_t bytes = n * sizeof(float);

    float *ha, *hb, *hc;
    ha = (float *)malloc(bytes);
    hb = (float *)malloc(bytes);
    hc = (float *)malloc(bytes);

    float *da, *db;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);

    init(ha, n);
    init(hb, n);

    // Create and initialize a new context
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // Copy the vectors over to the device
    cublasSetVector(n, sizeof(float), ha, 1, da, 1);
    cublasSetVector(n, sizeof(float), hb, 1, db, 1);

    // Launch simple saxpy kernel (single precision a * x + y)
    // Function signature: handle, # elements n, A, increment, B, increment
    const float scale = 2.0f;
    cublasSaxpy(handle, n, &scale, da, 1, db, 1);

    // Copy the result vector back out
    cublasGetVector(n, sizeof(float), db, 1, hc, 1);

    verify(ha, hb, hc, scale, n);

    // Clean up the created handle
    cublasDestroy(handle);

    // Release allocated memory
    cudaFree(da);
    cudaFree(db);
    free(ha);
    free(hb);

    return 0;
}