#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>

// Column-major matrix
void verify_col(const float *a, const float *b, const float *c, const unsigned int M, const unsigned int K, const unsigned int N) {
    float temp;
    float epsilon = 1e-5;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp = 0;
            for (int z = 0; z < K; z++) {
                temp += a[z * M + i] * b[j * K + z];
            }
            std::cout.precision(17);
            std::cout << i << " " << j << " " << temp << " " << c[j * M + i] << " " << abs(temp - c[j * M + i]) << std::endl;
            assert(abs(c[j * M + i] - temp) < epsilon);
        }
    }
}

// Row-major matrix
void verify_row(const float *a, const float *b, const float *c, const unsigned int M, const unsigned int K, const unsigned int N) {
    float temp;
    float epsilon = 1e-2;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            temp = 0;
            for (int z = 0; z < K; z++) {
                temp += a[i * K + z] * b[z * N + j];
            }
            std::cout.precision(17);
            std::cout << i << " " << j << " " << temp << " " << c[i * N + j] << " " << abs(temp - c[i * N + j]) << std::endl;
            assert(abs(c[i * N + j] - temp) < epsilon);
        }
    }
}

int main() {
    /*
    Problem size:
        A: M x K
        B: K x N
        => C = A * B
    */
    const unsigned int M = 1 << 10;
    const unsigned int K = 1 << 11;
    const unsigned int N = 1 << 12;

    // Declare pointers to matrices on device and host
    float *ha, *hb, *hc;
    float *da, *db, *dc;

    // Allocate memory
    ha = (float *)malloc(M * K * sizeof(float));
    hb = (float *)malloc(K * N * sizeof(float));
    hc = (float *)malloc(M * N * sizeof(float));
    cudaMalloc(&da, M * K * sizeof(float));
    cudaMalloc(&db, K * N * sizeof(float));
    cudaMalloc(&dc, M * N * sizeof(float));

    // Pseudo random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // Set the seed
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    // Fill the matrix with random numbers on the device
    curandGenerateUniform(prng, da, M * K);
    curandGenerateUniform(prng, db, K * N);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scalaing factors
    float alpha = 1.0f;
    float beta = 0.0f;

    // Calculate: C = (alpha * A) * B + (beta * C)
    // (M x K) * (K x N) = (M x N)

    // This outputs a column-major matrix dc
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, da, M, db, K, &beta, dc, M);
    // This outputs a row-major matrix dc
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, db, N, da, K, &beta, dc, N);

    // Copy back the three matrices
    cudaMemcpy(ha, da, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb, db, K * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hc, dc, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify solution
    // verify_col(ha, hb, hc, M, K, N);
    verify_row(ha, hb, hc, M, K, N);

    printf("COMPLETED SUCCESSFULLY\n");

    return 0;
}