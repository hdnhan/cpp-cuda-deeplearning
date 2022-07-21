#include "common.h"

namespace common {

template <typename T>
void init(T *a, const uint64_t &N, const T &low, const T &high) {
    srand((uint64_t)time(NULL));
    for (uint64_t i = 0; i < N; i++)
        a[i] = (T)rand() / (T)RAND_MAX * (high - low) + low;
}
template void init<float>(float *a, const uint64_t &N, const float &low, const float &high);
template void init<double>(double *a, const uint64_t &N, const double &low, const double &high);

template <typename T>
void init(T *a, const uint64_t &rows, const uint64_t &cols, const T &low, const T &high) {
    srand((uint64_t)time(NULL));
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            a[i * cols + j] = (T)rand() / (T)RAND_MAX * (high - low) + low;
}
template void init<float>(float *a, const uint64_t &rows, const uint64_t &cols, const float &low, const float &high);
template void init<double>(double *a, const uint64_t &rows, const uint64_t &cols, const double &low, const double &high);

template <typename T>
T diff(const T *a, const T *b, const uint64_t &N) {
    T max_diff = 0;
    for (uint64_t i = 0; i < N; i++)
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    return max_diff;
}
template float diff<float>(const float *a, const float *b, const uint64_t &N);
template double diff<double>(const double *a, const double *b, const uint64_t &N);

template <typename T>
T diff(const T *a, const T *b, const uint64_t &rows, const uint64_t &cols) {
    T max_diff = 0;
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            max_diff = std::max(max_diff, std::abs(a[i * cols + j] - b[i * cols + j]));
    return max_diff;
}
template float diff<float>(const float *a, const float *b, const uint64_t &rows, const uint64_t &cols);
template double diff<double>(const double *a, const double *b, const uint64_t &rows, const uint64_t &cols);

// 01-axpy: add a vector to a vector
template <typename T>
void addvv(const T *a, const T *b, T *c, const uint64_t &N) {
    for (uint64_t i = 0; i < N; i++)
        c[i] = a[i] + b[i];
}
template void addvv<float>(const float *a, const float *b, float *c, const uint64_t &N);
template void addvv<double>(const double *a, const double *b, double *c, const uint64_t &N);

template <typename T>
__global__ void cuaddvv(const T *a, const T *b, T *c, const uint64_t N) {
    uint64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}
template __global__ void cuaddvv<float>(const float *a, const float *b, float *c, const uint64_t N);
template __global__ void cuaddvv<double>(const double *a, const double *b, double *c, const uint64_t N);

// 01-dot: inner product of two vectors
template <typename T>
T dotvv(const T *a, const T *b, const uint64_t &N) {
    T sum = 0.0f;
    for (uint64_t i = 0; i < N; i++)
        sum += a[i] * b[i];
    return sum;
}
template float dotvv<float>(const float *a, const float *b, const uint64_t &N);
template double dotvv<double>(const double *a, const double *b, const uint64_t &N);

// 02-gemv: matrix-vector multiplication
template <typename T>
void mulmv(const T *A, const T *x, T *y, const uint64_t &M, const uint64_t &N){
    for (uint64_t i = 0; i < M; i++) {
        y[i] = 0.0f;
        for (uint64_t j = 0; j < N; j++)
            y[i] += A[i * N + j] * x[j];
    }
}
template void mulmv<float>(const float *A, const float *x, float *y, const uint64_t &M, const uint64_t &N);
template void mulmv<double>(const double *A, const double *x, double *y, const uint64_t &M, const uint64_t &N);

template <typename T>
__global__ void cumulmv(const T *A, const T *x, T *y, const uint64_t M, const uint64_t N) {
    uint64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < M) {
        y[tid] = 0.0f;
        for (uint64_t k = 0; k < N; k++)
            y[tid] += A[tid * N + k] * x[k];
    }
}
template __global__ void cumulmv<float>(const float *A, const float *x, float *y, const uint64_t M, const uint64_t N);
template __global__ void cumulmv<double>(const double *A, const double *x, double *y, const uint64_t M, const uint64_t N);

// 03-gemm: matrix-matrix multiplication
template <typename T>
void mulmm(const T *A, const T *B, T *C, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols) {
    for (uint64_t i = 0; i < arows; i++)
        for (uint64_t j = 0; j < bcols; j++) {
            T tmp = 0;
            for (uint64_t k = 0; k < N; k++)
                tmp += A[i * N + k] * B[k * bcols + j];
            C[i * bcols + j] = tmp;
        }
}
template void mulmm<float>(const float *A, const float *B, float *C, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols);
template void mulmm<double>(const double *A, const double *B, double *C, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols);

template <typename T>
__global__ void cumulmm(const T *A, const T *B, T *C, const uint64_t arows, const uint64_t N, const uint64_t bcols) {
    uint64_t rid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t cid = blockIdx.y * blockDim.y + threadIdx.y;

    if (rid >= arows || cid >= bcols) return;
    C[rid * bcols + cid] = 0;
    for (uint64_t k = 0; k < N; k++)
        C[rid * bcols + cid] += A[rid * N + k] * B[k * bcols + cid];
}
template __global__ void cumulmm<float>(const float *A, const float *B, float *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);
template __global__ void cumulmm<double>(const double *A, const double *B, double *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);

// Shared memory is used to store partial results
template <typename T>
__global__ void cumulmm_shared_mem(const T *A, const T *B, T *C, const uint64_t arows, const uint64_t N, const uint64_t bcols) {
    uint64_t rid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t cid = blockIdx.y * blockDim.y + threadIdx.y;

    // Statically allocated shared memory
    __shared__ T sa[1 << 10];  // = RTHREADS x CTHREADS
    __shared__ T sb[1 << 10];  // = RTHREADS x CTHREADS

    T tmp = 0;
    for (uint64_t i = 0; i < N; i += blockDim.y) {
        sa[threadIdx.x * blockDim.y + threadIdx.y] = 0;
        sb[threadIdx.x * blockDim.y + threadIdx.y] = 0;
        if (rid * N + i + threadIdx.y < arows * N)
            sa[threadIdx.x * blockDim.y + threadIdx.y] = A[rid * N + i + threadIdx.y];
        if (i * bcols + threadIdx.x * bcols + cid < N * bcols)
            sb[threadIdx.x * blockDim.y + threadIdx.y] = B[i * bcols + threadIdx.x * bcols + cid];

        __syncthreads();

        for (uint64_t j = 0; j < blockDim.x; j++)
            tmp += sa[threadIdx.x * blockDim.y + j] * sb[j * blockDim.y + threadIdx.y];
        __syncthreads();
    }
    if (rid < arows && cid < bcols)
        C[rid * bcols + cid] = tmp;
}
template __global__ void cumulmm_shared_mem<float>(const float *A, const float *B, float *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);
template __global__ void cumulmm_shared_mem<double>(const double *A, const double *B, double *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);

// Matrix multiplication on rows
template <typename T>
__global__ void cumulmm_rows(const T *A, const T *B, T *C, const uint64_t arows, const uint64_t N, const uint64_t bcols) {
    uint64_t rid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t cid = blockIdx.y * blockDim.y + threadIdx.y;

    if (rid >= arows || cid >= bcols) return;
    C[rid * bcols + cid] = 0;
    for (int k = 0; k < N; k++)
        C[rid * bcols + cid] += A[rid * N + k] * B[cid * N + k];
}
template __global__ void cumulmm_rows<float>(const float *A, const float *B, float *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);
template __global__ void cumulmm_rows<double>(const double *A, const double *B, double *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);

// Matrix multiplication on cols
template <typename T>
__global__ void cumulmm_cols(const T *A, const T *B, T *C, const uint64_t arows, const uint64_t N, const uint64_t bcols) {
    uint64_t rid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t cid = blockIdx.y * blockDim.y + threadIdx.y;

    if (rid >= arows || cid >= bcols) return;
    C[rid * bcols + cid] = 0;
    for (int k = 0; k < N; k++)
        C[rid * bcols + cid] += A[k * arows + rid] * B[k * bcols + cid];
}
template __global__ void cumulmm_cols<float>(const float *A, const float *B, float *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);
template __global__ void cumulmm_cols<double>(const double *A, const double *B, double *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);

// Matrix transpose
template <typename T>
void transpose(const T *A, T *At, const uint64_t &rows, const uint64_t &cols) {
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            At[j * rows + i] = A[i * cols + j];
}
template void transpose<float>(const float *A, float *At, const uint64_t &rows, const uint64_t &cols);
template void transpose<double>(const double *A, double *At, const uint64_t &rows, const uint64_t &cols);

template <typename T>
void translate(const T *A, T *At, const uint64_t &rows, const uint64_t &cols) {
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            At[i * cols + j] = A[j * rows + i];
}
template void translate<float>(const float *A, float *At, const uint64_t &rows, const uint64_t &cols);
template void translate<double>(const double *A, double *At, const uint64_t &rows, const uint64_t &cols);

}  // namespace common