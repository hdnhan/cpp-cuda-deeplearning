#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>

namespace common {

template <typename T>
void init(T *a, const uint64_t &N, const T &low = 0, const T &high = 1);

template <typename T>
void init(T *a, const uint64_t &rows, const uint64_t &cols, const T &low = 0, const T &high = 1);

template <typename T>
T diff(const T *a, const T *b, const uint64_t &N);

template <typename T>
T diff(const T *a, const T *b, const uint64_t &rows, const uint64_t &cols);

// 01-axpy: add a vector to a vector
template <typename T>
void addvv(const T *a, const T *b, T *c, const uint64_t &N);

template <typename T>
__global__ void cuaddvv(const T *a, const T *b, T *c, const uint64_t N);

// 01-dot: inner product of two vectors
template <typename T>
T dotvv(const T *a, const T *b, const uint64_t &N);

// 02-gemv: matrix-vector multiplication
template <typename T>
void mulmv(const T *A, const T *x, T *y, const uint64_t &M, const uint64_t &N);

template <typename T>
__global__ void cumulmv(const T *A, const T *x, T *y, const uint64_t M, const uint64_t N);

// 03-gemm: matrix-matrix multiplication
template <typename T>
void mulmm(const T *A, const T *B, T *C, const uint64_t &arows, const uint64_t &N, const uint64_t &bcols);

template <typename T>
__global__ void cumulmm(const T *A, const T *B, T *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);

template <typename T>
__global__ void cumulmm_shared_mem(const T *A, const T *B, T *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);

template <typename T>
__global__ void cumulmm_rows(const T *A, const T *B, T *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);

template <typename T>
__global__ void cumulmm_cols(const T *A, const T *B, T *C, const uint64_t arows, const uint64_t N, const uint64_t bcols);

template <typename T>
void transpose(const T *A, T *At, const uint64_t &rows, const uint64_t &cols);

template <typename T>
void translate(const T *A, T *At, const uint64_t &rows, const uint64_t &cols);

}  // namespace common