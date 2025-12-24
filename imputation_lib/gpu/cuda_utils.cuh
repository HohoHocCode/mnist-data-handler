#pragma once
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#ifndef CUDA_CHECK
#define CUDA_CHECK(e)                                                          \
  {                                                                            \
    if ((e) != cudaSuccess) {                                                  \
      printf("CUDA ERR: %s at %s:%d\n", cudaGetErrorString(e), __FILE__,       \
             __LINE__);                                                        \
    }                                                                          \
  }
#endif
#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(e)                                                        \
  {                                                                            \
    if ((e) != CUBLAS_STATUS_SUCCESS) {                                        \
      printf("cuBLAS ERR at %s:%d\n", __FILE__, __LINE__);                     \
    }                                                                          \
  }
#endif
#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(e)                                                      \
  {                                                                            \
    if ((e) != CUSOLVER_STATUS_SUCCESS) {                                      \
      printf("cuSOLVER ERR at %s:%d\n", __FILE__, __LINE__);                   \
    }                                                                          \
  }
#endif
#endif

namespace impute {

// Helper: Tiny Linear System Solver (Cholesky) for KxK matrix
// Running in a single thread
// Solves Ax = b. Result stored in x.
// A is modified (stores L).
__device__ inline void solve_cholesky_kxk(int K, float *A, float *b, float *x) {
  // A is KxK row-major, symmetric positive definite

  // 1. Cholesky Decomposition: L * L^T = A
  // Process in-place in A (store L in lower triangle)
  for (int i = 0; i < K; i++) {
    for (int j = 0; j <= i; j++) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++)
        sum += A[i * K + k] * A[j * K + k];

      if (i == j) {
        float val = A[i * K + i] - sum;
        if (val <= 0.0f)
          val = 1e-6f; // Regularize
        A[i * K + j] = sqrtf(val);
      } else {
        A[i * K + j] = (1.0f / A[j * K + j] * (A[i * K + j] - sum));
      }
    }
  }

  // 2. Forward Substitute: L * y = b
  // Store y in x temporarily
  for (int i = 0; i < K; i++) {
    float sum = 0.0f;
    for (int j = 0; j < i; j++)
      sum += A[i * K + j] * x[j];
    x[i] = (b[i] - sum) / A[i * K + i];
  }

  // 3. Backward Substitute: L^T * x_final = y
  // L^T is upper triangular part of A transposed (which is implicit from L
  // lower) L^T[i, j] = A[j, i] (but we stored L in A at [i,j] for i>=j) L^T has
  // elements L[j,i]. Row i of L^T is Column i of L. Elements are A[j, i] where
  // j >= i.

  for (int i = K - 1; i >= 0; i--) {
    float sum = 0.0f;
    for (int j = i + 1; j < K; j++)
      // U[i,j] = L[j,i] = A[j*K + i]
      sum += A[j * K + i] * x[j];

    // diag L[i,i] = A[i*K+i]
    x[i] = (x[i] - sum) / A[i * K + i];
  }
}

// Matrix Multiplication C = A * B (All KxK) - Device function
__device__ inline void matmul_kxk(int K, const float *A, const float *B,
                                  float *C) {
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k)
        sum += A[i * K + k] * B[k * K + j];
      C[i * K + j] = sum;
    }
  }
}

// Solve small linear system KxK via Cholesky (Double Precision)
__device__ inline void solve_linear_system_device(int K, double *A, double *b,
                                                  float *x) {
  for (int i = 0; i < K; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0;
      for (int k = 0; k < j; k++)
        sum += A[i * K + k] * A[j * K + k];
      if (i == j) {
        double val = A[i * K + i] - sum;
        if (val < 1e-12 || isnan(val) || isinf(val))
          val = 1e-12;
        A[i * K + j] = sqrt(val);
      } else {
        double diag = A[j * K + j];
        if (diag < 1e-12)
          diag = 1e-12;
        A[i * K + j] = (1.0 / diag * (A[i * K + j] - sum));
        if (isnan(A[i * K + j]) || isinf(A[i * K + j]))
          A[i * K + j] = 0.0;
      }
    }
  }
  double y[32];
  for (int i = 0; i < K; i++) {
    double sum = 0;
    for (int j = 0; j < i; j++)
      sum += A[i * K + j] * y[j];
    y[i] = (b[i] - sum) / A[i * K + i];
    if (isnan(y[i]) || isinf(y[i]))
      y[i] = 0.0;
  }
  for (int i = K - 1; i >= 0; i--) {
    double sum = 0;
    for (int j = i + 1; j < K; j++)
      sum += A[j * K + i] * (double)x[j];
    double res = (y[i] - sum) / A[i * K + i];
    if (isnan(res) || isinf(res))
      res = 0.0;
    x[i] = (float)res;
  }
}
} // namespace impute
