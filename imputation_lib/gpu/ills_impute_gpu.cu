#pragma once
#include "common_kernels.cu"
#include "cuda_utils.cuh"
#include "ills_impute_gpu.cuh"
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace impute {

constexpr int ILLS_MAX_K = 32;

// Robust Cholesky with NaN check and Double Precision
__device__ inline void solve_ills_robust(int K, double *A, double *b,
                                         float *x) {
  for (int i = 0; i < K; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
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
    double sum = 0.0;
    for (int j = 0; j < i; j++)
      sum += A[i * K + j] * y[j];
    y[i] = (b[i] - sum) / A[i * K + i];
    if (isnan(y[i]) || isinf(y[i]))
      y[i] = 0.0;
  }

  for (int i = K - 1; i >= 0; i--) {
    double sum = 0.0;
    for (int j = i + 1; j < K; j++)
      sum += A[j * K + i] * (double)x[j]; // Result in x
    double res = (y[i] - sum) / A[i * K + i];
    if (isnan(res) || isinf(res))
      res = 0.0;
    x[i] = (float)res;
  }
}

__global__ void solve_impute_ills_kernel(const float *X_in, float *X_out,
                                         const uint8_t *Mask,
                                         const int *Neighbors, int N, int D,
                                         int K) {
  int i = blockIdx.x;
  if (i >= N)
    return;

  __shared__ double ATA[32 * 32];
  __shared__ double ATy[32];
  __shared__ float weights[32];

  int tx = threadIdx.x;

  // 1. Initialize
  for (int idx = tx; idx < K * K; idx += blockDim.x)
    ATA[idx] = 0.0;
  for (int idx = tx; idx < K; idx += blockDim.x)
    ATy[idx] = 0.0;
  __syncthreads();

  // 2. Tiled Accumulation (Shared Memory caching)
  __shared__ float tile_vals[32 * 32];
  __shared__ float target_tile[32];

  for (int c_start = 0; c_start < D; c_start += 32) {
    int n_curr_c = (D - c_start < 32) ? (D - c_start) : 32;

    for (int idx = tx; idx < K * 32; idx += blockDim.x) {
      int k_idx = idx / 32;
      int col_off = idx % 32;
      if (k_idx < K && col_off < n_curr_c) {
        int nid = Neighbors[i * K + k_idx];
        tile_vals[idx] = (nid != -1) ? X_in[nid * D + c_start + col_off] : 0.0f;
      } else {
        tile_vals[idx] = 0.0f;
      }
    }
    if (tx < 32) {
      target_tile[tx] = (c_start + tx < D) ? X_in[i * D + c_start + tx] : 0.0f;
    }
    __syncthreads();

    // Accumulate ATy
    for (int k = tx; k < K; k += blockDim.x) {
      double local_sum = 0;
      for (int c = 0; c < n_curr_c; ++c) {
        if (Mask[i * D + c_start + c] == 1) {
          local_sum += (double)tile_vals[k * 32 + c] * target_tile[c];
        }
      }
      ATy[k] += local_sum;
    }

    // Accumulate ATA
    for (int idx = tx; idx < K * K; idx += blockDim.x) {
      int r = idx / K;
      int c_idx = idx % K;
      double local_sum = 0;
      for (int c = 0; c < n_curr_c; ++c) {
        if (Mask[i * D + c_start + c] == 1) {
          local_sum +=
              (double)tile_vals[r * 32 + c] * tile_vals[c_idx * 32 + c];
        }
      }
      ATA[idx] += local_sum;
    }
    __syncthreads();
  }

  if (tx == 0) {
    // Match CPU Ridge: fixed lambda
    double lambda = 1e-4; // Default ridge_alpha from CPU
    for (int k = 0; k < K; ++k)
      ATA[k * K + k] += lambda;

    solve_ills_robust(K, ATA, ATy, weights);
  }
  __syncthreads();

  for (int c = tx; c < D; c += blockDim.x) {
    if (Mask[i * D + c] == 0) {
      double pred = 0.0;
      for (int k = 0; k < K; ++k) {
        int nid = Neighbors[i * K + k];
        if (nid != -1)
          pred += (double)weights[k] * X_in[nid * D + c];
      }
      if (!isnan(pred) && !isinf(pred)) {
        // Soft bounding to prevent explosion
        if (pred > 1e6)
          pred = 1e6;
        if (pred < -1e6)
          pred = -1e6;
        X_out[i * D + c] = (float)pred;
      }
    }
  }
}

// Initial Mean Fill Kernel is in common_kernels.cu

void ills_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                      int max_iter, cudaStream_t stream) {
  if (K > ILLS_MAX_K)
    return;

  // 1. Initial Mean Fill
  int threads = 128;
  int blocks = (D + threads - 1) / threads;
  mean_fill_kernel<<<blocks, threads, 0, stream>>>(d_X, d_mask, N, D);

  int *d_Neighbors;
  cudaMalloc(&d_Neighbors, N * K * sizeof(int));

  float *d_X_next;
  cudaMalloc(&d_X_next, N * D * sizeof(float));

  for (int it = 0; it < max_iter; ++it) {
    cudaMemcpyAsync(d_X_next, d_X, N * D * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    select_neighbors_ills_kernel<<<N, 128, 0, stream>>>(d_X, d_Neighbors, N, D,
                                                        K);
    // Modified kernel to write to X_next
    solve_impute_ills_kernel<<<N, 128, 0, stream>>>(d_X, d_X_next, d_mask,
                                                    d_Neighbors, N, D, K);
    cudaMemcpyAsync(d_X, d_X_next, N * D * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
  }

  cudaFree(d_Neighbors);
  cudaFree(d_X_next);
}

} // namespace impute
