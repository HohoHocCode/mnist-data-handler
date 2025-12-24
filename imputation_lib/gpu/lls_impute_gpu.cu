#pragma once
#include "lls_impute_gpu.cuh"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "common_kernels.cu"
#include "cuda_utils.cuh"


namespace impute {

constexpr int MAX_K_LLS = 32;

__global__ void solve_impute_kernel(float *X, const uint8_t *Mask,
                                    const int *Neighbors, int N, int D, int K) {
  int i = blockIdx.x;
  if (i >= N)
    return;

  __shared__ float ATA[32 * 32];
  __shared__ float ATy[32];
  __shared__ float weights[32];

  int tx = threadIdx.x;
  int lane = tx % 32;
  int wid = tx / 32;
  int num_warps = blockDim.x / 32;

  // 1. Initialize
  for (int idx = tx; idx < K * K; idx += blockDim.x)
    ATA[idx] = 0.0f;
  for (int idx = tx; idx < K; idx += blockDim.x)
    ATy[idx] = 0.0f;
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
        tile_vals[idx] = (nid != -1) ? X[nid * D + c_start + col_off] : 0.0f;
      } else {
        tile_vals[idx] = 0.0f;
      }
    }
    if (tx < 32) {
      target_tile[tx] = (c_start + tx < D) ? X[i * D + c_start + tx] : 0.0f;
    }
    __syncthreads();

    // Accumulate ATy
    for (int k = tx; k < K; k += blockDim.x) {
      float local_sum = 0;
      for (int c = 0; c < n_curr_c; ++c) {
        if (Mask[i * D + c_start + c] == 1 && tile_vals[k * 32 + c] != 0.0f) {
          local_sum += tile_vals[k * 32 + c] * target_tile[c];
        }
      }
      ATy[k] += local_sum;
    }

    // Accumulate ATA
    for (int idx = tx; idx < K * K; idx += blockDim.x) {
      int r = idx / K;
      int c_idx = idx % K;
      float local_sum = 0;
      for (int c = 0; c < n_curr_c; ++c) {
        if (Mask[i * D + c_start + c] == 1) {
          local_sum += tile_vals[r * 32 + c] * tile_vals[c_idx * 32 + c];
        }
      }
      ATA[idx] += local_sum;
    }
    __syncthreads();
  }

  if (tx == 0) {
    // Ridge
    for (int k = 0; k < K; ++k)
      ATA[k * K + k] += 1e-2f;
    solve_cholesky_kxk(K, ATA, ATy, weights);
  }
  __syncthreads();

  // Impute
  for (int c = tx; c < D; c += blockDim.x) {
    if (Mask[i * D + c] == 0) {
      float pred = 0.0f;
      for (int k = 0; k < K; ++k) {
        int nid = Neighbors[i * K + k];
        if (nid != -1)
          pred += weights[k] * X[nid * D + c];
      }
      if (!isnan(pred) && !isinf(pred))
        X[i * D + c] = pred;
    }
  }
}

void lls_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                     cudaStream_t stream) {
  if (K > MAX_K_LLS) {
    printf("Error: K=%d exceeds MAX_K=%d for LLS GPU\n", K, MAX_K_LLS);
    return;
  }

  int *d_Neighbors;
  cudaMalloc(&d_Neighbors, N * K * sizeof(int));

  // 1. Select Neighbors (Using optimized Common Kernel with Strict Filter)
  select_neighbors_lls_kernel<<<N, 128, 0, stream>>>(d_X, d_mask, d_Neighbors,
                                                     N, D, K);

  // 2. Solve and Impute
  solve_impute_kernel<<<N, 128, 0, stream>>>(d_X, d_mask, d_Neighbors, N, D, K);

  cudaFree(d_Neighbors);
}

} // namespace impute
