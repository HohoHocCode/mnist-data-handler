#include "knn_impute_gpu.cuh"

#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#include "cuda_utils.cuh"

namespace {

constexpr int KNN_BLOCK = 256;
constexpr float BIG = 1e9f;

// Compute distance(i,j) using Warp-Parallel Reduction over D
__global__ void compute_distances_kernel(const float *__restrict__ X,
                                         const uint8_t *__restrict__ mask,
                                         int N, int D,
                                         float *__restrict__ dist) {
  int i = blockIdx.x;
  if (i >= N)
    return;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int num_warps = blockDim.x / 32;

  for (int j = warp_id; j < N; j += num_warps) {
    if (i == j) {
      if (lane_id == 0)
        dist[i * N + j] = BIG;
      continue;
    }

    float diff_sum = 0.0f;
    int count_sum = 0;

    for (int c = lane_id; c < D; c += 32) {
      int idx_i = i * D + c;
      int idx_j = j * D + c;

      uint8_t m_i = mask[idx_i];
      uint8_t m_j = mask[idx_j];

      if (m_i == 1 && m_j == 1) {
        float d = X[idx_i] - X[idx_j];
        diff_sum += d * d;
        count_sum++;
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      diff_sum += __shfl_down_sync(0xFFFFFFFF, diff_sum, offset);
      count_sum += __shfl_down_sync(0xFFFFFFFF, count_sum, offset);
    }

    if (lane_id == 0) {
      dist[i * N + j] =
          (count_sum == 0) ? BIG : sqrtf(diff_sum * ((float)D / count_sum));
    }
  }
}

// Impute missing entries using K nearest neighbors
__global__ void knn_fill_kernel(float *__restrict__ X,
                                const uint8_t *__restrict__ mask,
                                const float *__restrict__ dist, int N, int D,
                                int K) {
  int i = blockIdx.x;
  if (i >= N)
    return;

  const int MAXK = 32;
  int eff_K = min(K, MAXK);

  for (int c = threadIdx.x; c < D; c += blockDim.x) {
    int idx = i * D + c;

    if (mask[idx] == 1)
      continue;

    float best_d[32];
    int best_j[32];

    for (int k = 0; k < eff_K; k++) {
      best_d[k] = BIG;
      best_j[k] = -1;
    }

    for (int j = 0; j < N; j++) {
      if (mask[j * D + c] == 0)
        continue;

      float d = dist[i * N + j];
      if (d >= BIG)
        continue;

      for (int k = 0; k < eff_K; k++) {
        if (d < best_d[k]) {
          for (int t = eff_K - 1; t > k; t--) {
            best_d[t] = best_d[t - 1];
            best_j[t] = best_j[t - 1];
          }
          best_d[k] = d;
          best_j[k] = j;
          break;
        }
      }
    }

    float num = 0.0f;
    float den = 0.0f;

    for (int k = 0; k < eff_K; k++) {
      int neighbor = best_j[k];
      if (neighbor < 0)
        continue;
      float w = 1.0f / (best_d[k] + 1e-6f);
      num += w * X[neighbor * D + c];
      den += w;
    }

    X[idx] = (den > 0 ? num / den : 0.0f);
  }
}

} // namespace

namespace impute {

void knn_impute(float *d_X_rowmajor, const uint8_t *d_mask, int N, int D, int K,
                cudaStream_t stream) {
  if (!d_X_rowmajor || !d_mask)
    throw std::runtime_error("knn_impute: null pointer");

  if (K <= 0)
    throw std::runtime_error("knn_impute: invalid K");

  float *d_dist = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dist, sizeof(float) * N * N));

  {
    dim3 grid(N);
    dim3 block(KNN_BLOCK);
    compute_distances_kernel<<<grid, block, 0, stream>>>(d_X_rowmajor, d_mask,
                                                         N, D, d_dist);
  }

  {
    dim3 grid(N);
    dim3 block(KNN_BLOCK);
    knn_fill_kernel<<<grid, block, 0, stream>>>(d_X_rowmajor, d_mask, d_dist, N,
                                                D, K);
  }

  CUDA_CHECK(cudaFree(d_dist));
}

} // namespace impute
