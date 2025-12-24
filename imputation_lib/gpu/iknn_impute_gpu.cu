#include "common_kernels.cu"
#include "cuda_utils.cuh"
#include "iknn_impute_gpu.cuh"
#include "knn_impute_gpu.cuh"
#include "metrics.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace {

__global__ void iknn_compute_distances_kernel(const float *Xcur,
                                              const uint8_t *Mask, int N, int D,
                                              float *dist) {
  int i = blockIdx.x;
  if (i >= N)
    return;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int num_warps = blockDim.x / 32;

  for (int j = warp_id; j < N; j += num_warps) {
    if (i == j) {
      if (lane_id == 0)
        dist[i * N + j] = 1e9f;
      continue;
    }

    double sum = 0.0;
    int count = 0;

    for (int c = lane_id; c < D; c += 32) {
      if (Mask[i * D + c] == 1 && Mask[j * D + c] == 1) {
        double d = (double)Xcur[i * D + c] - (double)Xcur[j * D + c];
        sum += d * d;
        count++;
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
      count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }

    if (lane_id == 0) {
      dist[i * N + j] =
          (count > 0) ? (float)sqrt(sum * ((double)D / count)) : 1e9f;
    }
  }
}

__global__ void iknn_fill_kernel(const float *Xprev, float *Xcurr,
                                 const uint8_t *Mask, const float *dist, int N,
                                 int D, int K) {
  int i = blockIdx.x;
  if (i >= N)
    return;

  for (int c = threadIdx.x; c < D; c += blockDim.x) {
    if (Mask[i * D + c] == 1)
      continue;

    const int MAXK = 32;
    int eff_K = (K < MAXK) ? K : MAXK;
    float best_d[32];
    int best_j[32];
    for (int k = 0; k < eff_K; k++) {
      best_d[k] = 1e9f;
      best_j[k] = -1;
    }

    for (int j = 0; j < N; j++) {
      if (i == j || Mask[j * D + c] == 0)
        continue;
      float d = dist[i * N + j];
      if (d >= 1e8f)
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

    double num = 0, den = 0;
    for (int k = 0; k < eff_K; k++) {
      int neighbor_idx = best_j[k];
      if (neighbor_idx < 0)
        continue;
      double w = 1.0 / ((double)best_d[k] + 1e-6);
      num += w * (double)Xprev[neighbor_idx * D + c];
      den += w;
    }
    if (den > 0)
      Xcurr[i * D + c] = (float)(num / den);
    else
      Xcurr[i * D + c] = 0.0f;
  }
}

__global__ void diff_masked_kernel(const float *X_prev, const float *X_curr,
                                   const uint8_t *mask, float *diff, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (mask[idx] == 0) {
      float d = X_prev[idx] - X_curr[idx];
      diff[idx] = d * d;
    } else {
      diff[idx] = 0.0f;
    }
  }
}

} // namespace

namespace impute {

void iknn_impute(float *d_X, const uint8_t *d_mask, int N, int D, int K,
                 int max_iter, float tol, cudaStream_t stream) {
  if (!d_X || !d_mask)
    throw std::runtime_error("iknn_impute: null pointer");

  int size = N * D;
  float *d_prev;
  CUDA_CHECK(cudaMalloc(&d_prev, sizeof(float) * size));
  float *d_diff;
  CUDA_CHECK(cudaMalloc(&d_diff, sizeof(float) * size));
  float *d_dist;
  CUDA_CHECK(cudaMalloc(&d_dist, sizeof(float) * N * N));

  for (int it = 0; it < max_iter; it++) {
    CUDA_CHECK(cudaMemcpyAsync(d_prev, d_X, sizeof(float) * size,
                               cudaMemcpyDeviceToDevice, stream));

    iknn_compute_distances_kernel<<<N, 256, 0, stream>>>(d_prev, d_mask, N, D,
                                                         d_dist);
    iknn_fill_kernel<<<N, 256, 0, stream>>>(d_prev, d_X, d_mask, d_dist, N, D,
                                            K);

    diff_masked_kernel<<<(size + 255) / 256, 256, 0, stream>>>(
        d_prev, d_X, d_mask, d_diff, size);
    float diff_value = metrics::sum_reduction(d_diff, size, stream);
    if (std::sqrt(diff_value / size) < tol)
      break;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(d_prev));
  CUDA_CHECK(cudaFree(d_diff));
  CUDA_CHECK(cudaFree(d_dist));
}

} // namespace impute
