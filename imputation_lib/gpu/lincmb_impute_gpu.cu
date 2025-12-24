#pragma once
#include "common_kernels.cu"
#include "cuda_utils.cuh"
#include "knn_impute_gpu.cuh"
#include "lincmb_impute_gpu.cuh"
#include "svd_impute_gpu.cuh"
#include <cstdio>


namespace impute {

__global__ void lincmb_fusion_kernel(float *X, const uint8_t *Mask,
                                     const float *X_G, const float *X_L, int N,
                                     int D) {
  int g = blockIdx.x;
  if (g >= D)
    return;
  int tx = threadIdx.x;

  // Find w_g that minimizes error on observed values of this gene g
  // y ~ w*g + (1-w)*l => y - l ~ w*(g - l)
  // e = y - l, d = g - l
  // w = sum(e*d) / sum(d*d)

  double sum_ed = 0;
  double sum_d2 = 0;
  int count = 0;

  for (int i = tx; i < N; i += blockDim.x) {
    if (Mask[i * D + g]) {
      double y = (double)X[i * D + g];
      double g_val = (double)X_G[i * D + g];
      double l_val = (double)X_L[i * D + g];
      double e = y - l_val;
      double d = g_val - l_val;
      sum_ed += e * d;
      sum_d2 += d * d;
      count++;
    }
  }

  sum_ed = blockSumDouble(sum_ed);
  sum_d2 = blockSumDouble(sum_d2);
  count = blockSumInt(count);

  __shared__ double s_w;
  if (tx == 0) {
    double w = 0.5;
    if (count > 0 && sum_d2 > 1e-9) {
      w = sum_ed / sum_d2;
      if (w < 0.0)
        w = 0.0;
      if (w > 1.0)
        w = 1.0;
    }
    s_w = w;
  }
  __syncthreads();

  // Fill missing entries with weighted combination
  for (int i = tx; i < N; i += blockDim.x) {
    if (!Mask[i * D + g]) {
      X[i * D + g] =
          (float)(s_w * X_G[i * D + g] + (1.0 - s_w) * X_L[i * D + g]);
    }
  }
}

void lincmb_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int k_knn,
                        int rank_svd, cudaStream_t stream) {
  if (N <= 0 || D <= 0)
    return;

  // 1. Prepare buffers
  float *d_X_global, *d_X_local;
  cudaMalloc(&d_X_global, (size_t)N * D * sizeof(float));
  cudaMalloc(&d_X_local, (size_t)N * D * sizeof(float));

  cudaMemcpyAsync(d_X_global, d_X, (size_t)N * D * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(d_X_local, d_X, (size_t)N * D * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);

  // 2. Component Imputations
  svd_impute(d_X_global, d_mask, N, D, rank_svd, 10, 1e-4f, stream);
  knn_impute(d_X_local, d_mask, N, D, k_knn, stream);

  // 3. Fusion Kernel
  lincmb_fusion_kernel<<<D, 256, 0, stream>>>(d_X, d_mask, d_X_global,
                                              d_X_local, N, D);

  cudaFree(d_X_global);
  cudaFree(d_X_local);
}

} // namespace impute
