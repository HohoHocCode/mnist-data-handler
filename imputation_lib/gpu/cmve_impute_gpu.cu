#pragma once
#include "cmve_impute_gpu.cuh"
#include "common_kernels.cu"
#include "cuda_utils.cuh"
#include <cstdio>

namespace impute {

__global__ void cmve_optimized_kernel(float *X, const uint8_t *Mask,
                                      const float *means, const float *Cov,
                                      int N, int D, int K_req) {
  int g = blockIdx.x;
  if (g >= D)
    return;

  __shared__ struct Cand {
    float r2;
    int c;
  } s_cand[128];
  int tx = threadIdx.x;
  int K = K_req;
  if (K > 128)
    K = 128;

  for (int k = tx; k < 128; k += blockDim.x)
    s_cand[k] = {-1.0f, -1};
  __syncthreads();

  if (tx == 0) {
    for (int c = 0; c < D; c++) {
      if (c == g)
        continue;
      float dot_g = Cov[g * D + g];
      float dot_c = Cov[c * D + c];
      float cov = Cov[g * D + c];
      float r2 = (cov * cov) / (dot_g * dot_c + 1e-9f);

      if (r2 > s_cand[K - 1].r2) {
        s_cand[K - 1] = {r2, c};
        for (int k = K - 2; k >= 0; k--) {
          if (s_cand[k + 1].r2 > s_cand[k].r2) {
            Cand tmp = s_cand[k];
            s_cand[k] = s_cand[k + 1];
            s_cand[k + 1] = tmp;
          } else
            break;
        }
      }
    }
  }
  __syncthreads();

  struct Model {
    float alpha, beta, weight;
    int c;
  };
  __shared__ Model s_models[128];

  // Parallel model fitting
  for (int k = 0; k < K; k++) {
    int c = s_cand[k].c;
    if (c == -1) {
      if (tx == 0)
        s_models[k] = {0, 0, 0, -1};
      continue;
    }

    double sx = 0, sy = 0, sxy = 0, sxx = 0;
    int n_pair = 0;
    for (int i = tx; i < N; i += blockDim.x) {
      if (Mask[i * D + g] && Mask[i * D + c]) {
        float y = X[i * D + g], x = X[i * D + c];
        sx += x;
        sy += y;
        sxy += (double)x * y;
        sxx += (double)x * x;
        n_pair++;
      }
    }
    sx = blockSumDouble(sx);
    sy = blockSumDouble(sy);
    sxy = blockSumDouble(sxy);
    sxx = blockSumDouble(sxx);
    n_pair = blockSumInt(n_pair);

    if (tx == 0) {
      if (n_pair >= 5) {
        double denom = n_pair * sxx - sx * sx;
        if (abs(denom) > 1e-9) {
          double beta = (n_pair * sxy - sx * sy) / denom;
          double alpha = (sy - beta * sx) / (double)n_pair;
          double sse = 0;
          for (int i = 0; i < N; i++) {
            if (Mask[i * D + g] && Mask[i * D + c]) {
              double err =
                  (double)X[i * D + g] - (alpha + beta * (double)X[i * D + c]);
              sse += err * err;
            }
          }
          double mse = sse / (n_pair - 2);
          if (mse < 1e-6)
            mse = 1e-6;
          s_models[k] = {(float)alpha, (float)beta, (float)(1.0 / mse), c};
        } else
          s_models[k] = {0, 0, 0, -1};
      } else
        s_models[k] = {0, 0, 0, -1};
    }
    __syncthreads();
  }

  // Final Impute
  for (int i = tx; i < N; i += blockDim.x) {
    if (!Mask[i * D + g]) {
      double num = 0, den = 0;
      for (int k = 0; k < K; k++) {
        const auto &m = s_models[k];
        if (m.c != -1 && Mask[i * D + m.c]) {
          num += (double)(m.alpha + m.beta * (double)X[i * D + m.c]) * m.weight;
          den += (double)m.weight;
        }
      }
      if (den > 0)
        X[i * D + g] = (float)(num / den);
      else
        X[i * D + g] = means[g];
    }
  }
}

__global__ void cmve_means_kernel(const float *X, const uint8_t *Mask, int N,
                                  int D, float *means) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= D)
    return;
  double s = 0;
  int cnt = 0;
  for (int i = 0; i < N; i++)
    if (Mask[i * D + j]) {
      s += X[i * D + j];
      cnt++;
    }
  means[j] = (cnt > 0) ? (float)(s / cnt) : 0.0f;
}

__global__ void cmve_center_kernel(float *X, const float *means, int N, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * D)
    X[idx] -= means[idx % D];
}

__global__ void cmve_cov_kernel(const float *X, float *Cov, int N, int D) {
  int c1 = blockIdx.x;
  int c2 = blockIdx.y * blockDim.x + threadIdx.x;
  if (c1 >= D || c2 >= D || c1 > c2)
    return;
  double dot = 0;
  for (int i = 0; i < N; i++)
    dot += (double)X[i * D + c1] * X[i * D + c2];
  Cov[c1 * D + c2] = (float)dot;
  Cov[c2 * D + c1] = (float)dot;
}

void cmve_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                      cudaStream_t stream) {
  if (N <= 0 || D <= 0)
    return;
  if (K <= 0)
    K = 10;

  float *d_means;
  cudaMalloc(&d_means, D * sizeof(float));
  cmve_means_kernel<<<(D + 255) / 256, 256, 0, stream>>>(d_X, d_mask, N, D,
                                                         d_means);

  float *d_X_center;
  cudaMalloc(&d_X_center, (size_t)N * D * sizeof(float));
  cudaMemcpyAsync(d_X_center, d_X, (size_t)N * D * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  mean_fill_kernel<<<(D + 255) / 256, 256, 0, stream>>>(d_X_center, d_mask, N,
                                                        D);
  cmve_center_kernel<<<(N * D + 255) / 256, 256, 0, stream>>>(d_X_center,
                                                              d_means, N, D);

  float *d_Cov;
  cudaMalloc(&d_Cov, (size_t)D * D * sizeof(float));
  dim3 cov_blocks(D, (D + 31) / 32);
  cmve_cov_kernel<<<cov_blocks, 32, 0, stream>>>(d_X_center, d_Cov, N, D);

  mean_fill_kernel<<<(D + 255) / 256, 256, 0, stream>>>(d_X, d_mask, N, D);
  cmve_optimized_kernel<<<D, 128, 0, stream>>>(d_X, d_mask, d_means, d_Cov, N,
                                               D, K);

  cudaFree(d_means);
  cudaFree(d_X_center);
  cudaFree(d_Cov);
}

} // namespace impute
