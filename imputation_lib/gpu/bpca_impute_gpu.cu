#pragma once
#include "common_kernels.cu"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cub/cub.cuh>
#include <vector>


namespace impute {

__global__ void bpca_estep_warp_kernel(const float *X, const uint8_t *Mask,
                                       const float *W, const float *Mu,
                                       float Tau, float *Ex, float *Exx, int N,
                                       int D, int K) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= N)
    return;

  int tx = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;
  unsigned int active = __activemask();

  extern __shared__ double s_M_doubles[];
  double *s_M = &s_M_doubles[warp_id * 1024];

  // Initialize s_M to Identity
  for (int i = tx; i < 1024; i += 32)
    s_M[i] = 0.0;
  if (tx < K)
    s_M[tx * K + tx] = 1.0;
  __syncwarp(active);

  double RHS_local = 0.0;
  for (int j = 0; j < D; ++j) {
    if (Mask[row * D + j]) {
      float y_val = X[row * D + j] - Mu[j];
      float w_tx = (tx < K) ? W[j * K + tx] : 0.0f;
      for (int k1 = 0; k1 < K; ++k1) {
        float w_k1 = __shfl_sync(0xFFFFFFFF, w_tx, k1);
        if (tx < K) {
          s_M[k1 * K + tx] += (double)Tau * w_k1 * w_tx;
        }
      }
      if (tx < K)
        RHS_local += (double)w_tx * y_val;
    }
  }
  __syncwarp(active);

  __shared__ double s_RHS_shared[4 * 32];
  if (tx < K)
    s_RHS_shared[warp_id * 32 + tx] = RHS_local;
  __syncwarp(active);

  if (tx == 0) {
    double b_d[32], ex_d[32];
    for (int k = 0; k < K; k++)
      b_d[k] = Tau * s_RHS_shared[warp_id * 32 + k];

    // Cholesky s_M
    for (int i = 0; i < K; i++) {
      for (int j = 0; j <= i; j++) {
        double s = 0;
        for (int k = 0; k < j; k++)
          s += s_M[i * K + k] * s_M[j * K + k];
        if (i == j)
          s_M[i * K + j] = sqrt(max(s_M[i * K + j] - s, 1e-12));
        else
          s_M[i * K + j] = (s_M[i * K + j] - s) / s_M[j * K + j];
      }
    }
    // Solve Ly = b
    for (int i = 0; i < K; i++) {
      double s = 0;
      for (int j = 0; j < i; j++)
        s += s_M[i * K + j] * ex_d[j];
      ex_d[i] = (b_d[i] - s) / s_M[i * K + i];
    }
    // Solve L'x = y
    for (int i = K - 1; i >= 0; i--) {
      double s = 0;
      for (int j = i + 1; j < K; j++)
        s += s_M[j * K + i] * ex_d[j];
      ex_d[i] = (ex_d[i] - s) / s_M[i * K + i];
    }
    for (int k = 0; k < K; k++)
      Ex[row * K + k] = (float)ex_d[k];

    // Exx contribution
    for (int col = 0; col < K; col++) {
      double res[32] = {0};
      res[col] = 1.0;
      // Solve Lz = e_col
      for (int i = 0; i < K; i++) {
        double s = 0;
        for (int j = 0; j < i; j++)
          s += s_M[i * K + j] * res[j];
        res[i] = (res[i] - s) / s_M[i * K + i];
      }
      // Solve L'z_inv = z
      for (int i = K - 1; i >= 0; i--) {
        double s = 0;
        for (int j = i + 1; j < K; j++)
          s += s_M[j * K + i] * res[j];
        res[i] = (res[i] - s) / s_M[i * K + i];
      }
      for (int r = 0; r < K; r++) {
        Exx[row * K * K + r * K + col] = (float)(res[r] + ex_d[r] * ex_d[col]);
      }
    }
  }
}

__global__ void bpca_accum_stats_kernel(const float *Ex, const float *Exx,
                                        const float *X, const uint8_t *Mask,
                                        const float *Mu, double *SumEx,
                                        double *SumExx, double *RHS,
                                        double *SumErr, int N, int D, int K) {
  int j = blockIdx.x;
  int tx = threadIdx.x;
  if (j < D) {
    double local_err = 0;
    double local_rhs[32] = {0};
    for (int i = tx; i < N; i += blockDim.x) {
      float dy = X[i * D + j] - Mu[j];
      local_err += (double)dy * dy;
      for (int k = 0; k < K; k++)
        local_rhs[k] += (double)dy * Ex[i * K + k];
    }
    for (int k = 0; k < K; k++) {
      double total_k = blockSumDouble(local_rhs[k]);
      if (tx == 0)
        atomicAdd(&RHS[j * K + k], total_k);
    }
    double local_sum_err = blockSumDouble(local_err);
    if (tx == 0)
      atomicAdd(SumErr, local_sum_err);
  }
  if (j == D) {
    for (int k = tx; k < K; k += blockDim.x) {
      double s = 0;
      for (int i = 0; i < N; i++)
        s += (double)Ex[i * K + k];
      SumEx[k] = s;
    }
  }
  if (j == D + 1) {
    for (int k = tx; k < K * K; k += blockDim.x) {
      double s = 0;
      for (int i = 0; i < N; i++)
        s += (double)Exx[i * K * K + k];
      SumExx[k] = s;
    }
  }
}

__global__ void bpca_mu_kernel(const float *X, const uint8_t *Mask, int N,
                               int D, float *Mu) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= D)
    return;
  double s = 0;
  int c = 0;
  for (int i = 0; i < N; i++)
    if (Mask[i * D + j]) {
      s += X[i * D + j];
      c++;
    }
  Mu[j] = (c > 0) ? (float)(s / c) : 0.0f;
}

__global__ void bpca_impute_fill_kernel(float *X, const uint8_t *Mask,
                                        const float *W, const float *Mu,
                                        const float *Ex, int N, int D, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * D) {
    if (!Mask[idx]) {
      int i = idx / D, j = idx % D;
      double val = Mu[j];
      for (int k = 0; k < K; k++)
        val += (double)W[j * K + k] * Ex[i * K + k];
      X[idx] = (float)val;
    }
  }
}

void bpca_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K_req,
                      int max_iter, cudaStream_t stream) {
  int K = (K_req > D - 1) ? D - 1 : K_req;
  if (K > 32)
    K = 32;

  float *d_W, *d_Mu, *d_Alpha, *d_Ex, *d_Exx;
  CUDA_CHECK(cudaMalloc(&d_W, D * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Mu, D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Alpha, K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Ex, N * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Exx, N * K * K * sizeof(float)));

  double *d_SumEx, *d_SumExx, *d_RHS, *d_SumErr;
  CUDA_CHECK(cudaMalloc(&d_SumEx, K * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_SumExx, K * K * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_RHS, D * K * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_SumErr, sizeof(double)));

  mean_fill_kernel<<<(D + 255) / 256, 256, 0, stream>>>(d_X, d_mask, N, D);
  bpca_mu_kernel<<<(D + 255) / 256, 256, 0, stream>>>(d_X, d_mask, N, D, d_Mu);

  std::vector<float> h_W(D * K);
  for (int i = 0; i < D * K; ++i)
    h_W[i] = (float)((i % 100) * 0.01f);
  cudaMemcpyAsync(d_W, h_W.data(), D * K * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  std::vector<float> h_Alpha(K, 1.0f);
  cudaMemcpyAsync(d_Alpha, h_Alpha.data(), K * sizeof(float),
                  cudaMemcpyHostToDevice, stream);

  float h_Tau = 1.0f;

  for (int it = 0; it < max_iter; it++) {
    size_t shmem_size = 4 * 1024 * sizeof(double);
    bpca_estep_warp_kernel<<<(N + 3) / 4, 128, shmem_size, stream>>>(
        d_X, d_mask, d_W, d_Mu, h_Tau, d_Ex, d_Exx, N, D, K);

    cudaMemsetAsync(d_SumEx, 0, K * sizeof(double), stream);
    cudaMemsetAsync(d_SumExx, 0, K * K * sizeof(double), stream);
    cudaMemsetAsync(d_RHS, 0, D * K * sizeof(double), stream);
    cudaMemsetAsync(d_SumErr, 0, sizeof(double), stream);

    bpca_accum_stats_kernel<<<D + 2, 256, 0, stream>>>(
        d_Ex, d_Exx, d_X, d_mask, d_Mu, d_SumEx, d_SumExx, d_RHS, d_SumErr, N,
        D, K);

    std::vector<double> h_SumExx(K * K), h_RHS(D * K);
    double h_SumErr_val;
    cudaMemcpyAsync(h_SumExx.data(), d_SumExx, K * K * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_RHS.data(), d_RHS, D * K * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_SumErr_val, d_SumErr, sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> W_new(D * K);
    std::vector<double> LHS(K * K);
    for (int idx = 0; idx < K * K; idx++)
      LHS[idx] = h_SumExx[idx];
    for (int k = 0; k < K; k++)
      LHS[k * K + k] += (double)h_Alpha[k] / (h_Tau + 1e-12);

    std::vector<double> LHS_inv(K * K, 0.0);
    for (int k = 0; k < K; k++)
      LHS_inv[k * K + k] = 1.0;
    for (int p = 0; p < K; p++) {
      double pivot = LHS[p * K + p];
      if (std::abs(pivot) < 1e-12)
        pivot = (pivot >= 0) ? 1e-12 : -1e-12;
      for (int j = 0; j < K; j++) {
        LHS[p * K + j] /= pivot;
        LHS_inv[p * K + j] /= pivot;
      }
      for (int r = 0; r < K; r++)
        if (r != p) {
          double f = LHS[r * K + p];
          for (int j = 0; j < K; j++) {
            LHS[r * K + j] -= f * LHS[p * K + j];
            LHS_inv[r * K + j] -= f * LHS_inv[p * K + j];
          }
        }
    }
    for (int j = 0; j < D; j++) {
      for (int k = 0; k < K; k++) {
        double s = 0;
        for (int m = 0; m < K; m++)
          s += h_RHS[j * K + m] * LHS_inv[m * K + k];
        W_new[j * K + k] = (float)s;
      }
    }
    cudaMemcpyAsync(d_W, W_new.data(), D * K * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    for (int k = 0; k < K; k++) {
      float w_k_norm = 1e-8f;
      for (int j = 0; j < D; j++)
        w_k_norm += W_new[j * K + k] * W_new[j * K + k];
      h_Alpha[k] = (float)D / w_k_norm;
    }
    cudaMemcpyAsync(d_Alpha, h_Alpha.data(), K * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    double term2 = 0;
    for (int idx = 0; idx < D * K; idx++)
      term2 += h_RHS[idx] * (double)W_new[idx];
    double term3 = 0;
    for (int k1 = 0; k1 < K; k1++) {
      for (int k2 = 0; k2 < K; k2++) {
        double WtW = 0;
        for (int j = 0; j < D; j++)
          WtW += (double)W_new[j * K + k1] * (double)W_new[j * K + k2];
        term3 += WtW * h_SumExx[k2 * K + k1];
      }
    }
    double err = h_SumErr_val - 2.0 * term2 + term3;
    float var = (float)(err / (N * D));
    h_Tau = 1.0f / (var > 1e-5f ? var : 1e-5f);
    if (h_Tau > 1000.0f)
      h_Tau = 1000.0f;
  }

  bpca_impute_fill_kernel<<<(N * D + 255) / 256, 256, 0, stream>>>(
      d_X, d_mask, d_W, d_Mu, d_Ex, N, D, K);

  cudaFree(d_W);
  cudaFree(d_Mu);
  cudaFree(d_Alpha);
  cudaFree(d_Ex);
  cudaFree(d_Exx);
  cudaFree(d_SumEx);
  cudaFree(d_SumExx);
  cudaFree(d_RHS);
  cudaFree(d_SumErr);
}

} // namespace impute
