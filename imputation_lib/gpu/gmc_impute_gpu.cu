#pragma once
#include "common_kernels.cu"
#include "cuda_utils.cuh"
#include "gmc_impute_gpu.cuh"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

namespace impute {

__global__ void gmc_estep_kernel(const float *X, const uint8_t *Mask,
                                 const double *mu, const double *var,
                                 const double *pi, double *gamma, int N, int D,
                                 int K) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  double log_probs[32]; // Max K=32
  double max_log = -1e30;

  for (int k = 0; k < K; k++) {
    double lp = log(pi[k] + 1e-9);
    for (int c = 0; c < D; c++) {
      if (Mask[i * D + c]) {
        double diff = (double)X[i * D + c] - mu[k * D + c];
        lp += -0.5 * (log(2.0 * 3.141592653589793 * var[k * D + c]) +
                      (diff * diff) / var[k * D + c]);
      }
    }
    log_probs[k] = lp;
    if (lp > max_log)
      max_log = lp;
  }

  double sum_exp = 0;
  for (int k = 0; k < K; k++)
    sum_exp += exp(log_probs[k] - max_log);
  double lse = max_log + log(sum_exp);

  for (int k = 0; k < K; k++)
    gamma[i * K + k] = exp(log_probs[k] - lse);
}

__global__ void gmc_mstep_kernel(const float *X, const uint8_t *Mask,
                                 const double *gamma, double *mu, double *var,
                                 double *pi, int N, int D, int K) {
  int k = blockIdx.x;
  if (k >= K)
    return;
  int tx = threadIdx.x;

  // 1. Update pi[k]
  double sum_gamma = 0;
  for (int i = tx; i < N; i += blockDim.x)
    sum_gamma += gamma[i * K + k];
  sum_gamma = blockSumDouble(sum_gamma);
  if (tx == 0)
    pi[k] = sum_gamma / N;
  __syncthreads();

  // 2. Update mu[k] and var[k] per feature c
  for (int c = tx; c < D; c += blockDim.x) {
    double w_sum_x = 0;
    double w_sum_r = 0;
    for (int i = 0; i < N; i++) {
      if (Mask[i * D + c]) {
        w_sum_x += gamma[i * K + k] * (double)X[i * D + c];
        w_sum_r += gamma[i * K + k];
      }
    }
    if (w_sum_r > 1e-6)
      mu[k * D + c] = w_sum_x / w_sum_r;

    double w_sum_diff = 0;
    for (int i = 0; i < N; i++) {
      if (Mask[i * D + c]) {
        double d = (double)X[i * D + c] - mu[k * D + c];
        w_sum_diff += gamma[i * K + k] * (d * d);
      }
    }
    if (w_sum_r > 1e-6)
      var[k * D + c] = w_sum_diff / w_sum_r;
    if (var[k * D + c] < 1e-4)
      var[k * D + c] = 1e-4;
  }
}

__global__ void gmc_impute_kernel(float *X, const uint8_t *Mask,
                                  const double *gamma, const double *mu, int N,
                                  int D, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * D)
    return;
  int i = idx / D;
  int c = idx % D;

  if (!Mask[idx]) {
    double val = 0;
    for (int k = 0; k < K; k++)
      val += gamma[i * K + k] * mu[k * D + c];
    X[idx] = (float)val;
  }
}

void gmc_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                     int max_iter, cudaStream_t stream) {
  if (N <= 0 || D <= 0)
    return;
  if (K <= 0)
    K = 5;

  // 1. Deterministic Initialization on Host
  std::vector<double> h_mu(K * D), h_var(K * D, 1.0), h_pi(K, 1.0 / K);
  std::mt19937 rng(42);

  // col_means for initialization
  std::vector<float> h_X(N * D);
  std::vector<uint8_t> h_M(N * D);
  cudaMemcpy(h_X.data(), d_X, N * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_M.data(), d_mask, N * D * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);

  std::vector<double> col_means(D, 0.0);
  for (int c = 0; c < D; c++) {
    double s = 0;
    int cnt = 0;
    for (int i = 0; i < N; i++)
      if (h_M[i * D + c]) {
        s += h_X[i * D + c];
        cnt++;
      }
    if (cnt > 0)
      col_means[c] = s / cnt;
  }

  std::uniform_int_distribution<int> n_dist(0, N - 1);
  for (int k = 0; k < K; k++) {
    int idx = n_dist(rng);
    for (int c = 0; c < D; c++) {
      if (h_M[idx * D + c])
        h_mu[k * D + c] = h_X[idx * D + c];
      else
        h_mu[k * D + c] = col_means[c] + ((double)n_dist(rng) / N - 0.5) * 0.1;
    }
  }

  // 2. Transfer to Device
  double *d_mu, *d_var, *d_pi, *d_gamma;
  cudaMalloc(&d_mu, K * D * sizeof(double));
  cudaMalloc(&d_var, K * D * sizeof(double));
  cudaMalloc(&d_pi, K * sizeof(double));
  cudaMalloc(&d_gamma, N * K * sizeof(double));

  cudaMemcpy(d_mu, h_mu.data(), K * D * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_var, h_var.data(), K * D * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_pi, h_pi.data(), K * sizeof(double), cudaMemcpyHostToDevice);

  // 3. EM Loop on GPU
  for (int iter = 0; iter < max_iter; iter++) {
    gmc_estep_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
        d_X, d_mask, d_mu, d_var, d_pi, d_gamma, N, D, K);
    gmc_mstep_kernel<<<K, 256, 0, stream>>>(d_X, d_mask, d_gamma, d_mu, d_var,
                                            d_pi, N, D, K);
  }

  // 4. Final Impute
  gmc_impute_kernel<<<(N * D + 255) / 256, 256, 0, stream>>>(
      d_X, d_mask, d_gamma, d_mu, N, D, K);

  cudaFree(d_mu);
  cudaFree(d_var);
  cudaFree(d_pi);
  cudaFree(d_gamma);
}

} // namespace impute
