#pragma once
#include "arls_impute_gpu.cuh"
#include "common_kernels.cu"
#include "cuda_utils.cuh"
#include <algorithm>
#include <cstdio>
#include <vector>

namespace impute {

__global__ void compute_arls_means_kernel(const float *X, const uint8_t *Mask,
                                          int N, int D, float *means) {
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

__global__ void compute_arls_cov_kernel(const float *X, const uint8_t *Mask,
                                        const float *means, double *Cov, int N,
                                        int D) {
  int c1 = blockIdx.x;
  int c2 = blockIdx.y * blockDim.x + threadIdx.x;
  if (c1 >= D || c2 >= D || c1 > c2)
    return;

  double dot = 0;
  for (int i = 0; i < N; i++) {
    double v1 = Mask[i * D + c1] ? (double)X[i * D + c1] : (double)means[c1];
    double v2 = Mask[i * D + c2] ? (double)X[i * D + c2] : (double)means[c2];
    dot += (v1 - (double)means[c1]) * (v2 - (double)means[c2]);
  }
  Cov[c1 * D + c2] = dot;
  Cov[c2 * D + c1] = dot;
}

__global__ void arls_accumulate_kernel(const float *X, const uint8_t *Mask,
                                       const float *means, const int *preds,
                                       int num_preds, int g, int N, int D,
                                       double *d_ATA, double *d_ATy,
                                       int *d_n_pair) {
  int S = num_preds + 1;
  extern __shared__ double s_data[];
  double *s_ATA = s_data;
  double *s_ATy = s_data + (S * S);
  __shared__ int s_n_pair;

  int tx = threadIdx.x;
  for (int k = tx; k < S * S; k += blockDim.x)
    s_ATA[k] = 0;
  for (int k = tx; k < S; k += blockDim.x)
    s_ATy[k] = 0;
  if (tx == 0)
    s_n_pair = 0;
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    if (Mask[i * D + g]) {
      atomicAdd(&s_n_pair, 1);
      double row_preds[32];
      for (int k = 0; k < num_preds; k++) {
        int p = preds[k];
        row_preds[k] =
            Mask[i * D + p] ? (double)X[i * D + p] : (double)means[p];
      }
      atomicAdd(&s_ATy[0], (double)X[i * D + g]);
      atomicAdd(&s_ATA[0], 1.0);
      for (int k = 0; k < num_preds; k++) {
        atomicAdd(&s_ATA[k + 1], row_preds[k]);
        atomicAdd(&s_ATA[(k + 1) * S], row_preds[k]);
        for (int j = 0; j < num_preds; j++) {
          atomicAdd(&s_ATA[(k + 1) * S + (j + 1)], row_preds[k] * row_preds[j]);
        }
        atomicAdd(&s_ATy[k + 1], (double)X[i * D + g] * row_preds[k]);
      }
    }
  }
  __syncthreads();
  for (int k = tx; k < S * S; k += blockDim.x)
    atomicAdd(&d_ATA[k], s_ATA[k]);
  for (int k = tx; k < S; k += blockDim.x)
    atomicAdd(&d_ATy[k], s_ATy[k]);
  if (tx == 0)
    atomicAdd(d_n_pair, s_n_pair);
}

__global__ void arls_impute_kernel(float *X, const uint8_t *Mask,
                                   const float *means, const int *preds,
                                   int num_preds, const double *beta, int g,
                                   int N, int D) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  if (!Mask[i * D + g]) {
    double pred_val = beta[0];
    for (int k = 0; k < num_preds; k++) {
      int p = preds[k];
      double val = Mask[i * D + p] ? (double)X[i * D + p] : (double)means[p];
      pred_val += beta[k + 1] * val;
    }
    X[i * D + g] = (float)pred_val;
  }
}

void arls_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                      float ridge, cudaStream_t stream) {
  if (N <= 0 || D <= 0)
    return;
  if (K <= 0)
    K = 10;

  float *d_means;
  cudaMalloc(&d_means, D * sizeof(float));
  compute_arls_means_kernel<<<(D + 255) / 256, 256, 0, stream>>>(d_X, d_mask, N,
                                                                 D, d_means);

  double *d_Cov;
  cudaMalloc(&d_Cov, (size_t)D * D * sizeof(double));
  dim3 cov_blocks(D, (D + 31) / 32);
  compute_arls_cov_kernel<<<cov_blocks, 32, 0, stream>>>(d_X, d_mask, d_means,
                                                         d_Cov, N, D);

  std::vector<double> h_Cov(D * D);
  cudaMemcpyAsync(h_Cov.data(), d_Cov, D * D * sizeof(double),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  double *d_ATA, *d_ATy, *d_beta;
  int *d_n_pair, *d_preds;
  cudaMalloc(&d_ATA, (K + 1) * (K + 1) * sizeof(double));
  cudaMalloc(&d_ATy, (K + 1) * sizeof(double));
  cudaMalloc(&d_beta, (K + 1) * sizeof(double));
  cudaMalloc(&d_n_pair, sizeof(int));
  cudaMalloc(&d_preds, K * sizeof(int));

  std::vector<float> h_means(D);
  cudaMemcpy(h_means.data(), d_means, D * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int g = 0; g < D; g++) {
    std::vector<std::pair<float, int>> cand;
    for (int c = 0; c < D; c++) {
      if (g == c)
        continue;
      float r2 = (float)((h_Cov[g * D + c] * h_Cov[g * D + c]) /
                         (h_Cov[g * D + g] * h_Cov[c * D + c] + 1e-9));
      cand.push_back({r2, c});
    }
    std::sort(cand.rbegin(), cand.rend());
    int num_preds = std::min(K, (int)cand.size());
    std::vector<int> h_preds(num_preds);
    for (int k = 0; k < num_preds; k++)
      h_preds[k] = cand[k].second;

    cudaMemcpyAsync(d_preds, h_preds.data(), num_preds * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_ATA, 0, (K + 1) * (K + 1) * sizeof(double), stream);
    cudaMemsetAsync(d_ATy, 0, (K + 1) * sizeof(double), stream);
    cudaMemsetAsync(d_n_pair, 0, sizeof(int), stream);

    int S = num_preds + 1;
    arls_accumulate_kernel<<<(N + 255) / 256, 256, (S * S + S) * sizeof(double),
                             stream>>>(d_X, d_mask, d_means, d_preds, num_preds,
                                       g, N, D, d_ATA, d_ATy, d_n_pair);

    int h_n_pair;
    cudaMemcpyAsync(&h_n_pair, d_n_pair, sizeof(int), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);

    std::vector<double> h_beta(S, 0.0);
    if (h_n_pair < 5) {
      h_beta[0] = h_means[g];
      cudaMemcpyAsync(d_beta, h_beta.data(), S * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
      arls_impute_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
          d_X, d_mask, d_means, d_preds, 0, d_beta, g, N, D);
      continue;
    }

    std::vector<double> h_ATA(S * S), h_ATy(S);
    cudaMemcpy(h_ATA.data(), d_ATA, S * S * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ATy.data(), d_ATy, S * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < S; k++)
      h_ATA[k * S + k] += (double)ridge * h_n_pair;

    bool solved = true;
    for (int i = 0; i < S; i++) {
      int pivot = i;
      for (int j = i + 1; j < S; j++)
        if (std::abs(h_ATA[j * S + i]) > std::abs(h_ATA[pivot * S + i]))
          pivot = j;
      for (int k = 0; k < S; k++)
        std::swap(h_ATA[i * S + k], h_ATA[pivot * S + k]);
      std::swap(h_ATy[i], h_ATy[pivot]);
      if (std::abs(h_ATA[i * S + i]) < 1e-12) {
        solved = false;
        break;
      }
      for (int j = i + 1; j < S; j++) {
        double f = h_ATA[j * S + i] / h_ATA[i * S + i];
        h_ATy[j] -= f * h_ATy[i];
        for (int k = i; k < S; k++)
          h_ATA[j * S + k] -= f * h_ATA[i * S + k];
      }
    }
    if (solved) {
      for (int i = S - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < S; j++)
          sum += h_ATA[i * S + j] * h_beta[j];
        h_beta[i] = (h_ATy[i] - sum) / h_ATA[i * S + i];
      }
    } else {
      h_beta.assign(S, 0.0);
      h_beta[0] = h_means[g];
    }

    cudaMemcpyAsync(d_beta, h_beta.data(), S * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    arls_impute_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
        d_X, d_mask, d_means, d_preds, num_preds, d_beta, g, N, D);
  }
  cudaFree(d_means);
  cudaFree(d_Cov);
  cudaFree(d_ATA);
  cudaFree(d_ATy);
  cudaFree(d_beta);
  cudaFree(d_n_pair);
  cudaFree(d_preds);
}

} // namespace impute
