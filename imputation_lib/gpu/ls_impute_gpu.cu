#pragma once
#include "common_kernels.cu"
#include "cuda_utils.cuh"
#include "ls_impute_gpu.cuh"
#include <cstdio>

namespace impute_ls {

// Pearson Correlation Kernel (Row vs Row)
__global__ void ls_correlation_kernel(const float *X, const uint8_t *M,
                                      float *Corrs, int N, int D) {
  int r1 = blockIdx.x;
  int r2 = blockIdx.y * blockDim.x + threadIdx.x;
  if (r1 >= N || r2 >= N || r1 >= r2)
    return;

  double s1 = 0, s2 = 0, s11 = 0, s22 = 0, s12 = 0;
  int cnt = 0;
  for (int j = 0; j < D; j++) {
    if (M[r1 * D + j] && M[r2 * D + j]) {
      double v1 = X[r1 * D + j];
      double v2 = X[r2 * D + j];
      s1 += v1;
      s2 += v2;
      s11 += v1 * v1;
      s22 += v2 * v2;
      s12 += v1 * v2;
      cnt++;
    }
  }

  float r = 0;
  if (cnt > 2) {
    double num = (double)cnt * s12 - s1 * s2;
    double den1 = (double)cnt * s11 - s1 * s1;
    double den2 = (double)cnt * s22 - s2 * s2;
    if (den1 > 1e-9 && den2 > 1e-9)
      r = (float)fabs(num / sqrt(den1 * den2));
  }
  Corrs[r1 * N + r2] = r;
  Corrs[r2 * N + r1] = r;
}

// Rigid Neighbor Selection Kernel
__global__ void ls_select_neighbors_kernel(const uint8_t *M, const float *Corrs,
                                           int *Neighbors, int N, int D,
                                           int K) {
  int i = blockIdx.x;
  if (i >= N)
    return;

  // Identify missing columns of target row i
  int miss_cols[128]; // Max 128 missing cols for optimization
  int miss_cnt = 0;
  for (int j = 0; j < D && miss_cnt < 128; j++) {
    if (!M[i * D + j])
      miss_cols[miss_cnt++] = j;
  }
  if (miss_cnt == 0)
    return;

  // Each thread finds top K among a subset of rows?
  // Simplified: One thread per row i finds its own top K.
  // (Could be optimized with sorting, but N=100-2000 is small enough for simple
  // top-K insertion)
  struct Cand {
    int id;
    float corr;
  };
  Cand top[128];
  for (int k = 0; k < K; k++)
    top[k] = {-1, -1.0f};

  for (int r = 0; r < N; r++) {
    if (r == i)
      continue;

    // Rigid: Neighbor must observe all missing columns of i
    bool ok = true;
    for (int k = 0; k < miss_cnt; k++) {
      if (!M[r * D + miss_cols[k]]) {
        ok = false;
        break;
      }
    }
    if (!ok)
      continue;

    float c = Corrs[i * N + r];
    if (c > top[K - 1].corr) {
      top[K - 1] = {r, c};
      for (int k = K - 2; k >= 0; k--) {
        if (top[k + 1].corr > top[k].corr) {
          Cand tmp = top[k];
          top[k] = top[k + 1];
          top[k + 1] = tmp;
        } else
          break;
      }
    }
  }

  for (int k = 0; k < K; k++)
    Neighbors[i * K + k] = top[k].id;
}

// Row-wise Regression Solver
__global__ void ls_solve_kernel(float *X_out, const float *X_orig,
                                const uint8_t *Mask, const int *Neighbors,
                                int N, int D, int K) {
  int i = blockIdx.x;
  if (i >= N)
    return;

  int n_neigh = 0;
  int neighbors[128];
  for (int k = 0; k < K; k++) {
    if (Neighbors[i * K + k] != -1)
      neighbors[n_neigh++] = Neighbors[i * K + k];
  }
  if (n_neigh == 0)
    return;

  // Find missing cols
  int miss_cols[256];
  int miss_cnt = 0;
  for (int j = 0; j < D && miss_cnt < 256; j++) {
    if (!Mask[i * D + j])
      miss_cols[miss_cnt++] = j;
  }
  if (miss_cnt == 0)
    return;

  // Normal Equations: AtA * w = Atb
  double AtA[128 * 128]; // Max K=128
  double Atb[128];
  for (int k = 0; k < n_neigh; k++) {
    Atb[k] = 0;
    for (int k2 = 0; k2 < n_neigh; k2++)
      AtA[k * n_neigh + k2] = 0;
  }

  int n_samples = 0;
  for (int j = 0; j < D; j++) {
    if (Mask[i * D + j]) {
      bool all_obs = true;
      for (int k = 0; k < n_neigh; k++)
        if (!Mask[neighbors[k] * D + j]) {
          all_obs = false;
          break;
        }
      if (!all_obs)
        continue;

      n_samples++;
      double y = (double)X_orig[i * D + j];
      for (int k1 = 0; k1 < n_neigh; k1++) {
        double v1 = (double)X_orig[neighbors[k1] * D + j];
        Atb[k1] += v1 * y;
        for (int k2 = 0; k2 < n_neigh; k2++) {
          double v2 = (double)X_orig[neighbors[k2] * D + j];
          AtA[k1 * n_neigh + k2] += v1 * v2;
        }
      }
    }
  }

  if (n_samples < n_neigh) {
    // Fallback to mean imputation using neighbors
    for (int m = 0; m < miss_cnt; m++) {
      double sum = 0;
      for (int k = 0; k < n_neigh; k++)
        sum += (double)X_orig[neighbors[k] * D + miss_cols[m]];
      X_out[i * D + miss_cols[m]] = (float)(sum / n_neigh);
    }
  } else {
    // Add Ridge
    for (int k = 0; k < n_neigh; k++)
      AtA[k * n_neigh + k] += 1e-5;

    // Solve via Gaussian Elimination
    double w[128];
    bool success = true;
    for (int k = 0; k < n_neigh; k++) {
      int pivot = k;
      for (int r = k + 1; r < n_neigh; r++)
        if (abs(AtA[r * n_neigh + k]) > abs(AtA[pivot * n_neigh + k]))
          pivot = r;

      for (int c = 0; c < n_neigh; c++) {
        double tmp = AtA[k * n_neigh + c];
        AtA[k * n_neigh + c] = AtA[pivot * n_neigh + c];
        AtA[pivot * n_neigh + c] = tmp;
      }
      double tmpb = Atb[k];
      Atb[k] = Atb[pivot];
      Atb[pivot] = tmpb;

      if (abs(AtA[k * n_neigh + k]) < 1e-12) {
        success = false;
        break;
      }

      for (int r = k + 1; r < n_neigh; r++) {
        double f = AtA[r * n_neigh + k] / AtA[k * n_neigh + k];
        Atb[r] -= f * Atb[k];
        for (int c = k; c < n_neigh; c++)
          AtA[r * n_neigh + c] -= f * AtA[k * n_neigh + c];
      }
    }

    if (success) {
      for (int r = n_neigh - 1; r >= 0; r--) {
        double s = 0;
        for (int c = r + 1; c < n_neigh; c++)
          s += AtA[r * n_neigh + c] * w[c];
        w[r] = (Atb[r] - s) / AtA[r * n_neigh + r];
      }
      for (int m = 0; m < miss_cnt; m++) {
        double pred = 0;
        for (int k = 0; k < n_neigh; k++)
          pred += w[k] * (double)X_orig[neighbors[k] * D + miss_cols[m]];
        X_out[i * D + miss_cols[m]] = (float)pred;
      }
    }
  }
}

void ls_impute(float *d_X, uint8_t *d_mask, int N, int D, int K,
               cudaStream_t stream) {
  if (N <= 0 || D <= 0)
    return;
  if (K <= 0)
    K = 10;

  // 1. Correlation Matrix
  float *d_Corrs;
  cudaMalloc(&d_Corrs, (size_t)N * N * sizeof(float));
  dim3 threads(32);
  dim3 blocks(N, (N + 31) / 32);
  ls_correlation_kernel<<<blocks, threads, 0, stream>>>(d_X, d_mask, d_Corrs, N,
                                                        D);

  // 2. Neighbor Selection
  int *d_Neighbors;
  cudaMalloc(&d_Neighbors, (size_t)N * K * sizeof(int));
  ls_select_neighbors_kernel<<<N, 1, 0, stream>>>(d_mask, d_Corrs, d_Neighbors,
                                                  N, D, K);

  // 3. Solve
  float *d_X_orig;
  cudaMalloc(&d_X_orig, (size_t)N * D * sizeof(float));
  cudaMemcpyAsync(d_X_orig, d_X, (size_t)N * D * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);

  ls_solve_kernel<<<N, 1, 0, stream>>>(d_X, d_X_orig, d_mask, d_Neighbors, N, D,
                                       K);

  cudaFree(d_Corrs);
  cudaFree(d_Neighbors);
  cudaFree(d_X_orig);
}

} // namespace impute_ls
