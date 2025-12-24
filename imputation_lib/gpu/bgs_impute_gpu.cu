#pragma once
#include "bgs_impute_gpu.cuh"
#include "common_kernels.cu"
#include "cuda_utils.cuh"
#include <cstdio>

namespace impute {

__device__ inline void solve_bgs_eigen_device_opt(int n, double *U, double *V,
                                                  double *b, float *x) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      V[i * n + j] = (i == j) ? 1.0 : 0.0;
    }
  }

  for (int sweep = 0; sweep < 20; ++sweep) {
    bool changed = false;
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        double g = U[i * n + i];
        double h = U[j * n + j];
        double f = U[i * n + j];
        if (abs(f) > 1e-12) {
          changed = true;
          double zeta = (h - g) / (2.0 * f);
          double t =
              (zeta > 0 ? 1.0 : -1.0) / (abs(zeta) + sqrt(1.0 + zeta * zeta));
          double cs = 1.0 / sqrt(1.0 + t * t);
          double sn = t * cs;
          for (int k = 0; k < n; k++) {
            double v1 = U[k * n + i], v2 = U[k * n + j];
            U[k * n + i] = cs * v1 - sn * v2;
            U[k * n + j] = sn * v1 + cs * v2;
          }
          for (int k = 0; k < n; k++) {
            double v1 = V[k * n + i], v2 = V[k * n + j];
            V[k * n + i] = cs * v1 - sn * v2;
            V[k * n + j] = sn * v1 + cs * v2;
          }
        }
      }
    }
    if (!changed)
      break;
  }

  double b_proj[128];
  for (int j = 0; j < n; j++) {
    b_proj[j] = 0;
    for (int i = 0; i < n; i++)
      b_proj[j] += V[i * n + j] * b[i];
  }

  double x_proj[128];
  for (int j = 0; j < n; j++) {
    double eig = 0;
    for (int i = 0; i < n; i++)
      eig += U[i * n + j] * V[i * n + j];
    if (eig > 1e-9)
      x_proj[j] = b_proj[j] / eig;
    else
      x_proj[j] = 0;
  }

  for (int i = 0; i < n; i++) {
    double res = 0;
    for (int j = 0; j < n; j++)
      res += V[i * n + j] * x_proj[j];
    x[i] = (float)res;
  }
}

__global__ void bgs_optimized_solve_kernel(
    float *X_out, const uint8_t *Mask, const int *Neighbors,
    const float *X_orig, const float *X_curr, const float *col_means,
    double *GlobalWorkspace, int N, int D, int K, float ridge_base) {
  int i = blockIdx.x;
  if (i >= N)
    return;
  int tx = threadIdx.x;
  int lane = tx % 32;
  int wid = tx / 32;
  int num_warps = blockDim.x / 32;

  size_t row_stride = (size_t)K * K * 2;
  double *ATA = GlobalWorkspace + (i * row_stride);
  double *V_mat = ATA + (K * K);

  extern __shared__ double bgs_shmem_opt[];
  double *ATy = bgs_shmem_opt;
  double *dist2 = ATy + K;
  double *n_means = dist2 + K;

  __shared__ float h_shared;
  __shared__ float bgs_coeffs[128];

  __shared__ int n_neigh_eff;
  if (tx == 0) {
    int cnt = 0;
    for (int k = 0; k < K; k++)
      if (Neighbors[i * K + k] != -1)
        cnt++;
    n_neigh_eff = cnt;
  }
  __syncthreads();
  if (n_neigh_eff == 0)
    return;

  // 1. Distance for weighting
  for (int k = tx; k < K; k += blockDim.x) {
    int nid = Neighbors[i * K + k];
    if (nid == -1) {
      dist2[k] = 1e20;
      continue;
    }
    double d2 = 0;
    for (int c = 0; c < D; c++) {
      float diff = X_curr[i * D + c] - X_curr[nid * D + c];
      d2 += (double)(diff * diff);
    }
    dist2[k] = d2;
  }
  __syncthreads();

  if (tx == 0) {
    double max_d2 = 1e-6;
    for (int k = 0; k < K; k++) {
      if (Neighbors[i * K + k] != -1 && dist2[k] > max_d2)
        max_d2 = dist2[k];
    }
    h_shared = (float)max_d2;
  }
  __syncthreads();

  // 2. n_means
  for (int j = tx; j < D; j += blockDim.x) {
    double sum = 0;
    for (int k = 0; k < K; k++) {
      int nid = Neighbors[i * K + k];
      if (nid != -1)
        sum += (double)X_curr[nid * D + j];
    }
    n_means[j] = sum / n_neigh_eff;
  }
  __syncthreads();

  // 3. Accumulate ATA and ATy
  for (int k = tx; k < K; k += blockDim.x)
    ATy[k] = 0.0;
  for (int idx = tx; idx < K * K; idx += blockDim.x)
    ATA[idx] = 0.0;
  __syncthreads();

  for (int k_outer = 0; k_outer < K; k_outer++) {
    int nid_outer = Neighbors[i * K + k_outer];
    if (nid_outer == -1)
      continue;

    // ATy[k_outer]
    double aty_val = 0;
    for (int c = tx; c < D; c += blockDim.x) {
      if (Mask[i * D + c] == 1) {
        double y_cent = (double)X_orig[i * D + c] - n_means[c];
        double x_cent = (double)X_curr[nid_outer * D + c] - n_means[c];
        aty_val += x_cent * y_cent;
      }
    }
    aty_val = warpSumDouble(aty_val);
    if (lane == 0)
      atomicAdd(&ATy[k_outer], aty_val);

    // ATA[k_outer, k_inner]
    for (int k_inner = wid; k_inner < K; k_inner += num_warps) {
      int nid_inner = Neighbors[i * K + k_inner];
      if (nid_inner == -1)
        continue;
      double ata_val = 0;
      for (int c = lane; c < D; c += 32) {
        if (Mask[i * D + c] == 1) {
          double v1 = (double)X_curr[nid_outer * D + c] - n_means[c];
          double v2 = (double)X_curr[nid_inner * D + c] - n_means[c];
          ata_val += v1 * v2;
        }
      }
      ata_val = warpSumDouble(ata_val);
      if (lane == 0)
        atomicAdd(&ATA[k_outer * K + k_inner], ata_val);
    }
  }
  __syncthreads();

  // 4. Solve
  if (tx == 0) {
    for (int k = 0; k < K; k++) {
      if (Neighbors[i * K + k] != -1) {
        float v_k = exp(-dist2[k] / h_shared);
        ATA[k * K + k] += (double)ridge_base / max(v_k, 1e-6f);
      }
    }
    solve_bgs_eigen_device_opt(n_neigh_eff, ATA, V_mat, ATy, bgs_coeffs);
  }
  __syncthreads();

  // 5. Predict
  for (int c = tx; c < D; c += blockDim.x) {
    if (Mask[i * D + c] == 0) {
      double pred = n_means[c];
      for (int k = 0; k < K; k++) {
        int nid = Neighbors[i * K + k];
        if (nid != -1)
          pred += (double)bgs_coeffs[k] *
                  ((double)X_curr[nid * D + c] - n_means[c]);
      }
      if (isnan(pred) || isinf(pred))
        pred = (double)col_means[c];
      X_out[i * D + c] = (float)pred;
    }
  }
}

__global__ void compute_bgs_means_kernel(const float *X, const uint8_t *Mask,
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

void bgs_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int MaxGenes,
                     float Ridge, cudaStream_t stream) {
  if (N <= 0 || D <= 0)
    return;

  // 1. Prep d_X_curr (mean-filled fixed input)
  float *d_X_orig;
  cudaMalloc(&d_X_orig, (size_t)N * D * sizeof(float));
  cudaMemcpyAsync(d_X_orig, d_X, (size_t)N * D * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);

  float *d_col_means;
  cudaMalloc(&d_col_means, D * sizeof(float));
  compute_bgs_means_kernel<<<(D + 255) / 256, 256, 0, stream>>>(d_X, d_mask, N,
                                                                D, d_col_means);

  float *d_X_curr;
  cudaMalloc(&d_X_curr, (size_t)N * D * sizeof(float));
  cudaMemcpyAsync(d_X_curr, d_X, (size_t)N * D * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  mean_fill_kernel<<<(D + 255) / 256, 256, 0, stream>>>(d_X_curr, d_mask, N, D);

  // 2. Neighbor selection
  int *d_Neighbors;
  cudaMalloc(&d_Neighbors, (size_t)N * MaxGenes * sizeof(int));
  select_neighbors_ills_kernel<<<N, 128, 0, stream>>>(d_X_curr, d_Neighbors, N,
                                                      D, MaxGenes);

  // 3. Solve
  double *d_GlobalWorkspace;
  size_t ws_size = (size_t)N * MaxGenes * MaxGenes * 2 * sizeof(double);
  cudaMalloc(&d_GlobalWorkspace, ws_size);

  int shmem_size =
      (MaxGenes * sizeof(double)) * 2 + (D * sizeof(double)) + 1024;
  bgs_optimized_solve_kernel<<<N, 128, shmem_size, stream>>>(
      d_X, d_mask, d_Neighbors, d_X_orig, d_X_curr, d_col_means,
      d_GlobalWorkspace, N, D, MaxGenes, Ridge);

  cudaFree(d_X_orig);
  cudaFree(d_X_curr);
  cudaFree(d_Neighbors);
  cudaFree(d_col_means);
  cudaFree(d_GlobalWorkspace);
}

} // namespace impute
