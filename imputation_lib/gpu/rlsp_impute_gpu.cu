#pragma once
#include "common_kernels.cu"
#include "cuda_utils.cuh"
#include "rlsp_impute_gpu.cuh"
#include <cstdio>

namespace impute {

// Jacobi Eigenvalue Solver for small KxK symmetric matrix
__device__ inline void solve_eigen_jacobi_kxk(int K, double *A, double *V,
                                              double *E) {
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      V[i * K + j] = (i == j) ? 1.0 : 0.0;
    }
  }

  for (int sweep = 0; sweep < 15; ++sweep) {
    double max_offdiag = 0.0;
    for (int i = 0; i < K; ++i) {
      for (int j = i + 1; j < K; ++j) {
        double h = A[i * K + j];
        if (abs(h) > 1e-12) {
          double g = A[i * K + i];
          double f = A[j * K + j];
          double a = 0.5 * (f - g) / h;
          double t = (a >= 0 ? 1.0 : -1.0) / (abs(a) + sqrt(1.0 + a * a));
          double cs = 1.0 / sqrt(1.0 + t * t);
          double sn = t * cs;

          for (int k = 0; k < K; ++k) {
            double v1 = V[k * K + i], v2 = V[k * K + j];
            V[k * K + i] = cs * v1 - sn * v2;
            V[k * K + j] = sn * v1 + cs * v2;
          }
          A[i * K + i] -= t * h;
          A[j * K + j] += t * h;
          A[i * K + j] = 0.0;
          A[j * K + i] = 0.0;
          for (int k = 0; k < K; ++k) {
            if (k != i && k != j) {
              double g1 = A[i * K + k], g2 = A[j * K + k];
              A[i * K + k] = cs * g1 - sn * g2;
              A[k * K + i] = A[i * K + k];
              A[j * K + k] = sn * g1 + cs * g2;
              A[k * K + j] = A[j * K + k];
            }
          }
        }
        max_offdiag = max(max_offdiag, abs(h));
      }
    }
    if (max_offdiag < 1e-12)
      break;
  }
  for (int i = 0; i < K; ++i)
    E[i] = A[i * K + i];
}

__global__ void rlsp_kernel(float *X, const uint8_t *Mask, const int *Neighbors,
                            int N, int D, int K, int n_pc) {
  int i = blockIdx.x;
  if (i >= N)
    return;
  int tx = threadIdx.x;

  extern __shared__ float rlsp_shared_mem[];
  double *NeighborData = (double *)rlsp_shared_mem; // K * D
  double *Cov = NeighborData + K * D;               // K * K
  double *V_eig = Cov + K * K;                      // K * K
  double *E_eig = V_eig + K * K;                    // K
  float *P_comp = (float *)(E_eig + K);             // D * n_pc
  double *means = (double *)(P_comp + D * n_pc);    // D
  __shared__ float shared_weights[32];

  // 1. Load Neighbors and Center
  for (int j = tx; j < D; j += blockDim.x) {
    double sum = 0;
    for (int r = 0; r < K; r++) {
      int nid = Neighbors[i * K + r];
      double val = (nid != -1) ? (double)X[nid * D + j] : 0.0;
      NeighborData[r * D + j] = val;
      sum += val;
    }
    means[j] = sum / K;
    for (int r = 0; r < K; r++)
      NeighborData[r * D + j] -= means[j];
  }
  __syncthreads();

  // 2. Compute Covariance KxK (A * A^T) parallelized
  for (int r = tx; r < K * K * 32; r += blockDim.x) {
    int mat_idx = r / 32;
    int lane_idx = r % 32;
    int r1 = mat_idx / K;
    int r2 = mat_idx % K;
    if (r1 >= K)
      continue;

    double p_sum = 0;
    for (int c = lane_idx; c < D; c += 32) {
      p_sum += NeighborData[r1 * D + c] * NeighborData[r2 * D + c];
    }
    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2)
      p_sum += __shfl_down_sync(0xFFFFFFFF, p_sum, offset);

    if (lane_idx == 0)
      Cov[r1 * K + r2] = p_sum;
  }
  __syncthreads();

  // 3. Solve Eigenvalues (Thread 0)
  if (tx == 0) {
    solve_eigen_jacobi_kxk(K, Cov, V_eig, E_eig);
  }
  __syncthreads();

  // 4. Compute Principal Components P = A^T * V_eig * S^-1
  int n_pc_eff = min(n_pc, K - 1);
  if (n_pc_eff < 1)
    n_pc_eff = 1;

  for (int j_idx = tx; j_idx < D * n_pc_eff * 32; j_idx += blockDim.x) {
    int task_idx = j_idx / 32;
    int lane_idx = j_idx % 32;
    int col = task_idx / n_pc_eff;
    int pc = task_idx % n_pc_eff;
    if (col >= D)
      continue;

    double p_sum = 0;
    for (int r = lane_idx; r < K; r += 32) {
      p_sum += NeighborData[r * D + col] * V_eig[r * K + pc];
    }
    for (int offset = 16; offset > 0; offset /= 2)
      p_sum += __shfl_down_sync(0xFFFFFFFF, p_sum, offset);

    if (lane_idx == 0) {
      double s = sqrt(max(E_eig[pc], 1e-12));
      P_comp[col * n_pc + pc] = (float)(p_sum / s);
    }
  }
  __syncthreads();

  // 5. Solve Regression (Thread 0)
  if (tx == 0) {
    double PtP[32 * 32] = {0};
    double Pty[32] = {0};
    int d_obs = 0;
    for (int c = 0; c < D; ++c) {
      if (Mask[i * D + c] == 1) {
        d_obs++;
        double y = (double)X[i * D + c] - means[c];
        for (int p1 = 0; p1 < n_pc_eff; ++p1) {
          double p1_val = (double)P_comp[c * n_pc + p1];
          Pty[p1] += p1_val * y;
          for (int p2 = 0; p2 < n_pc_eff; ++p2) {
            double p2_val = (double)P_comp[c * n_pc + p2];
            PtP[p1 * n_pc_eff + p2] += p1_val * p2_val;
          }
        }
      }
    }

    if (d_obs > 0) {
      for (int p = 0; p < n_pc_eff; ++p)
        PtP[p * n_pc_eff + p] += 1e-1; // Ridge
      float weights_local[32] = {0};
      solve_linear_system_device(n_pc_eff, PtP, Pty, weights_local);
      for (int p = 0; p < n_pc_eff; ++p)
        shared_weights[p] = weights_local[p];
    } else {
      for (int p = 0; p < n_pc_eff; ++p)
        shared_weights[p] = 0;
    }
  }
  __syncthreads();

  // 6. Impute
  for (int mc = tx; mc < D; mc += blockDim.x) {
    if (Mask[i * D + mc] == 0) {
      double res = means[mc];
      for (int p = 0; p < n_pc_eff; ++p) {
        res += (double)shared_weights[p] * P_comp[mc * n_pc + p];
      }
      if (!isnan(res) && !isinf(res)) {
        if (res > 1e4)
          res = 1e4;
        if (res < -1e4)
          res = -1e4;
        X[i * D + mc] = (float)res;
      }
    }
  }
}

void rlsp_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                      int n_pc, cudaStream_t stream) {
  mean_fill_kernel<<<(D + 255) / 256, 256, 0, stream>>>(d_X, d_mask, N, D);

  int *d_Neighbors;
  cudaMalloc(&d_Neighbors, N * K * sizeof(int));
  select_neighbors_ills_kernel<<<N, 128, 0, stream>>>(d_X, d_Neighbors, N, D,
                                                      K);

  int shared_bytes = (K * D * sizeof(double)) + (K * K * sizeof(double)) * 2 +
                     (K * sizeof(double)) + (D * n_pc * sizeof(float)) +
                     (D * sizeof(double)) + 1024;

  rlsp_kernel<<<N, 128, shared_bytes, stream>>>(d_X, d_mask, d_Neighbors, N, D,
                                                K, n_pc);

  cudaFree(d_Neighbors);
}

} // namespace impute
