#include "svd_impute_gpu.cuh"
#include "svd_workspace.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cuda_utils.cuh"
#include <cub/cub.cuh>

namespace impute {

// ---------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------

__global__ void fast_col_mean_kernel(const float *X, const uint8_t *mask, int N,
                                     int D, float *mean) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= D)
    return;

  double sum = 0.0;
  int cnt = 0;

  for (int r = 0; r < N; r++) {
    int idx = r * D + c;
    if (mask[idx] == 1) {
      sum += (double)X[idx];
      cnt++;
    }
  }
  mean[c] = (cnt > 0) ? (float)(sum / cnt) : 0.0f;
}

// Optimization: Even faster mean using 1 warp per column if N is large
__global__ void warp_col_mean_kernel(const float *X, const uint8_t *mask, int N,
                                     int D, float *mean) {
  int c = blockIdx.x; // Block handles one column
  if (c >= D)
    return;

  int tx = threadIdx.x;
  double local_sum = 0.0;
  int local_cnt = 0;

  for (int r = tx; r < N; r += blockDim.x) {
    int idx = r * D + c;
    if (mask[idx] == 1) {
      local_sum += (double)X[idx];
      local_cnt++;
    }
  }

  // Reduce within block
  typedef cub::BlockReduce<double, 256> BlockReduceSum;
  typedef cub::BlockReduce<int, 256> BlockReduceCnt;
  __shared__ union {
    typename BlockReduceSum::TempStorage sum;
    typename BlockReduceCnt::TempStorage cnt;
  } temp_storage;

  double total_sum = BlockReduceSum(temp_storage.sum).Sum(local_sum);
  __syncthreads();
  int total_cnt = BlockReduceCnt(temp_storage.cnt).Sum(local_cnt);

  if (tx == 0) {
    mean[c] = (total_cnt > 0) ? (float)(total_sum / total_cnt) : 0.0f;
  }
}

__global__ void fill_mean_kernel(float *X, const uint8_t *mask,
                                 const float *mean, int N, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * D)
    return;
  if (mask[idx] == 0) { // MISSING
    int c = idx % D;
    X[idx] = mean[c];
  }
}

__global__ void row_to_col_kernel(const float *src, float *dst, int N, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * D)
    return;

  int r = idx / D;
  int c = idx % D;
  dst[c * N + r] = src[idx];
}

__global__ void col_to_row_kernel(const float *src, float *dst, int N, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * D)
    return;

  int r = idx / D;
  int c = idx % D;
  dst[idx] = src[c * N + r];
}

__global__ void col_to_row_update_missing_kernel(const float *src, float *dst,
                                                 const uint8_t *mask, int N,
                                                 int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * D)
    return;

  if (mask[idx] == 0) { // MISSING
    int r = idx / D;
    int c = idx % D;
    dst[idx] = src[c * N + r];
  }
}

__global__ void scale_columns_kernel(float *U, const float *S, int N, int R) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * R)
    return;

  int r = idx % N;
  int c = idx / N;

  U[c * N + r] *= S[c];
}

// ---------------------------------------------------------------------
// Main SVD-impute
// ---------------------------------------------------------------------

void svd_impute(float *d_X, uint8_t *d_mask, int N, int D, int R, int max_iter,
                float tol, cudaStream_t stream) {
  SvdWorkspace ws(N, D, R);
  ws.allocate();

  for (int it = 0; it < max_iter; it++) {
    // --- compute column means (OPTIMIZED)
    warp_col_mean_kernel<<<D, 256, 0, stream>>>(d_X, d_mask, N, D, ws.d_means);

    // --- fill missing by mean
    fill_mean_kernel<<<(N * D + 255) / 256, 256, 0, stream>>>(d_X, d_mask,
                                                              ws.d_means, N, D);

    // --- convert row-major → col-major
    row_to_col_kernel<<<(N * D + 255) / 256, 256, 0, stream>>>(d_X, ws.d_A_col,
                                                               N, D);

    int min_dim = (N < D) ? N : D;
    signed char jobu = 'S';
    signed char jobvt = 'S';

    if (N >= D) {
      CUSOLVER_CHECK(cusolverDnSgesvd(
          ws.solver, jobu, jobvt, N, D, ws.d_A_col, N, ws.d_S, ws.d_U, N,
          ws.d_VT, min_dim, ws.d_work, ws.work_size, ws.d_rwork, ws.d_info));

      // Standard Reconstruct A = U * S * VT
      scale_columns_kernel<<<(N * R + 255) / 256, 256, 0, stream>>>(
          ws.d_U, ws.d_S, N, R);
      float a = 1.f, b = 0.f;
      CUBLAS_CHECK(cublasSgemm(ws.cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, D, R, &a,
                               ws.d_U, N, ws.d_VT, min_dim, &b, ws.d_recon_col,
                               N));
    } else {
      col_to_row_kernel<<<(N * D + 255) / 256, 256, 0, stream>>>(
          ws.d_A_col, ws.d_recon_col, N, D);

      CUSOLVER_CHECK(cusolverDnSgesvd(
          ws.solver, jobu, jobvt, D, N, ws.d_recon_col, D, ws.d_S, ws.d_VT, D,
          ws.d_U, min_dim, ws.d_work, ws.work_size, ws.d_rwork, ws.d_info));

      scale_columns_kernel<<<(min_dim * R + 255) / 256, 256, 0, stream>>>(
          ws.d_VT, ws.d_S, D, R);

      float a = 1.f, b = 0.f;
      CUBLAS_CHECK(cublasSgemm(ws.cublas, CUBLAS_OP_T, CUBLAS_OP_T, N, D, R, &a,
                               ws.d_U, min_dim, ws.d_VT, D, &b, ws.d_A_col, N));
      CUDA_CHECK(cudaMemcpyAsync(ws.d_recon_col, ws.d_A_col,
                                 sizeof(float) * N * D,
                                 cudaMemcpyDeviceToDevice, stream));
    }

    // Check info
    int h_info = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_info, ws.d_info, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_info != 0) {
      printf("SVD FAILURE: Info = %d at iter %d\n", h_info, it);
    }

    // --- update missing values in d_X using reconstruction (d_recon_col is
    // col-major)
    col_to_row_update_missing_kernel<<<(N * D + 255) / 256, 256, 0, stream>>>(
        ws.d_recon_col, d_X, d_mask, N, D);
  }

  cudaStreamSynchronize(stream);
}
} // namespace impute
