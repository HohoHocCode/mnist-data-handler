#pragma once
#include <cfloat>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace impute {

// Initial Mean Fill Kernel
__global__ void mean_fill_kernel(float *X, const uint8_t *Mask, int N, int D) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= D)
    return;

  double sum = 0;
  int count = 0;
  for (int i = 0; i < N; ++i) {
    if (Mask[i * D + j] == 1) {
      sum += X[i * D + j];
      count++;
    }
  }
  float mean = (count > 0) ? (float)(sum / count) : 0.0f;

  for (int i = 0; i < N; ++i) {
    if (Mask[i * D + j] == 0)
      X[i * D + j] = mean;
  }
}

// Warp Reduction Helper (Float)
__device__ inline float warpSum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

// Warp Reduction Helper (Double)
__device__ inline double warpSumDouble(double val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

// Block Reduction Helper (Double)
__device__ inline double blockSumDouble(double val) {
  __shared__ double s_block_sum[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;
  val = warpSumDouble(val);
  if (lane == 0)
    s_block_sum[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x / 32)) ? s_block_sum[lane] : 0.0;
  if (wid == 0)
    val = warpSumDouble(val);
  return val; // Thread 0 has the sum
}

// Block Reduction Helper (Int)
__device__ inline int blockSumInt(int val) {
  __shared__ int s_block_int[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  if (lane == 0)
    s_block_int[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x / 32)) ? s_block_int[lane] : 0;
  if (wid == 0) {
    for (int offset = 16; offset > 0; offset /= 2)
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// Optimized Select Neighbors Kernel (Distributed)
__global__ void select_neighbors_ills_kernel(const float *X, int *Neighbors,
                                             int N, int D, int K) {
  int i = blockIdx.x;
  if (i >= N)
    return;

  int tid = threadIdx.x;
  int lane = tid % 32;
  int wid = tid / 32;

  struct Neighbor {
    int id;
    float dist;
  };
  Neighbor topk[128];
  for (int k = 0; k < K; ++k)
    topk[k] = {-1, FLT_MAX};

  for (int r = wid; r < N; r += (blockDim.x / 32)) {
    if (r == i)
      continue;

    float d2_partial = 0;
    for (int c = lane; c < D; c += 32) {
      float diff = X[i * D + c] - X[r * D + c];
      d2_partial += diff * diff;
    }

    float dist_sq = d2_partial;
    for (int offset = 16; offset > 0; offset /= 2)
      dist_sq += __shfl_down_sync(0xFFFFFFFF, dist_sq, offset);
    dist_sq = __shfl_sync(0xFFFFFFFF, dist_sq, 0);

    if (lane == 0) {
      if (dist_sq < topk[K - 1].dist) {
        topk[K - 1] = {r, dist_sq};
        for (int k = K - 2; k >= 0; --k) {
          if (topk[k + 1].dist < topk[k].dist) {
            Neighbor tmp = topk[k];
            topk[k] = topk[k + 1];
            topk[k + 1] = tmp;
          } else
            break;
        }
      }
    }
  }

  __shared__ Neighbor shared_top[32 * 128];
  if (lane == 0) {
    for (int k = 0; k < K; ++k)
      shared_top[wid * K + k] = topk[k];
  }
  __syncthreads();

  if (tid == 0) {
    Neighbor final_top[128];
    for (int k = 0; k < K; ++k)
      final_top[k] = {-1, FLT_MAX};
    int n_warps = blockDim.x / 32;

    for (int w = 0; w < n_warps; ++w) {
      for (int k = 0; k < K; ++k) {
        Neighbor cand = shared_top[w * K + k];
        if (cand.dist < final_top[K - 1].dist) {
          final_top[K - 1] = cand;
          for (int j = K - 2; j >= 0; --j) {
            if (final_top[j + 1].dist < final_top[j].dist) {
              Neighbor tmp = final_top[j];
              final_top[j] = final_top[j + 1];
              final_top[j + 1] = tmp;
            } else
              break;
          }
        }
      }
    }
    for (int k = 0; k < K; ++k)
      Neighbors[i * K + k] = final_top[k].id;
  }
}

// Optimized Select Neighbors Kernel with Strict Filter (for LLS)
__global__ void select_neighbors_lls_kernel(const float *X, const uint8_t *Mask,
                                            int *Neighbors, int N, int D,
                                            int K) {
  int i = blockIdx.x;
  if (i >= N)
    return;

  int tid = threadIdx.x;
  int lane = tid % 32;
  int wid = tid / 32;

  struct Neighbor {
    int id;
    float dist;
  };
  Neighbor topk[128];
  for (int k = 0; k < K; ++k)
    topk[k] = {-1, FLT_MAX};

  for (int r = wid; r < N; r += (blockDim.x / 32)) {
    if (r == i)
      continue;

    bool valid = true;
    for (int c = lane; c < D; c += 32) {
      bool fail = (Mask[i * D + c] == 0 && Mask[r * D + c] == 0);
      if (__any_sync(0xFFFFFFFF, fail)) {
        valid = false;
        break;
      }
    }
    if (!valid)
      continue;

    float d2_partial = 0;
    int count_partial = 0;
    for (int c = lane; c < D; c += 32) {
      if (Mask[i * D + c] == 1 && Mask[r * D + c] == 1) {
        float diff = X[i * D + c] - X[r * D + c];
        d2_partial += diff * diff;
        count_partial++;
      }
    }

    float dist_sq = d2_partial;
    int count = count_partial;
    for (int offset = 16; offset > 0; offset /= 2) {
      dist_sq += __shfl_down_sync(0xFFFFFFFF, dist_sq, offset);
      count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }
    dist_sq = __shfl_sync(0xFFFFFFFF, dist_sq, 0);
    count = __shfl_sync(0xFFFFFFFF, count, 0);

    if (lane == 0 && count > 0) {
      float dist = dist_sq / count;
      if (dist < topk[K - 1].dist) {
        topk[K - 1] = {r, dist};
        for (int k = K - 2; k >= 0; --k) {
          if (topk[k + 1].dist < topk[k].dist) {
            Neighbor tmp = topk[k];
            topk[k] = topk[k + 1];
            topk[k + 1] = tmp;
          } else
            break;
        }
      }
    }
  }

  __shared__ Neighbor shared_top[32 * 128];
  if (lane == 0) {
    for (int k = 0; k < K; ++k)
      shared_top[wid * K + k] = topk[k];
  }
  __syncthreads();

  if (tid == 0) {
    Neighbor final_top[128];
    for (int k = 0; k < K; ++k)
      final_top[k] = {-1, FLT_MAX};
    int n_warps = blockDim.x / 32;
    for (int w = 0; w < n_warps; ++w) {
      for (int k = 0; k < K; ++k) {
        Neighbor cand = shared_top[w * K + k];
        if (cand.dist < final_top[K - 1].dist) {
          final_top[K - 1] = cand;
          for (int j = K - 2; j >= 0; --j) {
            if (final_top[j + 1].dist < final_top[j].dist) {
              Neighbor tmp = final_top[j];
              final_top[j] = final_top[j + 1];
              final_top[j + 1] = tmp;
            } else
              break;
          }
        }
      }
    }
    for (int k = 0; k < K; ++k)
      Neighbors[i * K + k] = final_top[k].id;
  }
}
} // namespace impute
