#include "cuda_utils.cuh"
#include "sknn_impute_gpu.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

namespace impute {

// Optimized: Warp-parallel distance calculation (1 warp per reference row)
__global__ void sknn_distances_warp_kernel(const float *X, const uint8_t *M,
                                           int N, int D, int target_idx,
                                           const int *ref_indices, int num_ref,
                                           float *dists) {
  int warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane_id = threadIdx.x % 32;

  if (warp_global_id >= num_ref)
    return;

  int r_idx = ref_indices[warp_global_id];
  double sum = 0.0;
  int count = 0;

  // Warp-parallel loop over D features
  for (int c = lane_id; c < D; c += 32) {
    if (M[target_idx * D + c] && M[r_idx * D + c]) {
      double diff = (double)X[target_idx * D + c] - (double)X[r_idx * D + c];
      sum += diff * diff;
      count++;
    }
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    count += __shfl_down_sync(0xFFFFFFFF, count, offset);
  }

  if (lane_id == 0) {
    dists[warp_global_id] =
        (count == 0) ? 1e18f : (float)sqrt(sum * ((double)D / count));
  }
}

__global__ void sknn_impute_row_kernel(float *X, uint8_t *M, int N, int D,
                                       int target_idx,
                                       const int *neighbor_indices,
                                       const float *neighbor_dists, int K_eff) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= D || M[target_idx * D + c])
    return;

  double num = 0.0, den = 0.0;
  for (int k = 0; k < K_eff; ++k) {
    int r_idx = neighbor_indices[k];
    if (r_idx >= 0 && M[r_idx * D + c]) {
      double w = 1.0 / (double)(neighbor_dists[k] + 1e-6f);
      num += w * (double)X[r_idx * D + c];
      den += w;
    }
  }

  if (den > 0) {
    X[target_idx * D + c] = (float)(num / den);
  } else {
    // Fallback to column mean
    double csum = 0;
    int ccnt = 0;
    for (int i = 0; i < N; i++)
      if (M[i * D + c]) {
        csum += (double)X[i * D + c];
        ccnt++;
      }
    X[target_idx * D + c] = (ccnt > 0) ? (float)(csum / ccnt) : 0.0f;
  }
  M[target_idx * D + c] = 1;
}

void sknn_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                      cudaStream_t stream) {
  if (N <= 0 || D <= 0)
    return;

  // Copy mask to host for initial setup (one-time cost)
  std::vector<uint8_t> h_M(N * D);
  CUDA_CHECK(cudaMemcpyAsync(h_M.data(), d_mask, sizeof(uint8_t) * N * D,
                             cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Identify reference and target rows
  std::vector<int> ref_indices, tgt_indices;
  for (int i = 0; i < N; ++i) {
    bool has_missing = false;
    for (int c = 0; c < D; ++c)
      if (!h_M[i * D + c]) {
        has_missing = true;
        break;
      }
    if (has_missing)
      tgt_indices.push_back(i);
    else
      ref_indices.push_back(i);
  }

  if (ref_indices.empty() && !tgt_indices.empty()) {
    ref_indices.push_back(tgt_indices[0]);
    tgt_indices.erase(tgt_indices.begin());
  }

  // Sort targets by missingness (fewer missing first)
  std::sort(tgt_indices.begin(), tgt_indices.end(), [&](int a, int b) {
    int ma = 0, mb = 0;
    for (int c = 0; c < D; ++c) {
      if (!h_M[a * D + c])
        ma++;
      if (!h_M[b * D + c])
        mb++;
    }
    return ma < mb;
  });

  if (tgt_indices.empty())
    return;

  // Pre-allocate GPU buffers
  int *d_ref_indices, *d_knn_indices;
  float *d_dists, *d_knn_dists;
  CUDA_CHECK(cudaMalloc(&d_ref_indices, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dists, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_knn_indices, K * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_knn_dists, K * sizeof(float)));

  std::vector<int> current_ref = ref_indices;
  std::vector<float> h_dists(N);
  std::vector<int> knn_indices(K);
  std::vector<float> knn_dists(K);

  // Main loop
  for (int target_idx : tgt_indices) {
    int num_ref = (int)current_ref.size();

    // Upload reference indices
    CUDA_CHECK(cudaMemcpyAsync(d_ref_indices, current_ref.data(),
                               num_ref * sizeof(int), cudaMemcpyHostToDevice,
                               stream));

    // Compute distances with warp-parallel kernel (key optimization!)
    int warps_needed = num_ref;
    int threads_per_block = 256;
    int blocks =
        (warps_needed * 32 + threads_per_block - 1) / threads_per_block;
    sknn_distances_warp_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_X, d_mask, N, D, target_idx, d_ref_indices, num_ref, d_dists);

    // Download distances and do selection on CPU (safe, still fast for <12k
    // refs)
    CUDA_CHECK(cudaMemcpyAsync(h_dists.data(), d_dists, num_ref * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Build neighbor list
    std::vector<std::pair<float, int>> neighbors;
    neighbors.reserve(num_ref);
    for (int i = 0; i < num_ref; i++)
      if (h_dists[i] < 1e17f)
        neighbors.push_back({h_dists[i], current_ref[i]});

    if (neighbors.empty()) {
      sknn_impute_row_kernel<<<(D + 255) / 256, 256, 0, stream>>>(
          d_X, d_mask, N, D, target_idx, nullptr, nullptr, 0);
    } else {
      int K_eff = std::min(K, (int)neighbors.size());
      std::partial_sort(neighbors.begin(), neighbors.begin() + K_eff,
                        neighbors.end());

      for (int k = 0; k < K_eff; k++) {
        knn_indices[k] = neighbors[k].second;
        knn_dists[k] = neighbors[k].first;
      }

      CUDA_CHECK(cudaMemcpyAsync(d_knn_indices, knn_indices.data(),
                                 K_eff * sizeof(int), cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(d_knn_dists, knn_dists.data(),
                                 K_eff * sizeof(float), cudaMemcpyHostToDevice,
                                 stream));

      sknn_impute_row_kernel<<<(D + 255) / 256, 256, 0, stream>>>(
          d_X, d_mask, N, D, target_idx, d_knn_indices, d_knn_dists, K_eff);
    }
    current_ref.push_back(target_idx);
  }

  cudaStreamSynchronize(stream);

  CUDA_CHECK(cudaFree(d_ref_indices));
  CUDA_CHECK(cudaFree(d_dists));
  CUDA_CHECK(cudaFree(d_knn_indices));
  CUDA_CHECK(cudaFree(d_knn_dists));
}

} // namespace impute
