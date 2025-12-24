#pragma once
#include "cuda_utils.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>


namespace metrics {

inline float sum_reduction(const float *d_in, int N, cudaStream_t stream = 0) {
  if (N == 0)
    return 0.0f;

  float *d_out = nullptr;
  CUDA_CHECK(cudaMallocAsync(&d_out, sizeof(float), stream));

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Determine temp storage size
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N,
                         stream);

  CUDA_CHECK(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

  // Run reduction
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N,
                         stream);

  float h_out = 0.0f;
  CUDA_CHECK(cudaMemcpyAsync(&h_out, d_out, sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFreeAsync(d_out, stream));
  CUDA_CHECK(cudaFreeAsync(d_temp_storage, stream));

  return h_out;
}

} // namespace metrics
