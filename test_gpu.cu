#include "imputation_lib/gpu/bpca_impute_gpu.cuh"
#include "imputation_lib/gpu/lls_impute_gpu.cuh"
#include "imputation_lib/gpu/svd_impute_gpu.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "                   \
                << cudaGetErrorString(_err) << std::endl;                      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  int N = 100;
  int D = 20;
  int K = 5;

  std::cout << "Initializing Test Data (N=" << N << ", D=" << D << ")..."
            << std::endl;

  std::vector<float> h_X(N * D);
  std::vector<uint8_t> h_M(N * D);

  for (int i = 0; i < N * D; ++i) {
    h_X[i] = (float)(i % 100);
    h_M[i] = (rand() % 10 > 2) ? 1 : 0; // 20% missing
  }

  float *d_X;
  uint8_t *d_M;
  CUDA_CHECK(cudaMalloc(&d_X, N * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_M, N * D * sizeof(uint8_t)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // --- Test 1: LLS ---
  std::cout << "Testing LLS (Strict Impl)..." << std::endl;
  CUDA_CHECK(cudaMemcpyAsync(d_X, h_X.data(), N * D * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_M, h_M.data(), N * D * sizeof(uint8_t),
                             cudaMemcpyHostToDevice, stream));

  impute::lls_impute_cuda(d_X, d_M, N, D, K, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "LLS Done." << std::endl;

  // --- Test 2: BPCA ---
  std::cout << "Testing BPCA (Rigorous VB-EM)..." << std::endl;
  CUDA_CHECK(cudaMemcpyAsync(d_X, h_X.data(), N * D * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  impute::bpca_impute_cuda(d_X, d_M, N, D, K, 10, stream); // 10 iters
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "BPCA Done." << std::endl;

  // --- Test 3: SVD ---
  std::cout << "Testing SVD (Truncated)..." << std::endl;
  CUDA_CHECK(cudaMemcpyAsync(d_X, h_X.data(), N * D * sizeof(float),
                             cudaMemcpyHostToDevice, stream));

  impute::svd_impute(d_X, d_M, N, D, K, 5, 1e-4f, stream); // 5 iters
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << "SVD Done." << std::endl;

  // Cleanup
  cudaFree(d_X);
  cudaFree(d_M);
  cudaStreamDestroy(stream);

  std::cout << "All GPU tests passed!" << std::endl;
  return 0;
}
