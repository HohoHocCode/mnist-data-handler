#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>


// Include GPU implementation (host-side logic included directly)
#include "imputation_lib/gpu/gmc_impute_gpu.cu"

void print_matrix(const char *label, const std::vector<float> &X, int N,
                  int D) {
  std::cout << "--- " << label << " ---\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      printf("%7.4f ", X[i * D + j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main() {
  int N = 20;
  int D = 2;
  // Generate 2 clusters:
  // Cluster A: Mean [2, 2], Var [0.1, 0.1], N=10
  // Cluster B: Mean [8, 8], Var [0.1, 0.1], N=10

  std::vector<float> X_orig(N * D);
  std::vector<uint8_t> Mask(N * D, 1);

  for (int i = 0; i < 10; i++) {
    X_orig[i * D + 0] = 2.0f;
    X_orig[i * D + 1] = 2.0f;
  }
  for (int i = 10; i < 20; i++) {
    X_orig[i * D + 0] = 8.0f;
    X_orig[i * D + 1] = 8.0f;
  }

  // Missing:
  // Entry [0, 1] (Cluster A, should be ~2)
  Mask[0 * D + 1] = 0;
  X_orig[0 * D + 1] = 0.0f;

  // Entry [15, 0] (Cluster B, should be ~8)
  Mask[15 * D + 0] = 0;
  X_orig[15 * D + 0] = 0.0f;

  std::cout << "Running GMC Test (Toy Clusters)\n";
  // print_matrix("Input", X_orig, N, D);

  float *d_X;
  uint8_t *d_M;
  cudaMalloc(&d_X, N * D * sizeof(float));
  cudaMalloc(&d_M, N * D * sizeof(uint8_t));

  cudaMemcpy(d_X, X_orig.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_M, Mask.data(), N * D * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // K=2, Iter=50
  impute::gmc_impute_cuda(d_X, d_M, N, D, 2, 50);

  std::vector<float> X_new = X_orig;
  cudaMemcpy(X_new.data(), d_X, N * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_X);
  cudaFree(d_M);

  printf("Expected[0,1] ~ 2.0. Result: %.4f\n", X_new[0 * D + 1]);
  printf("Expected[15,0] ~ 8.0. Result: %.4f\n", X_new[15 * D + 0]);

  if (std::abs(X_new[0 * D + 1] - 2.0f) < 0.5f &&
      std::abs(X_new[15 * D + 0] - 8.0f) < 0.5f) {
    printf("SUCCESS: GMC handled missing data correctly.\n");
  } else {
    printf("FAILURE: GMC Imputation inaccurate.\n");
  }

  return 0;
}
