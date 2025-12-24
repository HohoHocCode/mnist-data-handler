#include "imputation_lib/gpu/cmve_impute_gpu.cu"
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


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
  int N = 10;
  int D = 4;
  // Data: Perfectly correlated
  // C0 = 1..10
  // C1 = 2*C0 + 1  (Corr 1.0 with C0)
  // C2 = 10 - C0   (Corr -1.0 with C0)
  // C3 = Noise

  std::vector<float> X_orig(N * D);
  std::vector<uint8_t> Mask(N * D, 1);

  for (int i = 0; i < N; i++) {
    float base = i + 1.0f;
    X_orig[i * D + 0] = base;
    X_orig[i * D + 1] = 2.0f * base + 1.0f;
    X_orig[i * D + 2] = 10.0f - base;
    X_orig[i * D + 3] = (float)((i * 7) % 3);
  }

  // Missing:
  // C1[5] (Base=6.0 -> Expect 2*6+1=13.0)
  Mask[5 * D + 1] = 0;
  X_orig[5 * D + 1] = 0.0f;

  // C2[8] (Base=9.0 -> Expect 10-9=1.0)
  Mask[8 * D + 2] = 0;
  X_orig[8 * D + 2] = 0.0f;

  std::cout << "Running CMVE Test (Toy Linear)\n";
  // print_matrix("Input", X_orig, N, D);

  float *d_X;
  uint8_t *d_M;
  cudaMalloc(&d_X, N * D * sizeof(float));
  cudaMalloc(&d_M, N * D * sizeof(uint8_t));

  cudaMemcpy(d_X, X_orig.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_M, Mask.data(), N * D * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // K=2, should pick C0 and C2 for C1
  impute::cmve_impute_cuda(d_X, d_M, N, D, 2);

  std::vector<float> X_new = X_orig;
  cudaMemcpy(X_new.data(), d_X, N * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_X);
  cudaFree(d_M);

  printf("Expected[5,1] ~ 13.0. Result: %.4f\n", X_new[5 * D + 1]);
  printf("Expected[8,2] ~ 1.0. Result: %.4f\n", X_new[8 * D + 2]);

  if (std::abs(X_new[5 * D + 1] - 13.0f) < 0.1f &&
      std::abs(X_new[8 * D + 2] - 1.0f) < 0.1f) {
    printf("SUCCESS: CMVE imputed linearly correlated data.\n");
  } else {
    printf("FAILURE: CMVE Inaccurate.\n");
  }

  return 0;
}
