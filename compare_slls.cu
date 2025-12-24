#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


#include "imputation_lib/i_imputer.hpp"

// Include CPU implementation
// Map 'impute' namespace to 'cpu_algo' locally if needed, but they share
// 'impute'. We rely on different class/function names.
#include "imputation_lib/cpu_impute/cpu_slls.cpp"

// Include GPU implementation (host-side logic included directly)
#include "imputation_lib/gpu/slls_impute_gpu.cu"

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
  int N = 6;
  int D = 4;

  // Toy Data:
  // Generate a correlated dataset.
  // Col 0 = Base
  // Col 1 = 2 * Col 0
  // Col 2 = 0.5 * Col 0 + 1
  // Col 3 = Noise

  std::vector<float> X_orig(N * D);
  std::vector<uint8_t> Mask(N * D, 1);

  float Base[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  for (int i = 0; i < N; ++i) {
    X_orig[i * D + 0] = Base[i];
    X_orig[i * D + 1] = 2.0f * Base[i];
    X_orig[i * D + 2] = 0.5f * Base[i] + 1.0f;
    X_orig[i * D + 3] = (float)(i % 2);
  }

  // Introduce missingness
  // Case 1: X[2][1] missing (Should be 2*3.0 = 6.0)
  Mask[2 * D + 1] = 0;
  X_orig[2 * D + 1] = 0.0f;

  // Case 2: X[4][2] missing (Should be 0.5*5.0 + 1.0 = 3.5)
  Mask[4 * D + 2] = 0;
  X_orig[4 * D + 2] = 0.0f;

  std::cout << "Running Scaled LLS Comparison (Toy Dataset)\n";
  print_matrix("Input Data (0.0 = Missing)", X_orig, N, D);

  // --- CPU Run ---
  std::vector<float> X_cpu = X_orig;
  impute::SLLSImputer cpu_algo;
  cpu_algo.impute(X_cpu.data(), Mask.data(), N, D);

  // --- GPU Run (New SLLS) ---
  std::vector<float> X_new = X_orig;
  std::vector<uint8_t> M_new = Mask;

  float *d_X;
  uint8_t *d_M;
  cudaMalloc(&d_X, N * D * sizeof(float));
  cudaMalloc(&d_M, N * D * sizeof(uint8_t));

  cudaMemcpy(d_X, X_new.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_M, M_new.data(), N * D * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  // K=3 nearest neighbors
  impute::slls_impute_cuda(d_X, d_M, N, D, 3);

  cudaMemcpy(X_new.data(), d_X, N * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_X);
  cudaFree(d_M);

  // --- Compare ---
  print_matrix("CPU Result (cpu_slls.cpp)", X_cpu, N, D);
  print_matrix("GPU Result (slls_impute_gpu.cu)", X_new, N, D);

  printf("Expected[2,1] ~ 6.0. CPU: %.4f, New: %.4f\n", X_cpu[2 * D + 1],
         X_new[2 * D + 1]);
  printf("Expected[4,2] ~ 3.5. CPU: %.4f, New: %.4f\n", X_cpu[4 * D + 2],
         X_new[4 * D + 2]);

  return 0;
}
