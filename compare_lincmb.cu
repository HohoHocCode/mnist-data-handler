#include "imputation_lib/cpu_impute/cpu_lincmb.cpp"
#include "imputation_lib/gpu/knn_impute_gpu.cu"
#include "imputation_lib/gpu/lincmb_impute_gpu.cu"
#include "imputation_lib/gpu/svd_impute_gpu.cu"
#include "imputation_lib/gpu/svd_workspace.cu"
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>

void compare_lincmb() {
  const int N = 100;
  const int D = 20;
  const int K = 5;
  const int R = 3;

  std::vector<float> X(N * D);
  std::vector<uint8_t> M(N * D);

  // Create synthetic data with some linear structure + noise
  for (int i = 0; i < N; i++) {
    float base = (float)(i % 5);
    for (int j = 0; j < D; j++) {
      X[i * D + j] = base + (float)j * 0.1f + (float)(rand() % 100) / 500.0f;
      M[i * D + j] = (rand() % 10 < 8) ? 1 : 0; // 20% missing
    }
  }

  std::vector<float> X_cpu = X;
  std::vector<float> X_gpu = X;

  // 1. CPU Run
  impute::CpuLinCmbImputer cpu_imp(K, R);
  cpu_imp.impute(X_cpu.data(), M.data(), N, D);

  // 2. GPU Run
  float *d_X;
  uint8_t *d_M;
  cudaMalloc(&d_X, N * D * sizeof(float));
  cudaMalloc(&d_M, N * D * sizeof(uint8_t));
  cudaMemcpy(d_X, X_gpu.data(), N * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_M, M.data(), N * D * sizeof(uint8_t), cudaMemcpyHostToDevice);

  impute::lincmb_impute_cuda(d_X, d_M, N, D, K, R);

  cudaMemcpy(X_gpu.data(), d_X, N * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_X);
  cudaFree(d_M);

  // 3. Compare Fusion Weights / Final Values
  // Note: SVD (cuSOLVER) vs ALS (CPU) will differ, so we expect some MAE.
  // However, the logic should be sound.
  double diff_sum = 0.0;
  int count = 0;
  for (int i = 0; i < N * D; i++) {
    if (M[i] == 0) {
      diff_sum += std::abs(X_cpu[i] - X_gpu[i]);
      count++;
    }
  }

  std::cout << "LinCmb CPU vs GPU Comparison (N=100, D=20)\n";
  std::cout << "Average Difference on Missing Entries: " << (diff_sum / count)
            << "\n";
  std::cout << "Note: Differences are expected due to ALS (CPU) vs SVD (GPU) "
               "divergence.\n";
}

int main() {
  srand(42);
  compare_lincmb();
  return 0;
}
