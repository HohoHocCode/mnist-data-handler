#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


#include "imputation_lib/gpu/cmve_impute_gpu.cu"

// Dataset Constants
const int N_FULL = 11756;
const int D_FULL = 200;

template <typename T> std::vector<T> load_bin(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    std::cerr << "Failed to open " << path << "\n";
    exit(1);
  }
  size_t size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<T> data(size / sizeof(T));
  f.read(reinterpret_cast<char *>(data.data()), size);
  return data;
}

int main() {
  std::string base = "d:/icu_dataset/processed/cuda_eval/";

  std::cout << "Loading Evaluation Data...\n";
  auto X = load_bin<float>(base + "X_miss_float32.bin");
  auto M = load_bin<uint8_t>(base + "M_train_uint8.bin");
  auto h_idx = load_bin<int32_t>(base + "holdout_idx_int32.bin");
  auto h_y = load_bin<float>(base + "holdout_y_float32.bin");

  if (X.size() != N_FULL * D_FULL) {
    std::cerr << "Dimension mismatch: " << X.size() << " vs " << N_FULL * D_FULL
              << "\n";
    return 1;
  }

  // Parameters: K=10 (Collateral Genes)
  std::cout << "Running CMVE Imputation (K=10)...\n";

  float *d_X;
  uint8_t *d_M;
  cudaMalloc(&d_X, X.size() * sizeof(float));
  cudaMalloc(&d_M, M.size() * sizeof(uint8_t));

  cudaMemcpy(d_X, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_M, M.data(), M.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // Run CMVE
  auto start = std::chrono::high_resolution_clock::now();
  impute::cmve_impute_cuda(d_X, d_M, N_FULL, D_FULL, 10);
  auto end = std::chrono::high_resolution_clock::now();

  std::vector<float> X_imp(X.size());
  cudaMemcpy(X_imp.data(), d_X, X.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_X);
  cudaFree(d_M);

  double duration = std::chrono::duration<double>(end - start).count();
  std::cout << "CMVE Time: " << duration << " s\n";

  // Evaluate MAE
  double sum_ae = 0.0;
  size_t n_holdout = h_idx.size();

  for (size_t i = 0; i < n_holdout; i++) {
    int idx = h_idx[i]; // Flat index
    double pred = (double)X_imp[idx];
    double truth = (double)h_y[i];
    sum_ae += std::abs(pred - truth);
  }

  double mae = (n_holdout > 0) ? (sum_ae / n_holdout) : 0.0;

  std::cout << "\nValidation Results:\n";
  std::cout << "Holdout Count: " << n_holdout << "\n";
  std::cout << "MAE: " << mae << "\n";

  return 0;
}
