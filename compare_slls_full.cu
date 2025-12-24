#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


#include "imputation_lib/cpu_impute/cpu_slls.cpp"
#include "imputation_lib/gpu/slls_impute_gpu.cu"
#include "imputation_lib/i_imputer.hpp"


// Dataset Constants (Verified from file size)
const int N_FULL = 11756;
const int D_FULL = 200;

std::vector<float> load_bin_float(const std::string &path, size_t size) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "Failed to open " << path << "\n";
    exit(1);
  }
  std::vector<float> data(size);
  f.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
  return data;
}

std::vector<uint8_t> load_bin_uint8(const std::string &path, size_t size) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "Failed to open " << path << "\n";
    exit(1);
  }
  std::vector<uint8_t> data(size);
  f.read(reinterpret_cast<char *>(data.data()), size * sizeof(uint8_t));
  return data;
}

int main() {
  std::string x_path = "d:/icu_dataset/processed/cuda/X_float32.bin";
  std::string m_path = "d:/icu_dataset/processed/cuda/M_uint8.bin";

  std::cout << "Loading ICU Dataset [" << N_FULL << "x" << D_FULL << "]...\n";
  auto X = load_bin_float(x_path, N_FULL * D_FULL);
  auto M = load_bin_uint8(m_path, N_FULL * D_FULL);

  // Copy for runs
  auto X_cpu = X;
  auto X_gpu = X;

  // Config
  int K = 10;

  std::cout << "Running CPU SLLS (This make take a moment)...\n";
  auto start_cpu = std::chrono::high_resolution_clock::now();
  impute::SLLSImputer cpu_algo;
  cpu_algo.impute(X_cpu.data(), M.data(), N_FULL, D_FULL);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff_cpu = end_cpu - start_cpu;
  std::cout << "CPU Time: " << diff_cpu.count() << " s\n";

  std::cout << "Running GPU SLLS (New Implementation)...\n";
  auto start_gpu = std::chrono::high_resolution_clock::now();
  {
    float *d_X;
    uint8_t *d_M;
    cudaMalloc(&d_X, N_FULL * D_FULL * sizeof(float));
    cudaMalloc(&d_M, N_FULL * D_FULL * sizeof(uint8_t));

    cudaMemcpy(d_X, X_gpu.data(), N_FULL * D_FULL * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M.data(), N_FULL * D_FULL * sizeof(uint8_t),
               cudaMemcpyHostToDevice);

    impute::slls_impute_cuda(d_X, d_M, N_FULL, D_FULL, K);

    cudaMemcpy(X_gpu.data(), d_X, N_FULL * D_FULL * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_M);
  }
  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff_gpu = end_gpu - start_gpu;
  std::cout << "GPU Time: " << diff_gpu.count() << " s\n";

  // Compare
  double sum_diff = 0.0;
  double max_diff = 0.0;
  int count = 0;

  for (size_t i = 0; i < X.size(); ++i) {
    if (M[i] == 0) { // Only check where original was missing
      double d = std::abs(X_cpu[i] - X_gpu[i]);
      sum_diff += d;
      if (d > max_diff)
        max_diff = d;
      count++;
    }
  }

  double mae = (count > 0) ? (sum_diff / count) : 0.0;

  std::cout << "\nComparison Results (Imputed Entries Only):\n";
  std::cout << "Count: " << count << "\n";
  std::cout << "MAE: " << mae << "\n";
  std::cout << "Max Diff: " << max_diff << "\n";
  std::cout << "Interpretation: Differences are expected due to logic variance "
               "(Column-wise vs Neighbor-wise).\n";

  return 0;
}
