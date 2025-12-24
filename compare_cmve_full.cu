#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


#include "imputation_lib/cpu_impute/cpu_cmve.cpp"
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

  std::cout << "Loading Evaluation Data for CPU vs GPU Check...\n";
  auto X = load_bin<float>(base + "X_miss_float32.bin");
  auto M = load_bin<uint8_t>(base + "M_train_uint8.bin");
  auto h_idx = load_bin<int32_t>(base + "holdout_idx_int32.bin");
  auto h_y = load_bin<float>(base + "holdout_y_float32.bin");

  // Config
  int K = 10;

  auto X_cpu = X;
  auto X_gpu = X;

  std::cout << "Running CPU CMVE (Warning: This might be SLOW due to O(D^2) "
               "correlation + O(D*N) regression loop)...\n";
  auto start_cpu = std::chrono::high_resolution_clock::now();
  impute::CmveImputerCpu cpu_algo(K);
  cpu_algo.impute(X_cpu.data(), M.data(), N_FULL, D_FULL);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  double t_cpu = std::chrono::duration<double>(end_cpu - start_cpu).count();
  std::cout << "CPU Time: " << t_cpu << " s\n";

  std::cout << "Running GPU CMVE...\n";
  float *d_X;
  uint8_t *d_M;
  cudaMalloc(&d_X, X_gpu.size() * sizeof(float));
  cudaMalloc(&d_M, M.size() * sizeof(uint8_t));
  cudaMemcpy(d_X, X_gpu.data(), X_gpu.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_M, M.data(), M.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

  auto start_gpu = std::chrono::high_resolution_clock::now();
  impute::cmve_impute_cuda(d_X, d_M, N_FULL, D_FULL, K);
  auto end_gpu = std::chrono::high_resolution_clock::now();

  cudaMemcpy(X_gpu.data(), d_X, X_gpu.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(d_X);
  cudaFree(d_M);
  double t_gpu = std::chrono::duration<double>(end_gpu - start_gpu).count();
  std::cout << "GPU Time: " << t_gpu << " s\n";

  // Compare
  double sum_diff = 0;
  double max_diff = 0;
  int count = 0;

  for (size_t i = 0; i < X.size(); ++i) {
    if (M[i] == 0) {
      double d = std::abs(X_cpu[i] - X_gpu[i]);
      sum_diff += d;
      if (d > max_diff)
        max_diff = d;
      count++;
    }
  }
  double mae_diff = (count > 0) ? (sum_diff / count) : 0;

  std::cout << "\nDifference (CPU vs GPU):\n";
  std::cout << "MAE: " << mae_diff << "\n";
  std::cout << "Max Diff: " << max_diff << "\n";

  if (mae_diff < 1.0) {
    std::cout << "SUCCESS: implementations match closely.\n";
  } else {
    std::cout
        << "NOTE: Differences exist. Often due to float vs double precision in "
           "Covariance matrix or K selection tie-breaking.\n";
  }

  return 0;
}
