#include "ImputingLibrary/cuda_bin_io.hpp"
#include "imputation_lib/cpu/iknn_impute_cpu.cpp"
#include "imputation_lib/cpu/knn_impute_cpu.cpp"
#include "imputation_lib/cpu/sknn_impute_cpu.cpp"
#include "imputation_lib/gpu/gpu_wrappers.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace impute;

void compare_results(const std::string &name, const std::vector<float> &cpu_res,
                     const std::vector<float> &gpu_res,
                     const std::vector<uint8_t> &mask, int N, int D) {
  double total_diff = 0;
  double max_diff = 0;
  int count = 0;
  int max_idx = -1;

  for (int i = 0; i < N * D; i++) {
    if (mask[i] == 0) {
      double diff = std::abs((double)cpu_res[i] - (double)gpu_res[i]);
      total_diff += diff;
      if (diff > max_diff) {
        max_diff = diff;
        max_idx = i;
      }
      count++;
    }
  }

  std::cout << "--- Parity Check: " << name << " ---" << std::endl;
  if (count > 0) {
    std::cout << "Average Difference (Missing Only): " << std::scientific
              << (total_diff / count) << std::endl;
    std::cout << "Max Difference: " << max_diff << " at index " << max_idx
              << std::endl;
    if (max_idx != -1) {
      std::cout << "  CPU value: " << cpu_res[max_idx]
                << ", GPU value: " << gpu_res[max_idx] << std::endl;
      int r = max_idx / D;
      int c = max_idx % D;
      std::cout << "  Row: " << r << ", Col: " << c << std::endl;
    }
  } else {
    std::cout << "No missing values to check." << std::endl;
  }
}

int main() {
  int N = 100;
  int D = 200;
  size_t total = (size_t)N * D;

  std::string data_dir = R"(D:\icu_dataset\processed\cuda_eval)";
  auto X_orig =
      read_f32(data_dir + "\\X_miss_float32.bin", total * sizeof(float));
  auto Mask =
      read_u8(data_dir + "\\M_train_uint8.bin", total * sizeof(uint8_t));

  if (X_orig.size() < total || Mask.size() < total) {
    std::cerr << "Failed to load dataset subset." << std::endl;
    return 1;
  }

  // 0. KNN Check
  {
    std::vector<float> X_cpu = X_orig;
    KnnImputerCpu cpu_knn(5);
    cpu_knn.impute(X_cpu.data(), Mask.data(), N, D);

    std::vector<float> X_gpu = X_orig;
    KnnImputerGpu gpu_knn(5);
    gpu_knn.impute(X_gpu.data(), Mask.data(), N, D);

    compare_results("KNN (K=5)", X_cpu, X_gpu, Mask, N, D);
  }

  // 1. SKNN Check
  {
    std::vector<float> X_cpu = X_orig;
    SknnImputerCpu cpu_sknn(5);
    cpu_sknn.impute(X_cpu.data(), Mask.data(), N, D);

    std::vector<float> X_gpu = X_orig;
    SknnImputerGpu gpu_sknn(5);
    gpu_sknn.impute(X_gpu.data(), Mask.data(), N, D);

    compare_results("SKNN (K=5)", X_cpu, X_gpu, Mask, N, D);
  }

  // 2. IKNN Check
  {
    std::vector<float> X_cpu = X_orig;
    IkNNImputerCpu cpu_iknn(5, 5);
    cpu_iknn.impute(X_cpu.data(), Mask.data(), N, D);

    std::vector<float> X_gpu = X_orig;
    IknnImputerGpu gpu_iknn(5, 5);
    gpu_iknn.impute(X_gpu.data(), Mask.data(), N, D);

    compare_results("IKNN (K=5, Iter=5)", X_cpu, X_gpu, Mask, N, D);
  }

  return 0;
}
