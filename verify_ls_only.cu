#include "ImputingLibrary/cuda_bin_io.hpp"
#include "imputation_lib/cpu/ls_impute_cpu.cpp"
#include "imputation_lib/gpu/gpu_wrappers.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


using namespace impute;

int main() {
  int N = 100;
  int D = 200;
  size_t total = (size_t)N * D;

  std::string data_dir = R"(D:\icu_dataset\processed\cuda_eval)";
  auto X_orig =
      read_f32(data_dir + "\\X_miss_float32.bin", total * sizeof(float));
  auto Mask =
      read_u8(data_dir + "\\M_train_uint8.bin", total * sizeof(uint8_t));

  std::cout << "Testing LS..." << std::endl;

  std::vector<float> X_cpu = X_orig;
  LsImputerCpu cpu_ls(5);
  cpu_ls.impute(X_cpu.data(), Mask.data(), N, D);
  std::cout << "CPU done." << std::endl;

  std::vector<float> X_gpu = X_orig;
  LsImputerGpu gpu_ls(5);
  gpu_ls.impute(X_gpu.data(), Mask.data(), N, D);
  std::cout << "GPU done." << std::endl;

  // Compare
  double total_diff = 0, max_diff = 0;
  int count = 0;
  for (size_t i = 0; i < total; i++) {
    if (Mask[i] == 0) {
      double diff = std::abs((double)X_cpu[i] - (double)X_gpu[i]);
      total_diff += diff;
      if (diff > max_diff)
        max_diff = diff;
      count++;
    }
  }
  std::cout << "LS: Avg=" << (total_diff / count) << " Max=" << max_diff
            << std::endl;
  return 0;
}
