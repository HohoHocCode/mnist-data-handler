#include "ImputingLibrary/cuda_bin_io.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// CPU Implementations
#include "imputation_lib/cpu/bgs_impute_cpu.cpp"
#include "imputation_lib/cpu/cmve_impute_cpu.cpp"
#include "imputation_lib/cpu/iknn_impute_cpu.cpp"
#include "imputation_lib/cpu/ills_impute_cpu.cpp"
#include "imputation_lib/cpu/knn_impute_cpu.cpp"
#include "imputation_lib/cpu/lincmb_impute_cpu.cpp"
#include "imputation_lib/cpu/lls_impute_cpu.cpp"
#include "imputation_lib/cpu/rlsp_impute_cpu.cpp"
#include "imputation_lib/cpu/sknn_impute_cpu.cpp"
#include "imputation_lib/cpu/slls_impute_cpu.cpp"

#include "imputation_lib/cpu/amvi_impute_cpu.cpp"
#include "imputation_lib/cpu/arls_impute_cpu.cpp"
#include "imputation_lib/cpu/gmc_impute_cpu.cpp"
#include "imputation_lib/cpu/ls_impute_cpu.cpp"

// GPU Implementations
#include "imputation_lib/gpu/gpu_wrappers.hpp"

using namespace impute;

void compare_results(const std::string &name, const std::vector<float> &cpu_res,
                     const std::vector<float> &gpu_res,
                     const std::vector<uint8_t> &mask, int N, int D) {
  double total_diff = 0;
  double max_diff = 0;
  int count = 0;

  for (int i = 0; i < N * D; i++) {
    if (mask[i] == 0) {
      double diff = std::abs((double)cpu_res[i] - (double)gpu_res[i]);
      total_diff += diff;
      if (diff > max_diff)
        max_diff = diff;
      count++;
    }
  }

  std::cout << "| **" << name << "** | ";
  if (count > 0) {
    std::cout << std::scientific << std::setprecision(2) << (total_diff / count)
              << " | " << max_diff << " | ";
    if (total_diff / count < 1.0)
      std::cout << "PASS |" << std::endl;
    else {
      std::cout << "FAIL |" << std::endl;
      int printed = 0;
      for (int i = 0; i < N * D && printed < 5; i++) {
        if (mask[i] == 0) {
          double diff = std::abs((double)cpu_res[i] - (double)gpu_res[i]);
          if (diff > 1e-3) {
            std::cout << "  Diff at [" << i / D << "," << i % D
                      << "]: CPU=" << cpu_res[i] << ", GPU=" << gpu_res[i]
                      << " (diff=" << diff << ")" << std::endl;
            printed++;
          }
        }
      }
    }
  } else {
    std::cout << "N/A | N/A | NO_MISSING |" << std::endl;
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

  std::cout << "| Algorithm | Avg Diff | Max Diff | Status |" << std::endl;
  std::cout << "| :--- | :--- | :--- | :--- |" << std::endl;

  // 1. GMC
  std::cout << "Testing GMC..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    GmcImputerCpu cpu(5, 5);
    cpu.impute(X_cpu.data(), Mask.data(), N, D);
    std::vector<float> X_gpu = X_orig;
    GmcImputerGpu gpu(5, 5);
    gpu.impute(X_gpu.data(), Mask.data(), N, D);
    compare_results("GMC", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **GMC** | - | - | CRASH |" << std::endl;
  }

  // 2. AMVI
  std::cout << "Testing AMVI..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    AmviImputerCpu cpu(20);
    std::cout << "  Calling AMVI CPU..." << std::endl;
    cpu.impute(X_cpu.data(), Mask.data(), N, D);

    std::vector<float> X_gpu = X_orig;
    AmviImputerGpu gpu(20);
    std::cout << "  Calling AMVI GPU..." << std::endl;
    gpu.impute(X_gpu.data(), Mask.data(), N, D);

    compare_results("AMVI", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **AMVI** | - | - | CRASH |" << std::endl;
  }

  // 3. ARLS
  std::cout << "Testing ARLS..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    ArlsImputerCpu cpu(10, 0.1f);
    cpu.impute(X_cpu.data(), Mask.data(), N, D);
    std::vector<float> X_gpu = X_orig;
    ArlsImputerGpu gpu(10, 0.1f);
    gpu.impute(X_gpu.data(), Mask.data(), N, D);
    compare_results("ARLS", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **ARLS** | - | - | CRASH |" << std::endl;
  }

  // 4. KNN (Opt)
  std::cout << "Testing KNN..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    KnnImputerCpu cpu(5);
    cpu.impute(X_cpu.data(), Mask.data(), N, D);
    std::vector<float> X_gpu = X_orig;
    KnnImputerGpu gpu(5);
    gpu.impute(X_gpu.data(), Mask.data(), N, D);
    compare_results("KNN (Opt)", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **KNN** | - | - | CRASH |" << std::endl;
  }

  // 5. ILLS
  std::cout << "Testing ILLS..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    IllsImputerCpu cpu(5, 5);
    cpu.impute(X_cpu.data(), Mask.data(), N, D);
    std::vector<float> X_gpu = X_orig;
    IllsImputerGpu gpu(5, 5);
    gpu.impute(X_gpu.data(), Mask.data(), N, D);
    compare_results("ILLS", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **ILLS** | - | - | CRASH |" << std::endl;
  }

  // 6. LS
  std::cout << "Testing LS..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    LsImputerCpu cpu(10);
    cpu.impute(X_cpu.data(), Mask.data(), N, D);
    std::vector<float> X_gpu = X_orig;
    LsImputerGpu gpu(10);
    gpu.impute(X_gpu.data(), Mask.data(), N, D);
    compare_results("LS", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **LS** | - | - | CRASH |" << std::endl;
  }

  // 7. BGS
  std::cout << "Testing BGS..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    BgsImputerCpu cpu(10, 0.1f);
    cpu.impute(X_cpu.data(), Mask.data(), N, D);
    std::vector<float> X_gpu = X_orig;
    BgsImputerGpu gpu(10, 0.1f);
    gpu.impute(X_gpu.data(), Mask.data(), N, D);
    compare_results("BGS", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **BGS** | - | - | CRASH |" << std::endl;
  }

  // 8. IKNN
  std::cout << "Testing IKNN..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    IkNNImputerCpu cpu(10, 5);
    cpu.impute(X_cpu.data(), Mask.data(), N, D);
    std::vector<float> X_gpu = X_orig;
    IknnImputerGpu gpu(10, 5);
    gpu.impute(X_gpu.data(), Mask.data(), N, D);
    compare_results("IKNN", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **IKNN** | - | - | CRASH |" << std::endl;
  }

  // 9. SKNN
  std::cout << "Testing SKNN..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    SknnImputerCpu cpu(10);
    cpu.impute(X_cpu.data(), Mask.data(), N, D);
    std::vector<float> X_gpu = X_orig;
    SknnImputerGpu gpu(10);
    gpu.impute(X_gpu.data(), Mask.data(), N, D);
    compare_results("SKNN", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **SKNN** | - | - | CRASH |" << std::endl;
  }

  // 10. LLS
  std::cout << "Testing LLS..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    LlsImputerCpu cpu(10);
    cpu.impute(X_cpu.data(), Mask.data(), N, D);
    std::vector<float> X_gpu = X_orig;
    LlsImputerGpu gpu(10);
    gpu.impute(X_gpu.data(), Mask.data(), N, D);
    compare_results("LLS", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **LLS** | - | - | CRASH |" << std::endl;
  }

  // 11. SLLS
  std::cout << "Testing SLLS..." << std::endl;
  try {
    std::vector<float> X_cpu = X_orig;
    SllsImputerCpu cpu(10);
    cpu.impute(X_cpu.data(), Mask.data(), N, D);
    std::vector<float> X_gpu = X_orig;
    SllsImputerGpu gpu(10);
    gpu.impute(X_gpu.data(), Mask.data(), N, D);
    compare_results("SLLS", X_cpu, X_gpu, Mask, N, D);
  } catch (...) {
    std::cout << "| **SLLS** | - | - | CRASH |" << std::endl;
  }

  std::cout << std::endl << "Suite Verification Complete." << std::endl;
  return 0;
}
