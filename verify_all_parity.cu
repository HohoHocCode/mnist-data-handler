#include "ImputingLibrary/cuda_bin_io.hpp"
#include "imputation_lib/cpu/amvi_impute_cpu.cpp"
#include "imputation_lib/cpu/arls_impute_cpu.cpp"
#include "imputation_lib/cpu/cmve_impute_cpu.cpp"
#include "imputation_lib/cpu/knn_impute_cpu.cpp"
#include "imputation_lib/cpu/ls_impute_cpu.cpp"
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
    else
      std::cout << "FAIL |" << std::endl;
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
    std::cerr << "Failed to load dataset." << std::endl;
    return 1;
  }

  std::cout << "| Algorithm | Avg Diff | Max Diff | Status |" << std::endl;
  std::cout << "| :--- | :--- | :--- | :--- |" << std::endl;

  // 1. KNN
  try {
    std::vector<float> X_cpu = X_orig;
    KnnImputerCpu cpu_knn(5);
    cpu_knn.impute(X_cpu.data(), Mask.data(), N, D);

    std::vector<float> X_gpu = X_orig;
    KnnImputerGpu gpu_knn(5);
    gpu_knn.impute(X_gpu.data(), Mask.data(), N, D);

    compare_results("KNN (K=5)", X_cpu, X_gpu, Mask, N, D);
  } catch (std::exception &e) {
    std::cout << "| **KNN** | ERROR | " << e.what() << " | CRASH |"
              << std::endl;
  }

  // 2. LS
  try {
    std::vector<float> X_cpu = X_orig;
    LsImputerCpu cpu_ls(5);
    cpu_ls.impute(X_cpu.data(), Mask.data(), N, D);

    std::vector<float> X_gpu = X_orig;
    LsImputerGpu gpu_ls(5);
    gpu_ls.impute(X_gpu.data(), Mask.data(), N, D);

    compare_results("LS (K=5)", X_cpu, X_gpu, Mask, N, D);
  } catch (std::exception &e) {
    std::cout << "| **LS** | ERROR | " << e.what() << " | CRASH |" << std::endl;
  }

  // 3. CMVE
  try {
    std::vector<float> X_cpu = X_orig;
    CmveImputerCpu cpu_cmve(5);
    cpu_cmve.impute(X_cpu.data(), Mask.data(), N, D);

    std::vector<float> X_gpu = X_orig;
    CmveImputerGpu gpu_cmve(5);
    gpu_cmve.impute(X_gpu.data(), Mask.data(), N, D);

    compare_results("CMVE (K=5)", X_cpu, X_gpu, Mask, N, D);
  } catch (std::exception &e) {
    std::cout << "| **CMVE** | ERROR | " << e.what() << " | CRASH |"
              << std::endl;
  }

  // 4. AMVI
  try {
    std::vector<float> X_cpu = X_orig;
    AmviImputerCpu cpu_amvi(10);
    cpu_amvi.impute(X_cpu.data(), Mask.data(), N, D);

    std::vector<float> X_gpu = X_orig;
    AmviImputerGpu gpu_amvi(10);
    gpu_amvi.impute(X_gpu.data(), Mask.data(), N, D);

    compare_results("AMVI (K=10)", X_cpu, X_gpu, Mask, N, D);
  } catch (std::exception &e) {
    std::cout << "| **AMVI** | ERROR | " << e.what() << " | CRASH |"
              << std::endl;
  }

  // 5. ARLS
  try {
    std::vector<float> X_cpu = X_orig;
    ArlsImputerCpu cpu_arls(5);
    cpu_arls.impute(X_cpu.data(), Mask.data(), N, D);

    std::vector<float> X_gpu = X_orig;
    ArlsImputerGpu gpu_arls(5);
    gpu_arls.impute(X_gpu.data(), Mask.data(), N, D);

    compare_results("ARLS (K=5)", X_cpu, X_gpu, Mask, N, D);
  } catch (std::exception &e) {
    std::cout << "| **ARLS** | ERROR | " << e.what() << " | CRASH |"
              << std::endl;
  }

  std::cout << std::endl << "Verification complete." << std::endl;
  return 0;
}
