#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "ImputingLibrary/experiment_runner.hpp"
#include "imputation_lib/gpu/gpu_wrappers.hpp"

// Include CUDA sources for linking
#include "imputation_lib/gpu/bgs_impute_gpu.cu"
#include "imputation_lib/gpu/common_kernels.cu"
#include "imputation_lib/gpu/rlsp_impute_gpu.cu"

namespace fs = std::filesystem;
using namespace impute;

int main() {
  std::string data_dir = R"(D:\icu_dataset\processed\cuda_eval)";
  std::string eval_dir = R"(D:\icu_dataset\processed\cuda_eval)";
  ExperimentRunner runner(data_dir, eval_dir);

  // Load Evaluation Data
  std::string idx_path = eval_dir + "\\holdout_idx_int32.bin";
  std::string y_path = eval_dir + "\\holdout_y_float32.bin";
  auto indices_raw = read_i32(idx_path, fs::file_size(idx_path));
  auto values = read_f32(y_path, fs::file_size(y_path));
  std::vector<int> indices(indices_raw.begin(), indices_raw.end());
  runner.set_evaluation_data(indices, values);

  // Limit to a smaller subset for speed during tuning
  runner.set_limit(2000);

  // 1. RLSP Tuning (K neighbors vs PC components)
  std::vector<std::shared_ptr<IImputer>> rlsp_variants;
  std::vector<int> K_vals = {5, 10, 20, 30};
  std::vector<int> PC_vals = {2, 5, 10, 15};

  for (int k : K_vals) {
    for (int pc : PC_vals) {
      if (pc < k) {
        rlsp_variants.push_back(std::make_shared<RlspImputerGpu>(k, pc));
      }
    }
  }
  runner.tune(rlsp_variants, "RLSP");

  // 2. BGS Tuning (Max Genes vs Ridge Penalty)
  std::vector<std::shared_ptr<IImputer>> bgs_variants;
  std::vector<int> Gene_vals = {5, 10, 20};
  std::vector<float> Ridge_vals = {1e-2f, 1e-1f, 1.0f};

  for (int g : Gene_vals) {
    for (float r : Ridge_vals) {
      bgs_variants.push_back(std::make_shared<BgsImputerGpu>(g, r));
    }
  }
  runner.tune(bgs_variants, "BGS");

  return 0;
}
