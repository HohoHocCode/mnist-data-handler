#include "ImputingLibrary/cli_parser.hpp"
#include "ImputingLibrary/experiment_runner.hpp"
#include "ImputingLibrary/logger.hpp"
#include "imputation_lib/gpu/gpu_wrappers.hpp"
#include <filesystem>
#include <iostream>
#include <memory>

using namespace impute;

size_t get_file_size(const std::string &path) {
  try {
    return std::filesystem::file_size(path);
  } catch (...) {
    return 0;
  }
}

int main(int argc, char **argv) {
  CLIParser parser(argc, argv);

  if (parser.has("help") || argc < 2) {
    parser.print_help();
    return 0;
  }

  std::string data_dir =
      parser.get("input", R"(D:\icu_dataset\processed\cuda_eval)");
  std::string eval_dir =
      parser.get("eval", R"(D:\icu_dataset\processed\cuda_eval)");

  Logger::info("Initializing Imputing Library Runner...");
  ExperimentRunner runner(data_dir, eval_dir);

  // Load evaluation data
  std::string holdout_idx_path = eval_dir + "\\holdout_idx_int32.bin";
  std::string holdout_y_path = eval_dir + "\\holdout_y_float32.bin";

  size_t idx_bytes = get_file_size(holdout_idx_path);
  size_t y_bytes = get_file_size(holdout_y_path);

  if (idx_bytes > 0 && y_bytes > 0) {
    auto indices = read_i32(holdout_idx_path, idx_bytes);
    auto values = read_f32(holdout_y_path, y_bytes);
    runner.set_evaluation_data(indices, values);
    Logger::info("Evaluation data loaded (" + std::to_string(indices.size()) +
                 " entries).");
  } else {
    Logger::warn("Evaluation data not found or empty at " + eval_dir);
  }

  // Configure Algorithm
  std::string algo_arg = parser.get("algo", "rlsp");
  int k = parser.get_int("k", 5);
  int pc = parser.get_int("pc", 2);
  float ridge = parser.get_float("ridge", 0.1f);
  int genes = parser.get_int("genes", 10);
  int rank = parser.get_int("rank", 10);
  int iter = parser.get_int("iter", 5);
  int chunk_size = parser.get_int("chunk", 2048);

  if (algo_arg == "rlsp") {
    runner.add_algorithm(std::make_shared<RlspImputerGpu>(k, pc));
  } else if (algo_arg == "bgs") {
    runner.add_algorithm(std::make_shared<BgsImputerGpu>(genes, ridge));
  } else if (algo_arg == "svd") {
    runner.add_algorithm(std::make_shared<SvdImputerGpu>(rank));
  } else if (algo_arg == "bpca") {
    runner.add_algorithm(std::make_shared<BpcaImputerGpu>(k));
  } else if (algo_arg == "lls") {
    runner.add_algorithm(std::make_shared<LlsImputerGpu>(k));
  } else if (algo_arg == "ills") {
    runner.add_algorithm(std::make_shared<IllsImputerGpu>(k));
  } else if (algo_arg == "knn") {
    runner.add_algorithm(std::make_shared<KnnImputerGpu>(k));
  } else if (algo_arg == "sknn") {
    runner.add_algorithm(std::make_shared<SknnImputerGpu>(k));
  } else if (algo_arg == "iknn") {
    runner.add_algorithm(std::make_shared<IknnImputerGpu>(k));
  } else if (algo_arg == "ls") {
    runner.add_algorithm(std::make_shared<LsImputerGpu>(k));
  } else if (algo_arg == "slls") {
    runner.add_algorithm(std::make_shared<SllsImputerGpu>(k));
  } else if (algo_arg == "gmc") {
    runner.add_algorithm(std::make_shared<GmcImputerGpu>(k, iter));
  } else if (algo_arg == "cmve") {
    runner.add_algorithm(std::make_shared<CmveImputerGpu>(k));
  } else if (algo_arg == "amvi") {
    runner.add_algorithm(std::make_shared<AmviImputerGpu>(k));
  } else if (algo_arg == "arls") {
    runner.add_algorithm(std::make_shared<ArlsImputerGpu>(k));
  } else if (algo_arg == "lincmb") {
    // k is used for KNN, we'll use a fixed rank or similar for now or add a
    // parameter
    runner.add_algorithm(std::make_shared<LinCmbImputerGpu>(k, rank));
  } else {
    Logger::error("Unknown algorithm: " + algo_arg);
    return 1;
  }

  runner.run_streaming("benchmark_results.csv", chunk_size);
  Logger::info("Benchmark Complete. Results saved to benchmark_results.csv");

  return 0;
}
