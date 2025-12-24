#include "experiment_runner.hpp"

// Include refactored CPU implementations
#include "../imputation_lib/cpu/bpca_impute_cpu.cpp"
#include "../imputation_lib/cpu/iknn_impute_cpu.cpp"
#include "../imputation_lib/cpu/knn_impute_cpu.cpp"
#include "../imputation_lib/cpu/lls_impute_cpu.cpp"
#include "../imputation_lib/cpu/ls_impute_cpu.cpp"
#include "../imputation_lib/cpu/slls_impute_cpu.cpp"
#include "../imputation_lib/cpu/svd_impute_cpu.cpp"


#include <filesystem>
#include <iostream>

using namespace impute;
namespace fs = std::filesystem;

int main() {
  std::string data_dir = R"(D:\icu_dataset\processed\cuda_eval)";
  std::string eval_dir = R"(D:\icu_dataset\processed\cuda_eval)";

  std::cout << "Initializing Experiment Runner...\n";
  ExperimentRunner runner(data_dir, eval_dir);

  if (fs::exists(eval_dir)) {
    std::string idx_path = eval_dir + "\\holdout_idx_int32.bin";
    std::string y_path = eval_dir + "\\holdout_y_float32.bin";

    if (fs::exists(idx_path) && fs::exists(y_path)) {
      try {
        size_t idx_bytes = fs::file_size(idx_path);
        size_t y_bytes = fs::file_size(y_path);

        std::cout << "Loading Eval: Idx bytes=" << idx_bytes
                  << " Y bytes=" << y_bytes << "\n";

        auto indices_raw = read_i32(idx_path, idx_bytes);
        auto values = read_f32(y_path, y_bytes);

        std::vector<int> indices;
        indices.reserve(indices_raw.size());
        for (auto i : indices_raw)
          indices.push_back(static_cast<int>(i));

        runner.set_evaluation_data(indices, values);
        std::cout << "Evaluation data loaded: " << indices.size()
                  << " samples.\n";
      } catch (const std::exception &e) {
        std::cerr << "Error loading evaluation data: " << e.what() << "\n";
      }
    }
  }

  // Register Algorithms using new names
  runner.add_algorithm(std::make_shared<KnnImputerCpu>(10));
  runner.add_algorithm(std::make_shared<LlsImputerCpu>(10));  // Strict LLS
  runner.add_algorithm(std::make_shared<LsImputerCpu>(10));   // Strict LS
  runner.add_algorithm(std::make_shared<SllsImputerCpu>(10)); // Strict SLLS
  runner.add_algorithm(std::make_shared<BpcaImputerCpu>());
  runner.add_algorithm(std::make_shared<IkNNImputerCpu>(10, 5));
  runner.add_algorithm(std::make_shared<SvdImputerCpu>(10, 10));

  std::cout << "Starting Benchmarks (1000 rows for realistic metrics)...\n";
  runner.set_limit(1000);
  runner.run("cpu_results.csv");
  std::cout << "Analysis Complete. Results saved to cpu_results.csv.\n";

  return 0;
}
