#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>


int main() {
  // 1. Calculate column means from the training data X_miss.bin
  std::ifstream x_file(R"(D:\icu_dataset\processed\cuda_eval\X_miss.bin)",
                       std::ios::binary);
  std::ifstream mask_file(R"(D:\icu_dataset\processed\cuda_eval\Mask.bin)",
                          std::ios::binary);

  int N = 11756;
  int D = 200;
  std::vector<float> X(N * D);
  std::vector<uint8_t> M(N * D);
  x_file.read((char *)X.data(), N * D * sizeof(float));
  mask_file.read((char *)M.data(), N * D * sizeof(uint8_t));

  std::vector<double> col_sums(D, 0);
  std::vector<int> col_counts(D, 0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      if (M[i * D + j] == 1) { // 1 = Observed in my system
        col_sums[j] += X[i * D + j];
        col_counts[j]++;
      }
    }
  }
  std::vector<float> means(D, 0.0f);
  for (int j = 0; j < D; ++j) {
    if (col_counts[j] > 0)
      means[j] = (float)(col_sums[j] / col_counts[j]);
  }

  // 2. Evaluate Mean Fill on holdout data
  std::ifstream idx_file(
      R"(D:\icu_dataset\processed\cuda_eval\holdout_idx_int32.bin)",
      std::ios::binary);
  std::ifstream val_file(
      R"(D:\icu_dataset\processed\cuda_eval\holdout_y_float32.bin)",
      std::ios::binary);

  idx_file.seekg(0, std::ios::end);
  size_t n_holdout = idx_file.tellg() / sizeof(int);
  idx_file.seekg(0, std::ios::beg);

  std::vector<int> indices(n_holdout);
  std::vector<float> truths(n_holdout);
  idx_file.read((char *)indices.data(), n_holdout * sizeof(int));
  val_file.read((char *)truths.data(), n_holdout * sizeof(float));

  double sum_abs = 0;
  double sum_sq = 0;
  for (size_t k = 0; k < n_holdout; ++k) {
    int col = indices[k] % D;
    float pred = means[col];
    float truth = truths[k];
    float diff = std::abs(pred - truth);
    sum_abs += diff;
    sum_sq += (double)diff * diff;
  }

  std::cout << "--- Mean Fill Baseline ---\n";
  std::cout << "MAE: " << (sum_abs / n_holdout) << "\n";
  std::cout << "RMSE: " << std::sqrt(sum_sq / n_holdout) << "\n";

  return 0;
}
