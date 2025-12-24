#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>


// Simple function to test the logic used in ExperimentRunner
void test_metrics_logic() {
  // 2x2 matrix
  // Row 0: [1.0, 2.0]
  // Row 1: [3.0, 4.0]
  std::vector<float> ground_truth = {1.0f, 2.0f, 3.0f, 4.0f};

  // Holdout indices (simulating indices into flat array)
  // Let's say we hold out 2.0 (idx 1) and 3.0 (idx 2)
  std::vector<int> holdout_idx = {1, 2};
  std::vector<float> holdout_y = {2.0f, 3.0f};

  // Simulated Preditions (after imputation)
  // Let's say we predicted 2.5 and 2.8
  std::vector<float> X_pred = {1.0f, 2.5f, 2.8f, 4.0f};

  // Logic from ExperimentRunner::evaluate
  double mae_sum = 0;
  double mse_sum = 0;
  int count = 0;

  for (size_t k = 0; k < holdout_idx.size(); ++k) {
    int idx = holdout_idx[k];
    float y_true = holdout_y[k];
    float y_pred = X_pred[idx];

    float diff = std::abs(y_true - y_pred);
    mae_sum += diff;
    mse_sum += diff * diff;
    count++;
  }

  double mae = mae_sum / count;
  double rmse = std::sqrt(mse_sum / count);

  std::cout << "Test Results:\n";
  std::cout << "Count: " << count << " (Expected 2)\n";
  std::cout << "MAE: " << mae
            << " (Expected |2.0-2.5| + |3.0-2.8| / 2 = 0.35)\n";
  std::cout << "RMSE: " << rmse
            << " (Expected sqrt((0.5^2 + 0.2^2)/2) = sqrt(0.145) = 0.38078)\n";

  assert(std::abs(mae - 0.35) < 1e-5);
  assert(std::abs(rmse - 0.380788) < 1e-5);

  std::cout << "\n>>> Metrics Logic Verified <<<\n";
}

int main() {
  test_metrics_logic();
  return 0;
}
