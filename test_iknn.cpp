#include "imputation_lib/cpu_impute/cpu_iknn.cpp"
#include "imputation_lib/i_imputer.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>


using namespace impute;

int main() {
  std::cout << "Testing CPU IKNN Impute..." << std::endl;
  const int N = 5, D = 4;
  // Simple dataset with a missing entry at (0,3)
  float X_orig[N * D] = {1.0f, 2.0f, 3.0f, 0.0f, 2.1f, 1.9f, 3.2f,
                         5.0f, 0.9f, 2.2f, 2.8f, 1.2f, 1.1f, 2.1f,
                         3.1f, 0.0f, 5.0f, 1.0f, 0.5f, 2.2f};
  std::uint8_t Mask[N * D] = {1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 0, 1, 1, 1, 1};
  float X_test[N * D];
  std::copy(std::begin(X_orig), std::end(X_orig), std::begin(X_test));

  IkNNImputer imputer(2, 3); // K=2, 3 iterations
  imputer.impute(X_test, Mask, N, D);

  std::cout << "Imputed [0,3]: " << X_test[0 * D + 3] << std::endl;
  std::cout << "Imputed [3,3]: " << X_test[3 * D + 3] << std::endl;
  if (X_test[0 * D + 3] != 0.0f && X_test[3 * D + 3] != 0.0f) {
    std::cout << "SUCCESS: IKNN imputed values." << std::endl;
  } else {
    std::cout << "FAILURE: IKNN did not impute." << std::endl;
  }
  return 0;
}
