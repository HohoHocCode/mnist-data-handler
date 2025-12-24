#include "imputation_lib/cpu_impute/cpu_lls.cpp"
#include "imputation_lib/i_imputer.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>


using namespace impute;

int main() {
  std::cout << "Testing CPU LLS Impute..." << std::endl;
  const int N = 4, D = 3;
  // Data matrix with a missing entry (set to 0.0)
  float X_orig[N * D] = {1.0f, 2.0f, 0.0f, // row 0, col 2 missing
                         2.0f, 3.0f, 4.0f, 3.0f, 5.0f, 6.0f, 4.0f, 6.0f, 8.0f};
  std::uint8_t Mask[N * D] = {1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float X_test[N * D];
  std::copy(std::begin(X_orig), std::end(X_orig), std::begin(X_test));

  LLSImputer imputer;
  imputer.impute(X_test, Mask, N, D);

  std::cout << "Imputed [0,2]: " << X_test[0 * D + 2] << std::endl;
  // Simple check: value should be close to linear prediction based on other
  // rows.
  if (X_test[0 * D + 2] != 0.0f) {
    std::cout << "SUCCESS: LLS imputed value." << std::endl;
  } else {
    std::cout << "FAILURE: LLS did not impute." << std::endl;
  }
  return 0;
}
