#include "imputation_lib/cpu_impute/cpu_bpca.cpp"
// Unnecessary includes removed
#include <iostream>
#include <vector>

using namespace impute;

int main() {
  std::cout << "Testing CPU BPCA Impute..." << std::endl;
  const int N = 3, D = 3;
  // Data matrix with missing entries (set to 0.0)
  float X_orig[N * D] = {
      1.0f, 0.0f, 3.0f,                  // missing at (0,1)
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 0.0f // missing at (2,2)
  };
  std::uint8_t Mask[N * D] = {1, 0, 1, 1, 1, 1, 1, 1, 0};
  float X_test[N * D];
  std::copy(std::begin(X_orig), std::end(X_orig), std::begin(X_test));

  CpuBpcaImputer imputer;
  imputer.impute(X_test, Mask, N, D);

  std::cout << "Imputed [0,1]: " << X_test[0 * D + 1] << std::endl;
  std::cout << "Imputed [2,2]: " << X_test[2 * D + 2] << std::endl;
  if (X_test[0 * D + 1] != 0.0f && X_test[2 * D + 2] != 0.0f) {
    std::cout << "SUCCESS: BPCA imputed values." << std::endl;
  } else {
    std::cout << "FAILURE: BPCA did not impute." << std::endl;
  }
  return 0;
}
