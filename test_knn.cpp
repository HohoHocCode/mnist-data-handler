#include "imputation_lib/cpu_impute/cpu_knn.cpp"
#include "imputation_lib/i_imputer.hpp"
#include <iomanip>
#include <iostream>
#include <vector>


using namespace impute;

int main() {
  std::cout << "Testing CPU KNN Impute..." << std::endl;
  const int N = 5, D = 4;
  float X_orig[N * D] = {1.0, 2.0, 3.0, 0.0, 2.1, 1.9, 3.2, 5.0, 0.9, 2.2,
                         2.8, 1.2, 1.1, 2.1, 3.1, 0.0, 5.0, 1.0, 0.5, 2.2};
  std::uint8_t Mask[N * D] = {1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 0, 1, 1, 1, 1};

  CpuKnnImputer imputer(2);
  float X_test[N * D];
  std::copy(std::begin(X_orig), std::end(X_orig), std::begin(X_test));

  imputer.impute(X_test, Mask, N, D);

  std::cout << "Imputed [0,3]: " << X_test[0 * D + 3] << "\n";
  std::cout << "Imputed [3,3]: " << X_test[3 * D + 3] << "\n";

  if (X_test[3] != 0.0f && X_test[15] != 0.0f) {
    std::cout << "SUCCESS: KNN Imputed results.\n";
  } else {
    std::cout << "FAILURE: KNN did not impute.\n";
  }
  return 0;
}
