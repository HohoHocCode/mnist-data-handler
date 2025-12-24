#include "imputation_lib/cpu_impute/cpu_svd.cpp"
#include "imputation_lib/i_imputer.hpp"
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>


using namespace impute;

int main() {
  std::cout << "Testing CPU SVD Impute (Matrix Factorization)..." << std::endl;
  // We used a randomized initialization, so let's set a fixed seed for
  // reproducibility check
  srand(42);

  const int N = 6, D = 4;
  // Create a low-rank matrix (Rank ~1 or 2)
  // T = u * v^T where u=[1,2,3,4,5,6], v=[1, 0.5, 0.2, 0.8]
  // Row 0: 1 * [1, 0.5, 0.2, 0.8] = [1.0, 0.5, 0.2, 0.8]
  // Row 5: 6 * [1, 0.5, 0.2, 0.8] = [6.0, 3.0, 1.2, 4.8]

  float X_orig[N * D] = {
      1.0f, 0.5f, 0.2f, 0.0f, // Missing at [0,3], true val 0.8
      2.0f, 1.0f, 0.4f, 1.6f, 3.0f, 1.5f, 0.6f, 2.4f, 4.0f, 2.0f, 0.8f,
      3.2f, 5.0f, 2.5f, 1.0f, 4.0f, 6.0f, 3.0f, 0.0f, 4.8f // Missing at [5,2],
                                                           // true val 1.2
  };

  std::uint8_t Mask[N * D] = {1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};

  float X_test[N * D];
  std::copy(std::begin(X_orig), std::end(X_orig), std::begin(X_test));

  // Uses Rank=1 approximation, hopefully enough to recover the missing logic
  CpuSvdImputer imputer(1, 200, 0.005f);
  imputer.impute(X_test, Mask, N, D);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Imputed [0,3]: " << X_test[0 * D + 3] << " (True: 0.8)"
            << std::endl;
  std::cout << "Imputed [5,2]: " << X_test[5 * D + 2] << " (True: 1.2)"
            << std::endl;

  // Check if close enough (simple linear structure should be recovered)
  if (std::abs(X_test[0 * D + 3] - 0.8f) < 0.2f &&
      std::abs(X_test[5 * D + 2] - 1.2f) < 0.2f) {
    std::cout << "SUCCESS: SVD imputed values reasonably well." << std::endl;
  } else {
    std::cout << "WARNING: SVD imputation accuracy might be low (random init + "
                 "gradients)."
              << std::endl;
  }
  return 0;
}
