#pragma once

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace impute {

/**
 * @brief Utility to check parity between two imputation results.
 */
class ParityChecker {
public:
  struct Result {
    bool passed;
    float max_diff;
    float avg_diff;
    int fail_count;
  };

  /**
   * @brief Compare two matrices X1 and X2.
   * Only compares values where Mask == 0 (imputed values).
   */
  static Result compare(const float *X1, const float *X2,
                        const std::uint8_t *Mask, int N, int D,
                        float epsilon = 1e-4f) {
    Result res = {true, 0.0f, 0.0f, 0};
    double sum_diff = 0.0;
    int count = 0;

    for (int i = 0; i < N * D; ++i) {
      if (Mask[i] == 0) { // Check only imputed values
        float diff = std::abs(X1[i] - X2[i]);

        if (std::isnan(diff) || std::isinf(diff)) {
          res.passed = false;
          res.max_diff = INFINITY; // Mark as huge
          res.fail_count++;
        } else {
          if (diff > res.max_diff)
            res.max_diff = diff;
          sum_diff += diff;
          count++;

          if (diff > epsilon) {
            res.passed = false;
            res.fail_count++;
          }
        }
      }
    }

    if (count > 0) {
      res.avg_diff = static_cast<float>(sum_diff / count);
    }

    return res;
  }

  static void print_report(const std::string &label, const Result &res) {
    std::cout << "--- Parity Report: " << label << " ---\n";
    std::cout << "Result:   " << (res.passed ? "PASSED" : "FAILED") << "\n";
    std::cout << "Max Diff: " << std::fixed << std::setprecision(6)
              << res.max_diff << "\n";
    std::cout << "Avg Diff: " << res.avg_diff << "\n";
    if (!res.passed) {
      std::cout << "Failures: " << res.fail_count << " elements\n";
    }
    std::cout << "---------------------------\n";
  }
};

} // namespace impute
