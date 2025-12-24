#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace impute {
/**
 * @brief Helper to solve a linear system Ax = b using Gaussian elimination.
 *        A is assumed to be a square matrix of size n.
 */
static std::vector<double> solve_linear_system_slls(std::vector<double> A,
                                                    std::vector<double> b) {
  const size_t n = b.size();
  // Augment A with b
  for (size_t i = 0; i < n; ++i) {
    A.insert(A.begin() + i * (n + 1) + n, b[i]);
  }
  // Forward elimination
  for (size_t i = 0; i < n; ++i) {
    // Pivot
    size_t pivot = i;
    double maxVal = std::abs(A[i * (n + 1) + i]);
    for (size_t row = i + 1; row < n; ++row) {
      double val = std::abs(A[row * (n + 1) + i]);
      if (val > maxVal) {
        maxVal = val;
        pivot = row;
      }
    }
    if (std::abs(A[pivot * (n + 1) + i]) < 1e-12) {
      throw std::runtime_error("Singular matrix in linear solver");
    }
    if (pivot != i) {
      for (size_t col = i; col <= n; ++col) {
        std::swap(A[i * (n + 1) + col], A[pivot * (n + 1) + col]);
      }
    }
    double diag = A[i * (n + 1) + i];
    for (size_t row = i + 1; row < n; ++row) {
      double factor = A[row * (n + 1) + i] / diag;
      for (size_t col = i; col <= n; ++col) {
        A[row * (n + 1) + col] -= factor * A[i * (n + 1) + col];
      }
    }
  }
  // Back substitution
  std::vector<double> x(n);
  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    double sum = A[i * (n + 1) + n];
    for (size_t j = i + 1; j < n; ++j) {
      sum -= A[i * (n + 1) + j] * x[j];
    }
    x[i] = sum / A[i * (n + 1) + i];
  }
  return x;
}

/**
 * @brief Sequential LLS (SLLS) imputer.
 */
class SLLSImputer : public IImputer {
public:
  std::string name() const override { return "SLLSImpute"; }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    std::vector<float> X_cur(X, X + N * D);
    for (int col = 0; col < D; ++col) {
      std::vector<std::vector<double>> X_train;
      std::vector<double> y_train;
      for (int i = 0; i < N; ++i) {
        if (Mask[i * D + col] == 1) {
          std::vector<double> row;
          row.push_back(1.0); // intercept
          for (int k = 0; k < col; ++k) {
            row.push_back(static_cast<double>(X_cur[i * D + k]));
          }
          X_train.push_back(std::move(row));
          y_train.push_back(static_cast<double>(X_cur[i * D + col]));
        }
      }
      std::vector<double> coeffs;
      if (!X_train.empty()) {
        size_t P = col + 1;
        std::vector<double> XtX(P * P, 0.0);
        std::vector<double> Xty(P, 0.0);
        for (size_t r = 0; r < X_train.size(); ++r) {
          const auto &row = X_train[r];
          for (size_t a = 0; a < P; ++a) {
            Xty[a] += row[a] * y_train[r];
            for (size_t b = 0; b < P; ++b) {
              XtX[a * P + b] += row[a] * row[b];
            }
          }
        }
        try {
          coeffs = solve_linear_system_slls(XtX, Xty);
        } catch (...) {
          coeffs.clear();
        }
      }
      for (int i = 0; i < N; ++i) {
        if (Mask[i * D + col] == 0) { // missing
          if (coeffs.empty()) {
            double sum = 0.0;
            int cnt = 0;
            for (int r = 0; r < N; ++r) {
              if (Mask[r * D + col] == 1) {
                sum += X_cur[r * D + col];
                ++cnt;
              }
            }
            X_cur[i * D + col] =
                (cnt > 0) ? static_cast<float>(sum / cnt) : 0.0f;
          } else {
            double pred = coeffs[0]; // intercept
            size_t idx = 1;
            for (int k = 0; k < col; ++k) {
              pred += coeffs[idx++] * static_cast<double>(X_cur[i * D + k]);
            }
            X_cur[i * D + col] = static_cast<float>(pred);
          }
        }
      }
    }
    std::copy(X_cur.begin(), X_cur.end(), X);
  }
};

} // namespace impute
