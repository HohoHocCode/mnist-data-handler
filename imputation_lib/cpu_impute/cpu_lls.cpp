#include "../i_imputer.hpp"
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
static std::vector<double> solve_linear_system_lls(std::vector<double> A,
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
      // swap rows
      for (size_t col = i; col <= n; ++col) {
        std::swap(A[i * (n + 1) + col], A[pivot * (n + 1) + col]);
      }
    }
    // Eliminate below
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
 * @brief CPU implementation of LLSImpute (multiple linear regression).
 */
class LLSImputer : public IImputer {
public:
  std::string name() const override { return "LLSImpute"; }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    std::vector<std::vector<double>> coeffs(D);
    for (int col = 0; col < D; ++col) {
      std::vector<std::vector<double>> X_train;
      std::vector<double> y_train;
      for (int i = 0; i < N; ++i) {
        if (Mask[i * D + col] == 1) { // observed
          std::vector<double> row;
          row.push_back(1.0); // intercept term
          for (int k = 0; k < D; ++k) {
            if (k == col)
              continue;
            row.push_back(static_cast<double>(X[i * D + k]));
          }
          X_train.push_back(std::move(row));
          y_train.push_back(static_cast<double>(X[i * D + col]));
        }
      }
      if (X_train.empty())
        continue;

      size_t P = D;
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
        coeffs[col] = solve_linear_system_lls(XtX, Xty);
      } catch (...) {
        coeffs[col].clear();
      }
    }
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 0) { // missing
          const auto &beta = coeffs[j];
          if (beta.empty()) {
            double sum = 0.0;
            int cnt = 0;
            for (int r = 0; r < N; ++r) {
              if (Mask[r * D + j] == 1) {
                sum += X[r * D + j];
                ++cnt;
              }
            }
            X[i * D + j] = (cnt > 0) ? static_cast<float>(sum / cnt) : 0.0f;
          } else {
            double pred = beta[0]; // intercept
            size_t idx = 1;
            for (int k = 0; k < D; ++k) {
              if (k == j)
                continue;
              pred += beta[idx++] * static_cast<double>(X[i * D + k]);
            }
            X[i * D + j] = static_cast<float>(pred);
          }
        }
      }
    }
  }
};

} // namespace impute
