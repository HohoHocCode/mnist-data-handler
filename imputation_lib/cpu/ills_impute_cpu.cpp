#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace impute {

// Helper: Solve (A^T A + lambda*I) x = A^T b
static std::vector<double>
solve_ols_ridge_ills(const std::vector<std::vector<double>> &A,
                     const std::vector<double> &b, double ridge_alpha = 1e-4) {
  if (A.empty())
    return {};
  size_t n = A.size();
  size_t k = A[0].size();

  std::vector<double> AtA(k * k, 0.0);
  std::vector<double> Atb(k, 0.0);

  for (size_t r = 0; r < n; ++r) {
    for (size_t i = 0; i < k; ++i) {
      Atb[i] += A[r][i] * b[r];
      for (size_t j = 0; j < k; ++j) {
        AtA[i * k + j] += A[r][i] * A[r][j];
      }
    }
  }

  for (size_t i = 0; i < k; ++i)
    AtA[i * k + i] += ridge_alpha;

  std::vector<double> Mat = AtA;
  std::vector<double> &rhs = Atb;
  std::vector<double> x(k);

  for (size_t i = 0; i < k; ++i) {
    size_t pivot = i;
    double max_val = std::abs(Mat[i * k + i]);
    for (size_t r = i + 1; r < k; ++r) {
      if (std::abs(Mat[r * k + i]) > max_val) {
        max_val = std::abs(Mat[r * k + i]);
        pivot = r;
      }
    }

    if (pivot != i) {
      for (size_t c = i; c < k; ++c)
        std::swap(Mat[i * k + c], Mat[pivot * k + c]);
      std::swap(rhs[i], rhs[pivot]);
    }

    if (std::abs(Mat[i * k + i]) < 1e-12)
      return std::vector<double>(k, 0.0);

    for (size_t r = i + 1; r < k; ++r) {
      double factor = Mat[r * k + i] / Mat[i * k + i];
      for (size_t c = i; c < k; ++c)
        Mat[r * k + c] -= factor * Mat[i * k + c];
      rhs[r] -= factor * rhs[i];
    }
  }

  for (int i = (int)k - 1; i >= 0; --i) {
    double sum = rhs[i];
    for (size_t j = i + 1; j < k; ++j)
      sum -= Mat[i * k + j] * x[j];
    x[i] = sum / Mat[i * k + i];
  }

  return x;
}

class IllsImputerCpu : public IImputer {
public:
  IllsImputerCpu(int k = 10, int max_iter = 5) : K(k), MaxIter(max_iter) {}

  std::string name() const override {
    return "IllsImputerCpu (Iterative LLS, K=" + std::to_string(K) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    std::vector<float> X_curr(X, X + N * D);

    // Initial Mean Imputation
    std::vector<float> col_means(D, 0.0f);
    for (int j = 0; j < D; ++j) {
      double sum = 0;
      int count = 0;
      for (int i = 0; i < N; ++i) {
        if (Mask[i * D + j] == 1) {
          sum += X[i * D + j];
          count++;
        }
      }
      col_means[j] = (count > 0) ? (float)(sum / count) : 0.0f;
    }

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 0)
          X_curr[i * D + j] = col_means[j];
      }
    }

    // Iterative LLS
    for (int it = 0; it < MaxIter; ++it) {
      std::vector<float> X_next = X_curr;

      for (int i = 0; i < N; ++i) {
        std::vector<int> missing_cols;
        for (int j = 0; j < D; ++j) {
          if (Mask[i * D + j] == 0)
            missing_cols.push_back(j);
        }
        if (missing_cols.empty())
          continue;

        // 1. Find Neighbors based on full X_curr (no strict mask)
        struct Candidate {
          int id;
          double dist;
        };
        std::vector<Candidate> candidates;
        for (int r = 0; r < N; ++r) {
          if (r == i)
            continue;
          double dist_sq = 0.0;
          for (int j = 0; j < D; ++j) {
            double d = X_curr[i * D + j] - X_curr[r * D + j];
            dist_sq += d * d;
          }
          candidates.push_back({r, dist_sq});
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate &a, const Candidate &b) {
                    return a.dist < b.dist;
                  });

        int eff_k = std::min(K, (int)candidates.size());
        std::vector<int> neighbors;
        for (int k = 0; k < eff_k; ++k)
          neighbors.push_back(candidates[k].id);

        // 2. Linear Regression (A w = b)
        // b = target row values (observed)
        // A = neighbor row values at the same indices
        std::vector<std::vector<double>> A;
        std::vector<double> b;
        for (int j = 0; j < D; ++j) {
          if (Mask[i * D + j] == 1) {
            std::vector<double> row_A;
            for (int nid : neighbors)
              row_A.push_back(X_curr[nid * D + j]);
            A.push_back(row_A);
            b.push_back(X_curr[i * D + j]);
          }
        }

        if (A.size() >= eff_k) {
          std::vector<double> w = solve_ols_ridge_ills(A, b);
          for (int mj : missing_cols) {
            double pred = 0;
            for (size_t k = 0; k < neighbors.size(); ++k) {
              pred += w[k] * X_curr[neighbors[k] * D + mj];
            }
            X_next[i * D + mj] = (float)pred;
          }
        } else {
          // Fallback to average of neighbors
          for (int mj : missing_cols) {
            double sum = 0;
            for (int nid : neighbors)
              sum += X_curr[nid * D + mj];
            X_next[i * D + mj] = (float)(sum / eff_k);
          }
        }
      }

      // Check for convergence or update
      X_curr = X_next;
    }

    // Copy back
    for (int i = 0; i < N * D; ++i)
      X[i] = X_curr[i];
  }

private:
  int K;
  int MaxIter;
};

} // namespace impute
