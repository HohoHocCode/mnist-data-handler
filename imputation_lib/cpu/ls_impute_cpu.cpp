#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>


namespace impute {

// Duplicate helper or we could have put it in a header. keeping it
// self-contained.
static std::vector<double>
solve_ols_ridge_ls(const std::vector<std::vector<double>> &A,
                   const std::vector<double> &b, double ridge_alpha = 1e-5) {
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

/**
 * @brief Strict LSimpute (Bo et al. 2004) - LSimpute_gene version.
 *        Row-Neighbor based regression using Absolute Pearson Correlation.
 *
 * Logic:
 *  - Selection: k neighbors with highest absolute correlation (1 - |r|).
 *  - Regression: OLS estimation (same as LLS).
 */
class LsImputerCpu : public IImputer {
public:
  LsImputerCpu(int k = 10) : K(k) {}

  std::string name() const override {
    return "LsImputerCpu (Strict Bo2004, K=" + std::to_string(K) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    std::vector<float> X_out(X, X + N * D);

    for (int i = 0; i < N; ++i) {
      std::vector<int> missing_cols;
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 0)
          missing_cols.push_back(j);
      }
      if (missing_cols.empty())
        continue;

      struct Candidate {
        int id;
        double correlation;
      };
      std::vector<Candidate> candidates;

      // Precompute means for i? No, mean depends on common overlap.

      for (int r = 0; r < N; ++r) {
        if (r == i)
          continue;
        bool usable = true;
        for (int mc : missing_cols) {
          if (Mask[r * D + mc] == 0) {
            usable = false;
            break;
          }
        }
        if (!usable)
          continue;

        // Pearson Correlation on common observed
        double sum_i = 0, sum_r = 0, sum_ii = 0, sum_rr = 0, sum_ir = 0;
        int cnt = 0;

        for (int j = 0; j < D; ++j) {
          if (Mask[i * D + j] == 1 && Mask[r * D + j] == 1) {
            double vi = X[i * D + j];
            double vr = X[r * D + j];
            sum_i += vi;
            sum_r += vr;
            sum_ii += vi * vi;
            sum_rr += vr * vr;
            sum_ir += vi * vr;
            cnt++;
          }
        }

        if (cnt > 2) {
          double num = (double)cnt * sum_ir - sum_i * sum_r;
          double den_i = (double)cnt * sum_ii - sum_i * sum_i;
          double den_r = (double)cnt * sum_rr - sum_r * sum_r;
          if (den_i > 0 && den_r > 0) {
            double r_val = num / std::sqrt(den_i * den_r);
            candidates.push_back({r, std::abs(r_val)}); // Absolute correlation
          }
        }
      }

      if (candidates.empty())
        continue;

      // Sort by absolute correlation (highest first)
      std::sort(candidates.begin(), candidates.end(),
                [](const Candidate &a, const Candidate &b) {
                  return a.correlation > b.correlation;
                });

      int effective_k = std::min(K, (int)candidates.size());
      std::vector<int> neighbor_indices;
      for (int k = 0; k < effective_k; ++k)
        neighbor_indices.push_back(candidates[k].id);

      // Form Regression (similar to LLS)
      std::vector<std::vector<double>> A;
      std::vector<double> b;

      for (int c = 0; c < D; ++c) {
        if (Mask[i * D + c] == 0)
          continue;
        bool all_obs = true;
        for (int nid : neighbor_indices) {
          if (Mask[nid * D + c] == 0) {
            all_obs = false;
            break;
          }
        }
        if (!all_obs)
          continue;

        std::vector<double> row_A;
        for (int nid : neighbor_indices)
          row_A.push_back(X[nid * D + c]);
        A.push_back(row_A);
        b.push_back(X[i * D + c]);
      }

      if (A.size() < effective_k) {
        for (int mc : missing_cols) {
          double sum = 0;
          for (int nid : neighbor_indices)
            sum += X[nid * D + mc];
          X_out[i * D + mc] = (float)(sum / effective_k);
        }
      } else {
        std::vector<double> w = solve_ols_ridge_ls(A, b);
        for (int mc : missing_cols) {
          double pred = 0.0;
          for (int k_idx = 0; k_idx < effective_k; ++k_idx) {
            int nid = neighbor_indices[k_idx];
            pred += w[k_idx] * X[nid * D + mc];
          }
          X_out[i * D + mc] = (float)pred;
        }
      }
    }

    for (int i = 0; i < N * D; ++i)
      X[i] = X_out[i];
  }

private:
  int K;
};

} // namespace impute
