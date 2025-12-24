#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace impute {

// Helper: Solve (A^T A + lambda*I) x = A^T b
// dimension of x is k (number of neighbors)
static std::vector<double>
solve_ols_ridge(const std::vector<std::vector<double>> &A,
                const std::vector<double> &b, double ridge_alpha = 1e-5) {
  if (A.empty())
    return {};
  size_t n = A.size();    // samples (common columns)
  size_t k = A[0].size(); // features (neighbors)

  // Normal Equations: (AtA + alpha*I) x = Atb

  // 1. Compute AtA (k x k) and Atb (k x 1)
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

  // Add Ridge Regularization
  for (size_t i = 0; i < k; ++i) {
    AtA[i * k + i] += ridge_alpha;
  }

  // 2. Solve linear system (using simple Gaussian Elimination)
  // AtA is Symmetric Positive Definite (due to Ridge), so stable.

  // Augment with Atb
  std::vector<double> Mat = AtA;
  std::vector<double> &rhs = Atb; // alias for clarity, modified in place if we
                                  // wanted, but we'll do row ops

  // We solve Mat * x = rhs
  // Let's implement simple Gaussian with partial pivoting for k x k
  std::vector<double> x(k);
  std::vector<int> p(k);
  std::iota(p.begin(), p.end(), 0);

  // Forward
  for (size_t i = 0; i < k; ++i) {
    // Pivot
    size_t pivot = i;
    double max_val = std::abs(Mat[i * k + i]);
    for (size_t r = i + 1; r < k; ++r) {
      if (std::abs(Mat[r * k + i]) > max_val) {
        max_val = std::abs(Mat[r * k + i]);
        pivot = r;
      }
    }

    // Swap rows in Mat and rhs
    if (pivot != i) {
      for (size_t c = i; c < k; ++c)
        std::swap(Mat[i * k + c], Mat[pivot * k + c]);
      std::swap(rhs[i], rhs[pivot]);
    }

    if (std::abs(Mat[i * k + i]) < 1e-12) {
      // Singular, return zeros or handle gracefully
      return std::vector<double>(k, 0.0);
    }

    // Eliminate
    for (size_t r = i + 1; r < k; ++r) {
      double factor = Mat[r * k + i] / Mat[i * k + i];
      for (size_t c = i; c < k; ++c) {
        Mat[r * k + c] -= factor * Mat[i * k + c];
      }
      rhs[r] -= factor * rhs[i];
    }
  }

  // Backward
  for (int i = (int)k - 1; i >= 0; --i) {
    double sum = rhs[i];
    for (size_t j = i + 1; j < k; ++j) {
      sum -= Mat[i * k + j] * x[j];
    }
    x[i] = sum / Mat[i * k + i];
  }

  return x;
}

/**
 * @brief Strict LLS (Kim et al. 2005) - Local Least Squares.
 *        Row-Neighbor based regression.
 *
 * Algorithm:
 *  For each gene (row) 'g' with missing values:
 *    1. Identify k nearest neighbor genes based on L2 distance on valid columns
 * common to both.
 *    2. Form A (neighbors' values on common obs cols) and b (target gene values
 * on common obs cols).
 *    3. Solve x = argmin ||Ax - b||^2 (simulated via w = (AtA + eps)^-1 At b).
 *    4. Impute missing values in 'g' using w and neighbors' values at missing
 * locations.
 */
class LlsImputerCpu : public IImputer {
public:
  LlsImputerCpu(int k = 10) : K(k) {}

  std::string name() const override {
    return "LlsImputerCpu (Strict Kim2005, K=" + std::to_string(K) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    // We work on a copy to allow independent updates (non-sequential by default
    // in LLS paper, though often usually done on static or copy) The paper
    // implies we use the original observed data for finding neighbors.
    // Iteration usually refers to algorithms like ILLS. Regular LLS is often
    // one-shot. However, if a neighbor has missing data where the target is
    // observed, we must handle strictly. Kim 2005: "The missing values are
    // estimated... k neighbor genes." We usually ignore columns where either
    // target or neighbor is missing for distance calc. For regression: We use
    // columns where Target is Observed AND Neighbor is Observed. Wait, if
    // Neighbor has missing value at the Target's missing index, we cannot use
    // that neighbor for prediction! So candidates for neighbors must NOT be
    // missing at the target's missing feature index? Kim 2005 constraint:
    // "genes that do not have missing values in the corresponding positions".

    // Optimization: Pre-compute valid masks or just do it brute force for
    // correctness first. Since N ~ 1000-10000, brute force per row is O(N^2 *
    // D). 10k*10k*200 = 2e10 ops. Might be slow for 10k rows on CPU. But for
    // 1000 rows (verification) it's fine.

    std::vector<float> X_out(X, X + N * D);

    for (int i = 0; i < N; ++i) {
      // Find missing indices in this row
      std::vector<int> missing_cols;
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 0)
          missing_cols.push_back(j);
      }
      if (missing_cols.empty())
        continue;

      // 1. Find Neighbors
      // Candidates: All other rows.
      // Must define distance.
      // Kim 2005: L2 norm.
      struct Candidate {
        int id;
        double dist;
      };
      std::vector<Candidate> candidates;

      // Debug: Count candidates before and after filter
      int total_others = 0;
      int passed_strict = 0;

      for (int r = 0; r < N; ++r) {
        if (r == i)
          continue;
        total_others++;

        // Check validity: Neighbor 'r' must be observed at ALL indices where
        // 'i' is missing? This is a strict requirement often used. Otherwise we
        // can't form the prediction.
        bool usable = true;
        for (int mc : missing_cols) {
          if (Mask[r * D + mc] == 0) {
            usable = false;
            break;
          }
        }
        if (!usable)
          continue;
        passed_strict++;

        // Calculate Distance on common observed indices
        double dist_sq = 0.0;
        int cnt = 0;
        for (int j = 0; j < D; ++j) {
          // Both must be observed
          if (Mask[i * D + j] == 1 && Mask[r * D + j] == 1) {
            double d = X[i * D + j] - X[r * D + j];
            dist_sq += d * d;
            cnt++;
          }
        }

        if (cnt > 0) {
          // Normalize distance by number of overlap? Paper uses standard norms.
          // We'll normalize to allow fair comparison.
          candidates.push_back({r, dist_sq / cnt});
        }
      }

      if (candidates.empty()) {
        continue; // Cannot impute
      }

      // Sort by distance
      std::sort(candidates.begin(), candidates.end(),
                [](const Candidate &a, const Candidate &b) {
                  return a.dist < b.dist;
                });

      // Select Top K
      int effective_k = std::min(K, (int)candidates.size());

      // 2. Form Regression Problem
      // y (target) ~ w * Neighbors
      // Use common observed columns between target 'i' and ALL selected
      // neighbors. This intersection might be small if K is large. Strictly: we
      // iterate columns j where Im[i,j]=1 AND Mask[all_neighbors, j]=1

      std::vector<int> neighbor_indices;
      for (int k = 0; k < effective_k; ++k)
        neighbor_indices.push_back(candidates[k].id);

      std::vector<std::vector<double>> A; // Samples x Neighbors
      std::vector<double> b;              // Samples (Target values)

      for (int c = 0; c < D; ++c) {
        if (Mask[i * D + c] == 0)
          continue; // Target missing, skip (this is what we want to predict)

        // Check if all neighbors observed
        bool all_obs = true;
        for (int nid : neighbor_indices) {
          if (Mask[nid * D + c] == 0) {
            all_obs = false;
            break;
          }
        }
        if (!all_obs)
          continue;

        // Add sample
        std::vector<double> row_A;
        for (int nid : neighbor_indices) {
          row_A.push_back(X[nid * D + c]);
        }
        A.push_back(row_A);
        b.push_back(X[i * D + c]);
      }

      // 3. Solve Weights
      // If A is underdetermined or empty, fallback?
      if (A.size() < effective_k) {
        // Not enough data to solve regression?
        // Fallback to average of neighbors at missing cols
        for (int mc : missing_cols) {
          double sum = 0;
          for (int nid : neighbor_indices)
            sum += X[nid * D + mc];
          X_out[i * D + mc] = (float)(sum / effective_k);
        }
      } else {
        // Solve OLS
        std::vector<double> w = solve_ols_ridge(A, b);

        // 4. Predict
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

    // copy back
    for (int i = 0; i < N * D; ++i)
      X[i] = X_out[i];
  }

private:
  int K;
};

} // namespace impute
