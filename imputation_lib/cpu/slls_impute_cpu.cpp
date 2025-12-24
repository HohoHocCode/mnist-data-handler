#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>


namespace impute {

static std::vector<double>
solve_ols_ridge_slls(const std::vector<std::vector<double>> &A,
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
 * @brief Strict SLLS (Sequential LLS) - Chu et al. (2008).
 *
 * Logic:
 * 1. Sort genes by missing rate.
 * 2. Impute genes one by one.
 * 3. Once imputed, a gene becomes "observed" and can be used as a neighbor for
 * subsequent genes.
 */
class SllsImputerCpu : public IImputer {
public:
  SllsImputerCpu(int k = 10) : K(k) {}

  std::string name() const override {
    return "SllsImputerCpu (Strict Chu2008, K=" + std::to_string(K) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    // 1. Calculate missing counts and sort indices
    struct GeneInfo {
      int id;
      int missing_cnt;
    };
    std::vector<GeneInfo> genes(N);
    for (int i = 0; i < N; ++i) {
      int cnt = 0;
      for (int j = 0; j < D; ++j)
        if (Mask[i * D + j] == 0)
          cnt++;
      genes[i] = {i, cnt};
    }

    // Sort: Ascending order of missing_cnt.
    // Those with 0 missing are already "done". We iterate strictly those with
    // missing_cnt > 0.
    std::sort(genes.begin(), genes.end(),
              [](const GeneInfo &a, const GeneInfo &b) {
                return a.missing_cnt < b.missing_cnt;
              });

    // Working buffers: We need dynamic mask tracking.
    // Copy Mask to modify it as we go?
    // Yes, in SLLS, once imputed, we treat it as observed for the next step.
    std::vector<uint8_t> current_mask(Mask, Mask + N * D);

    for (const auto &g : genes) {
      int i = g.id;
      if (g.missing_cnt == 0)
        continue; // Skip fully observed

      // Identify missing cols
      std::vector<int> missing_cols;
      for (int j = 0; j < D; ++j) {
        if (current_mask[i * D + j] == 0)
          missing_cols.push_back(j);
      }

      // 1. Find Neighbors among "currently observed or imputed" pool
      struct Candidate {
        int id;
        double dist;
      };
      std::vector<Candidate> candidates;

      for (int r = 0; r < N; ++r) {
        if (r == i)
          continue;

        // Neighbor r must be valid at missing_cols of i
        bool usable = true;
        for (int mc : missing_cols) {
          if (current_mask[r * D + mc] == 0) {
            usable = false;
            break;
          }
        }
        if (!usable)
          continue;

        // Dist
        double dist_sq = 0.0;
        int cnt = 0;
        for (int j = 0; j < D; ++j) {
          if (current_mask[i * D + j] == 1 && current_mask[r * D + j] == 1) {
            double d = X[i * D + j] - X[r * D + j];
            dist_sq += d * d;
            cnt++;
          }
        }
        if (cnt > 0)
          candidates.push_back({r, dist_sq / cnt});
      }

      if (candidates.empty())
        continue;

      std::sort(candidates.begin(), candidates.end(),
                [](const Candidate &a, const Candidate &b) {
                  return a.dist < b.dist;
                });

      // Select Top K
      int effective_k = std::min(K, (int)candidates.size());
      std::vector<int> neighbor_indices;
      for (int k = 0; k < effective_k; ++k)
        neighbor_indices.push_back(candidates[k].id);

      // Regression
      std::vector<std::vector<double>> A;
      std::vector<double> b;

      for (int c = 0; c < D; ++c) {
        // Only use columns where target is observed
        if (current_mask[i * D + c] == 0)
          continue;

        // Only use columns where ALL neighbors are observed
        bool all_obs = true;
        for (int nid : neighbor_indices) {
          if (current_mask[nid * D + c] == 0) {
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
          X[i * D + mc] = (float)(sum / effective_k);
          current_mask[i * D + mc] = 1; // Mark as observed!
        }
      } else {
        std::vector<double> w = solve_ols_ridge_slls(A, b);
        for (int mc : missing_cols) {
          double pred = 0.0;
          for (int k_idx = 0; k_idx < effective_k; ++k_idx) {
            int nid = neighbor_indices[k_idx];
            pred += w[k_idx] * X[nid * D + mc];
          }
          X[i * D + mc] = (float)pred;
          current_mask[i * D + mc] = 1; // Mark as observed!
        }
      }
    }
  }

private:
  int K;
};

} // namespace impute
