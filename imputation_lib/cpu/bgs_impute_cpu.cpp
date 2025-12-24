#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace impute {

class BgsImputerCpu : public IImputer {
public:
  BgsImputerCpu(int max_genes = 20, float ridge = 1e-2f)
      : MaxGenes(max_genes), Ridge(ridge) {}

  std::string name() const override {
    return "BgsImputerCpu (MaxGenes=" + std::to_string(MaxGenes) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    if (N == 0 || D == 0)
      return;

    // 1. Initial Mean Imputation
    std::vector<float> col_means(D, 0.0f);
    std::vector<int> col_counts(D, 0);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 1) {
          col_means[j] += X[i * D + j];
          col_counts[j]++;
        }
      }
    }
    for (int j = 0; j < D; ++j) {
      if (col_counts[j] > 0)
        col_means[j] /= col_counts[j];
    }

    std::vector<float> X_curr(N * D);
    for (int i = 0; i < N * D; ++i) {
      X_curr[i] = (Mask[i] == 1) ? X[i] : col_means[i % D];
    }

    // 2. Process each row
    for (int i = 0; i < N; ++i) {
      std::vector<int> missing_cols;
      std::vector<int> obs_cols;
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 0)
          missing_cols.push_back(j);
        else
          obs_cols.push_back(j);
      }
      if (missing_cols.empty())
        continue;

      // 3. Find K Neighbors
      struct Score {
        int id;
        float d2;
      };
      std::vector<Score> neighbors;
      for (int r = 0; r < N; ++r) {
        if (r == i)
          continue;
        float d2 = 0;
        for (int j = 0; j < D; ++j) {
          float diff = X_curr[i * D + j] - X_curr[r * D + j];
          d2 += diff * diff;
        }
        neighbors.push_back({r, d2});
      }

      std::sort(neighbors.begin(), neighbors.end(),
                [](const Score &a, const Score &b) { return a.d2 < b.d2; });

      int n_neighbors = std::min(MaxGenes, (int)neighbors.size());
      if (n_neighbors == 0)
        continue;

      // 4. Center data for BGS robustness
      std::vector<float> neighbor_means(D, 0.0f);
      for (int j = 0; j < D; ++j) {
        float sum = 0;
        for (int k = 0; k < n_neighbors; ++k)
          sum += X_curr[neighbors[k].id * D + j];
        neighbor_means[j] = sum / n_neighbors;
      }

      // 5. Solve Weighted Ridge System via Eigen-decomposition
      std::vector<float> ATA(n_neighbors * n_neighbors, 0.0f);
      std::vector<float> ATy(n_neighbors, 0.0f);

      for (int c : obs_cols) {
        float y_centered = X[i * D + c] - neighbor_means[c];
        for (int r1 = 0; r1 < n_neighbors; ++r1) {
          float val1 = X_curr[neighbors[r1].id * D + c] - neighbor_means[c];
          ATy[r1] += val1 * y_centered;
          for (int r2 = r1; r2 < n_neighbors; ++r2) {
            float val2 = X_curr[neighbors[r2].id * D + c] - neighbor_means[c];
            float dot = val1 * val2;
            ATA[r1 * n_neighbors + r2] += dot;
            if (r1 != r2)
              ATA[r2 * n_neighbors + r1] += dot;
          }
        }
      }

      // Bayesian Weights: Far neighbors have smaller variances (higher penalty)
      float h = std::max(neighbors[n_neighbors - 1].d2, 1e-6f);
      for (int k = 0; k < n_neighbors; ++k) {
        float v_k = std::exp(-neighbors[k].d2 / h);
        ATA[k * n_neighbors + k] += Ridge / std::max(v_k, 1e-6f);
      }

      // Solve for weights_reg
      std::vector<float> weights_reg(n_neighbors, 0.0f);
      solve_robust_eigen(ATA, ATy, weights_reg, n_neighbors);

      // 6. Impute
      for (int mc : missing_cols) {
        float pred = neighbor_means[mc];
        for (int k = 0; k < n_neighbors; ++k) {
          pred += weights_reg[k] *
                  (X_curr[neighbors[k].id * D + mc] - neighbor_means[mc]);
        }
        if (std::isnan(pred) || std::isinf(pred))
          pred = col_means[mc];
        X[i * D + mc] = pred;
      }
    }
  }

private:
  int MaxGenes;
  float Ridge;

  void solve_robust_eigen(const std::vector<float> &A,
                          const std::vector<float> &b, std::vector<float> &x,
                          int n) {
    std::vector<float> V(n * n, 0.0f);
    for (int i = 0; i < n; ++i)
      V[i * n + i] = 1.0f;
    std::vector<float> U = A;

    // Small Jacobi for stability
    for (int sweep = 0; sweep < 20; ++sweep) {
      bool changed = false;
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          double g = U[i * n + i];
          double h = U[j * n + j];
          double f = U[i * n + j];
          if (std::abs(f) > 1e-12) {
            changed = true;
            double zeta = (h - g) / (2.0 * f);
            double t = (zeta > 0 ? 1.0 : -1.0) /
                       (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
            double cs = 1.0 / std::sqrt(1.0 + t * t);
            double sn = t * cs;
            for (int k = 0; k < n; k++) {
              float v1 = U[k * n + i], v2 = U[k * n + j];
              U[k * n + i] = (float)(cs * v1 - sn * v2);
              U[k * n + j] = (float)(sn * v1 + cs * v2);
            }
            for (int k = 0; k < n; k++) {
              float v1 = V[k * n + i], v2 = V[k * n + j];
              V[k * n + i] = (float)(cs * v1 - sn * v2);
              V[k * n + j] = (float)(sn * v1 + cs * v2);
            }
          }
        }
      }
      if (!changed)
        break;
    }

    std::vector<float> b_proj(n, 0.0f);
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++)
        b_proj[j] += V[i * n + j] * b[i];
    }

    std::vector<float> x_proj(n, 0.0f);
    for (int j = 0; j < n; j++) {
      double eig = 0;
      for (int i = 0; i < n; i++)
        eig += (double)U[i * n + j] * V[i * n + j];
      if (eig > 1e-9)
        x_proj[j] = (float)((double)b_proj[j] / eig);
      else
        x_proj[j] = 0;
    }

    for (int i = 0; i < n; i++) {
      x[i] = 0;
      for (int j = 0; j < n; j++)
        x[i] += V[i * n + j] * x_proj[j];
    }
  }
};

} // namespace impute
