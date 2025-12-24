#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace impute {

class RlspImputerCpu : public IImputer {
public:
  RlspImputerCpu(int k, int n_pc, float ridge = 1e-2f)
      : K(k), N_PC(n_pc), Ridge(ridge) {}

  std::string name() const override {
    return "RlspImputerCpu (K=" + std::to_string(K) +
           ", PC=" + std::to_string(N_PC) + ")";
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

      // Find K Neighbors
      struct Score {
        int id;
        float dist;
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
                [](const Score &a, const Score &b) { return a.dist < b.dist; });

      // Process neighbors
      int n_neighbors = std::min(K, (int)neighbors.size());
      int d_obs = (int)obs_cols.size();
      if (n_neighbors == 0 || d_obs == 0)
        continue;

      // 3. Local PCA (on centered data)
      std::vector<float> centered_neighbors(n_neighbors * D);
      std::vector<float> neighbor_means(D, 0.0f);
      for (int c = 0; c < D; ++c) {
        float sum = 0;
        for (int k = 0; k < n_neighbors; ++k)
          sum += X_curr[neighbors[k].id * D + c];
        neighbor_means[c] = sum / n_neighbors;
        for (int k = 0; k < n_neighbors; ++k)
          centered_neighbors[k * D + c] =
              X_curr[neighbors[k].id * D + c] - neighbor_means[c];
      }

      std::vector<float> U_pc, S_diag_pc, V_pc;
      compute_svd_jacobi(centered_neighbors, n_neighbors, D, U_pc, S_diag_pc,
                         V_pc);

      // PCs: P = V_pc (top npc)
      int npc = std::min({N_PC, n_neighbors - 1, D - 1});
      if (npc < 1)
        npc = 1;

      std::vector<float> P_matrix(D * npc);
      for (int c = 0; c < D; ++c) {
        for (int p = 0; p < npc; ++p) {
          P_matrix[c * npc + p] = V_pc[c * D + p];
        }
      }

      // 4. Solve Regression: weights = (P_obs^T P_obs + Ridge*I)^-1 P_obs^T *
      // (y_obs - mean_obs)
      std::vector<float> PtP_mat(npc * npc, 0.0f);
      std::vector<float> Pty_mat(npc, 0.0f);
      for (int c : obs_cols) {
        float y_centered = X[i * D + c] - neighbor_means[c];
        for (int p1 = 0; p1 < npc; ++p1) {
          Pty_mat[p1] += P_matrix[c * npc + p1] * y_centered;
          for (int p2 = 0; p2 < npc; ++p2) {
            PtP_mat[p1 * npc + p2] +=
                P_matrix[c * npc + p1] * P_matrix[c * npc + p2];
          }
        }
      }

      for (int p = 0; p < npc; ++p)
        PtP_mat[p * npc + p] += Ridge;

      std::vector<float> weights_reg(npc);
      solve_linear_system(PtP_mat, Pty_mat, weights_reg, npc);

      // 5. Impute: pred = mean + P * weights
      for (int mc : missing_cols) {
        float pred = neighbor_means[mc];
        for (int p = 0; p < npc; ++p) {
          pred += weights_reg[p] * P_matrix[mc * npc + p];
        }
        X[i * D + mc] = pred;
      }
    }
  }

private:
  int K;
  int N_PC;
  float Ridge;

  void solve_linear_system(const std::vector<float> &A,
                           const std::vector<float> &b, std::vector<float> &x,
                           int n) {
    // Small Gaussian elimination or Cholesky
    std::vector<float> L = A;
    x = b;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < i; ++j) {
        float sum = 0;
        for (int k = 0; k < j; ++k)
          sum += L[i * n + k] * L[j * n + k];
        L[i * n + j] = (L[i * n + j] - sum) / L[j * n + j];
      }
      float sum = 0;
      for (int k = 0; k < i; ++k)
        sum += L[i * n + k] * L[i * n + k];
      float val = L[i * n + i] - sum;
      L[i * n + i] = std::sqrt(std::max(val, 1e-9f));
    }
    // Forward
    for (int i = 0; i < n; ++i) {
      float sum = 0;
      for (int j = 0; j < i; ++j)
        sum += L[i * n + j] * x[j];
      x[i] = (x[i] - sum) / L[i * n + i];
    }
    // Backward
    for (int i = n - 1; i >= 0; --i) {
      float sum = 0;
      for (int j = i + 1; j < n; ++j)
        sum += L[j * n + i] * x[j];
      x[i] = (x[i] - sum) / L[i * n + i];
    }
  }

  void compute_svd_jacobi(const std::vector<float> &A, int m, int n,
                          std::vector<float> &U, std::vector<float> &S,
                          std::vector<float> &V) {
    // Lightweight Jacobi SVD
    V.assign(n * n, 0.0f);
    for (int i = 0; i < n; ++i)
      V[i * n + i] = 1.0f;
    U = A; // Initial U is A

    for (int sweep = 0; sweep < 15; ++sweep) {
      bool changed = false;
      for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
          double a = 0, b = 0, c = 0;
          for (int k = 0; k < m; ++k) {
            a += (double)U[k * n + i] * U[k * n + i];
            b += (double)U[k * n + j] * U[k * n + j];
            c += (double)U[k * n + i] * U[k * n + j];
          }
          if (std::abs(c) > 1e-9) {
            changed = true;
            double zeta = (b - a) / (2.0 * c);
            double t = (zeta > 0 ? 1.0 : -1.0) /
                       (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
            double cs = 1.0 / std::sqrt(1.0 + t * t);
            double sn = cs * t;
            for (int k = 0; k < m; ++k) {
              float v1 = U[k * n + i], v2 = U[k * n + j];
              U[k * n + i] = (float)(cs * v1 - sn * v2);
              U[k * n + j] = (float)(sn * v1 + cs * v2);
            }
            for (int k = 0; k < n; ++k) {
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
    S.assign(n, 0.0f);
    for (int j = 0; j < n; ++j) {
      double n2 = 0;
      for (int i = 0; i < m; ++i)
        n2 += (double)U[i * n + j] * U[i * n + j];
      S[j] = (float)std::sqrt(n2);
      if (S[j] > 1e-9f) {
        for (int i = 0; i < m; ++i)
          U[i * n + j] /= S[j];
      }
    }
    // Sort
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int p1, int p2) { return S[p1] > S[p2]; });
    std::vector<float> Un(m * n), Vn(n * n), Sn(n);
    for (int j = 0; j < n; ++j) {
      Sn[j] = S[idx[j]];
      for (int i = 0; i < m; ++i)
        Un[i * n + j] = U[i * n + idx[j]];
      for (int i = 0; i < n; ++i)
        Vn[i * n + j] = V[i * n + idx[j]];
    }
    U = Un;
    V = Vn;
    S = Sn;
  }
};

} // namespace impute
