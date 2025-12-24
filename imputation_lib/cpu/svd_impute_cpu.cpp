#pragma once
#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace impute {

/**
 * @brief Rigorous SVDimpute (Troyanskaya 2001) using Iterative EM-SVD.
 */
class SvdImputerCpu : public IImputer {
public:
  SvdImputerCpu(int k, int max_iters = 10, float tol = 1e-4f)
      : K(k), MaxIters(max_iters), Tol(tol) {}

  std::string name() const override {
    return "SvdImputerCpu (K=" + std::to_string(K) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    if (N == 0 || D == 0)
      return;

    // 1. Initial imputation with column means
    std::vector<double> col_means(D, 0.0);
    std::vector<int> col_cnts(D, 0);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 1) {
          col_means[j] += X[i * D + j];
          col_cnts[j]++;
        }
      }
    }
    for (int j = 0; j < D; ++j) {
      if (col_cnts[j] > 0)
        col_means[j] /= col_cnts[j];
    }

    std::vector<float> X_curr(N * D);
    for (int i = 0; i < N * D; ++i) {
      if (Mask[i] == 1) {
        X_curr[i] = X[i];
      } else {
        X_curr[i] = static_cast<float>(col_means[i % D]);
      }
    }

    // 2. Iterative EM-SVD
    for (int iter = 0; iter < MaxIters; ++iter) {
      std::vector<float> U, S, V;
      compute_svd(X_curr, N, D, U, S, V);

      // Reconstruct using top K
      // X_recon = U[:, :K] * S[:K] * V[:, :K]^T
      // U: N x D (orthogonalized columns), S: D, V: D x D (orthogonal)
      // Note: compute_svd returns full matrices.
      int k_eff = std::min(K, D);

      std::vector<float> X_new = X_curr;
      float max_diff = 0.0f;

      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
          if (Mask[i * D + j] == 0) { // Only update missing
            float recon_val = 0.0f;
            for (int k = 0; k < k_eff; ++k) {
              // A = U * S * V^T => A_ij = sum_k U_ik * S_k * V_jk
              // U is N x D, V is D x D.
              // Jacobi SVD stores U as N x D (col major in the sense that
              // columns are ortho) But my U is N x D row-major-like storage but
              // processed by column. Let's re-verify indexing in compute_svd:
              // U[row * n_cols + col]
              recon_val += U[i * D + k] * S[k] * V[j * D + k];
            }
            float diff = std::abs(X_curr[i * D + j] - recon_val);
            max_diff = std::max(max_diff, diff);
            X_new[i * D + j] = recon_val;
          }
        }
      }

      X_curr = X_new;
      if (max_diff < Tol)
        break;
    }

    // 3. Final copy
    for (int i = 0; i < N * D; ++i) {
      if (Mask[i] == 0)
        X[i] = X_curr[i];
    }
  }

private:
  int K;
  int MaxIters;
  float Tol;

  void compute_svd(const std::vector<float> &A, int Rows, int Cols,
                   std::vector<float> &U, std::vector<float> &S,
                   std::vector<float> &V) {
    int n = Cols;
    int m = Rows;
    V.assign(n * n, 0.0f);
    for (int i = 0; i < n; ++i)
      V[i * n + i] = 1.0f;
    U = A;

    bool changed = true;
    int sweep = 0;
    while (changed &&
           sweep < 20) { // Jacobi SVD usually converges in few sweeps
      changed = false;
      sweep++;
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
              float temp = U[k * n + i];
              U[k * n + i] = (float)(cs * temp - sn * U[k * n + j]);
              U[k * n + j] = (float)(sn * temp + cs * U[k * n + j]);
            }
            for (int k = 0; k < n; ++k) {
              float temp = V[k * n + i];
              V[k * n + i] = (float)(cs * temp - sn * V[k * n + j]);
              V[k * n + j] = (float)(sn * temp + cs * V[k * n + j]);
            }
          }
        }
      }
    }

    S.assign(n, 0.0f);
    for (int j = 0; j < n; ++j) {
      double normSq = 0;
      for (int i = 0; i < m; ++i)
        normSq += (double)U[i * n + j] * U[i * n + j];
      S[j] = (float)std::sqrt(normSq);
      if (S[j] > 1e-9) {
        for (int i = 0; i < m; ++i)
          U[i * n + j] /= S[j];
      }
    }

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int i, int j) { return S[i] > S[j]; });

    std::vector<float> newS(n), newU(m * n), newV(n * n);
    for (int j = 0; j < n; ++j) {
      newS[j] = S[idx[j]];
      for (int i = 0; i < m; ++i)
        newU[i * n + j] = U[i * n + idx[j]];
      for (int i = 0; i < n; ++i)
        newV[i * n + j] = V[i * n + idx[j]];
    }
    S = newS;
    U = newU;
    V = newV;
  }
};

} // namespace impute
