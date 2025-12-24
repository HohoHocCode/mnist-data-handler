#include "slls_impute_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <utility>
#include <vector>

// Helper to solve KxK linear system with Cholesky on CPU
// A: KxK (row-major), b: Kx1, x: Kx1
static void solve_ls_cpu(int K, const std::vector<double> &A,
                         const std::vector<double> &b, std::vector<double> &x) {
  std::vector<double> L(K * K, 0.0);

  // Cholesky L * L^T = A
  for (int i = 0; i < K; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0;
      for (int k = 0; k < j; k++)
        sum += L[i * K + k] * L[j * K + k];

      if (i == j) {
        double val = A[i * K + i] - sum;
        L[i * K + j] = (val > 0) ? std::sqrt(val) : 1e-6;
      } else {
        L[i * K + j] = (1.0 / L[j * K + j]) * (A[i * K + j] - sum);
      }
    }
  }

  // Forward Ly = b
  std::vector<double> y(K);
  for (int i = 0; i < K; i++) {
    double sum = 0;
    for (int j = 0; j < i; j++)
      sum += L[i * K + j] * y[j];
    y[i] = (b[i] - sum) / L[i * K + i];
  }

  // Backward L^T x = y
  for (int i = K - 1; i >= 0; i--) {
    double sum = 0;
    for (int j = i + 1; j < K; j++)
      sum += L[j * K + i] * x[j];
    x[i] = (y[i] - sum) / L[i * K + i];
  }
}

namespace impute {

void slls_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                      cudaStream_t /*stream*/) {
  if (N <= 0 || D <= 0)
    return;

  // 1. Copy GPU -> CPU
  std::vector<float> X(N * D);
  std::vector<uint8_t> M(N * D);

  cudaMemcpy(X.data(), d_X, sizeof(float) * N * D, cudaMemcpyDeviceToHost);
  cudaMemcpy(M.data(), d_mask, sizeof(uint8_t) * N * D, cudaMemcpyDeviceToHost);

  // 2. Identify Target (missing) and Reference (complete) genes
  std::vector<int> ref;
  std::vector<int> tgt;
  std::vector<int> missing_cnt(N, 0);

  for (int i = 0; i < N; ++i) {
    int obs_cnt = 0;
    for (int c = 0; c < D; ++c)
      if (M[i * D + c] == 1)
        obs_cnt++;

    missing_cnt[i] = D - obs_cnt;
    if (missing_cnt[i] > 0)
      tgt.push_back(i);
    else
      ref.push_back(i);
  }

  if (tgt.empty() || ref.empty())
    return; // Nothing to do or no info

  // 3. Sort targets by missing rate (ascending) -> "Sequential"
  std::sort(tgt.begin(), tgt.end(),
            [&](int a, int b) { return missing_cnt[a] < missing_cnt[b]; });

  // 4. Sequential Imputation Loop
  for (int g : tgt) {
    // Find K nearest neighbors from 'ref'
    // Distance metric: Euclidean on shared observed features (or correlation)
    // Liew paper uses Euclidean usually for LLS/SKNN context or correlation. We
    // use L2 for consistency with KNN imp.

    std::vector<std::pair<double, int>> dists;
    dists.reserve(ref.size());

    for (int r : ref) {
      double ss = 0;
      int count = 0;
      for (int c = 0; c < D; c++) {
        // Ignore if target is missing (standard)
        // If reference is missing (should not happen if ref is truly complete,
        // but SKNN promotes filled genes to ref, so they are complete now)
        if (M[g * D + c] == 1 && M[r * D + c] == 1) {
          double diff = X[g * D + c] - X[r * D + c];
          ss += diff * diff;
          count++;
        }
        // The original code had an extra `count++` here, which is removed.
      }
      // Normalize by count to handle different overlap sizes?
      // Standard L2 implies count shouldn't vary much or we trust raw sum.
      // Let's use correlation-like metric or just raw L2 on common valid.
      // Logic: LLS formulation usually requires valid overlap.
      if (count > 0)
        dists.emplace_back(ss, r);
    }

    if (dists.empty())
      continue; // No neighbors found

    // Sort top K
    // Optimization: Partial sort
    int K_eff = std::min(K, (int)dists.size());
    std::partial_sort(dists.begin(), dists.begin() + K_eff, dists.end());

    // 5. Formulate LLS: Aw = b
    // A (KxK): Covariance/Gram matrix of neighbors
    // b (Kx1): Covariance of target G vs neighbors
    // Here "Covariance" often approximated by dot product on valid features.
    // LSimpute/LLSimpute usually works on transposed matrix (Genes as samples),
    // but our X is NxM (Users x Items? Or Genes x Samples?)
    // Usually X is N=Genes, D=Samples.
    // We want to predict missing entry X[g, c].
    // We use K neighbors (other genes) that have value at c.
    // Weights w are trained on columns where 'g' and neighbors are ALL
    // observed.

    std::vector<int> neighbors;
    for (int k = 0; k < K_eff; k++)
      neighbors.push_back(dists[k].second);

    // Find common observed columns for training
    std::vector<int> common_cols;
    for (int c = 0; c < D; c++) {
      if (M[g * D + c] == 0)
        continue; // Gene g is missing in this column, skip
      bool all_neigh_ok = true;
      for (int r : neighbors) {
        if (M[r * D + c] == 0) {
          all_neigh_ok = false;
          break;
        } // Neighbor is missing, not a common observed column
      }
      if (all_neigh_ok)
        common_cols.push_back(c);
    }

    // If not enough common columns to solve system (need >= K), fallback to
    // weighted avg or smaller K For stability, we use however many we have,
    // with regularization.

    if (common_cols.empty()) {
      // Fallback: Mean/KNN
      // Just keep it 0 or prev fill?
      continue;
    }

    std::vector<double> A_mat(K_eff * K_eff, 0.0);
    std::vector<double> b_vec(K_eff, 0.0);

    // Fill A and b
    for (int i = 0; i < K_eff; i++) {
      int n_i = neighbors[i];
      // Fill b_i = dot(g, n_i)
      for (int c : common_cols) {
        b_vec[i] += X[g * D + c] * X[n_i * D + c];
      }

      // Fill A_ij = dot(n_i, n_j)
      for (int j = i; j < K_eff; j++) {
        int n_j = neighbors[j];
        double val = 0;
        for (int c : common_cols) {
          val += X[n_i * D + c] * X[n_j * D + c];
        }
        A_mat[i * K_eff + j] = val;
        A_mat[j * K_eff + i] = val;
      }
    }

    // Regularize diagonal
    for (int i = 0; i < K_eff; i++)
      A_mat[i * K_eff + i] += 1e-4; // Ridge

    // Solve
    std::vector<double> w_vec(K_eff);
    solve_ls_cpu(K_eff, A_mat, b_vec, w_vec);

    // 6. Predict missing values
    for (int c = 0; c < D; c++) {
      if (M[g * D + c] == 0) { // If gene g is missing in this column
        double pred = 0;
        // Formula: y = X_neighbors * w
        for (int k = 0; k < K_eff; k++) {
          int r = neighbors[k];
          // If neighbor effectively "complete", use its value
          // (ref set guarantees it's observed or filled)
          pred += X[r * D + c] * w_vec[k];
        }
        X[g * D + c] = (float)pred;
        M[g * D + c] = 1; // Mark as filled (observed)
      }
    }

    // 7. Promote to reference
    ref.push_back(g);
  }

  // 8. Copy back
  cudaMemcpy(d_X, X.data(), sizeof(float) * N * D, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, M.data(), sizeof(uint8_t) * N * D, cudaMemcpyHostToDevice);
}

} // namespace impute
