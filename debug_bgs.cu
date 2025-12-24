#include "ImputingLibrary/cuda_bin_io.hpp"
#include "imputation_lib/gpu/gpu_wrappers.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>


using namespace impute;

// Minimal CPU BGS for Validation
void bgs_cpu_min(float *X, const uint8_t *Mask, int N, int D, int K,
                 float Ridge) {
  std::vector<float> col_means(D, 0.0f);
  std::vector<int> col_counts(D, 0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      if (Mask[i * D + j]) {
        col_means[j] += X[i * D + j];
        col_counts[j]++;
      }
    }
  }
  for (int j = 0; j < D; ++j)
    if (col_counts[j] > 0)
      col_means[j] /= col_counts[j];

  std::vector<float> X_curr(N * D);
  for (int i = 0; i < N * D; ++i)
    X_curr[i] = Mask[i] ? X[i] : col_means[i % D];

  for (int i = 0; i < N; ++i) {
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
        float d = X_curr[i * D + j] - X_curr[r * D + j];
        d2 += d * d;
      }
      neighbors.push_back({r, d2});
    }
    std::sort(neighbors.begin(), neighbors.end(),
              [](const Score &a, const Score &b) { return a.d2 < b.d2; });

    int n_neigh = std::min(K, (int)neighbors.size());
    if (n_neigh == 0)
      continue;

    std::vector<float> n_means(D, 0.0f);
    for (int j = 0; j < D; ++j) {
      float s = 0;
      for (int k = 0; k < n_neigh; ++k)
        s += X_curr[neighbors[k].id * D + j];
      n_means[j] = s / n_neigh;
    }

    int K_eff = n_neigh;
    std::vector<double> ATA(K_eff * K_eff, 0.0);
    std::vector<double> ATy(K_eff, 0.0);

    // Build System
    for (int c = 0; c < D; ++c) {
      if (Mask[i * D + c]) {
        double y = (double)X[i * D + c] - n_means[c];
        for (int r1 = 0; r1 < K_eff; ++r1) {
          double x1 = (double)X_curr[neighbors[r1].id * D + c] - n_means[c];
          ATy[r1] += x1 * y;
          for (int r2 = 0; r2 < K_eff; ++r2) {
            double x2 = (double)X_curr[neighbors[r2].id * D + c] - n_means[c];
            ATA[r1 * K_eff + r2] += x1 * x2;
          }
        }
      }
    }

    // Regularize
    float h = std::max(neighbors[K_eff - 1].d2, 1e-6f);
    for (int k = 0; k < K_eff; ++k) {
      float v = std::exp(-neighbors[k].d2 / h);
      ATA[k * K_eff + k] += (double)Ridge / std::max(v, 1e-6f);
    }

    // Solve via simple Gaussian Elimination for Validation
    std::vector<double> A = ATA;
    std::vector<double> b = ATy;
    std::vector<float> w(K_eff);

    for (int k = 0; k < K_eff; ++k) {
      int p = k;
      for (int m = k + 1; m < K_eff; ++m)
        if (std::abs(A[m * K_eff + k]) > std::abs(A[p * K_eff + k]))
          p = m;
      for (int m = 0; m < K_eff; ++m)
        std::swap(A[p * K_eff + m], A[k * K_eff + m]);
      std::swap(b[p], b[k]);

      double diag = A[k * K_eff + k];
      if (std::abs(diag) < 1e-9)
        continue;
      for (int m = k + 1; m < K_eff; ++m) {
        double f = A[m * K_eff + k] / diag;
        for (int n = k; n < K_eff; ++n)
          A[m * K_eff + n] -= f * A[k * K_eff + n];
        b[m] -= f * b[k];
      }
    }
    for (int k = K_eff - 1; k >= 0; --k) {
      double sum = 0;
      for (int m = k + 1; m < K_eff; ++m)
        sum += A[k * K_eff + m] * w[m];
      w[k] = (float)((b[k] - sum) / A[k * K_eff + k]);
    }

    // Impute
    for (int j = 0; j < D; ++j) {
      if (!Mask[i * D + j]) {
        double res = n_means[j];
        for (int k = 0; k < K_eff; ++k)
          res += w[k] * ((double)X_curr[neighbors[k].id * D + j] - n_means[j]);
        X[i * D + j] = (float)res;
      }
    }
  }
}

// Validation helper
bool validate_bgs(int N, int D, int K) {
  std::cout << "Testing BGS Parity (Rand) N=" << N << ", D=" << D << ", K=" << K
            << std::endl;

  // Create random data
  std::vector<float> X(N * D);
  std::vector<uint8_t> M(N * D);
  for (int i = 0; i < N * D; i++) {
    X[i] = (float)(rand() % 100) / 10.0f;
    M[i] = (rand() % 10 > 2) ? 1 : 0;
  }

  std::vector<float> X_cpu = X;
  bgs_cpu_min(X_cpu.data(), M.data(), N, D, K, 0.01f);

  std::vector<float> X_gpu = X;
  try {
    BgsImputerGpu gpu(K, 0.01f);
    gpu.impute(X_gpu.data(), M.data(), N, D);

    double total_diff = 0;
    double max_diff = 0;
    int count = 0;
    for (int i = 0; i < N * D; ++i) {
      if (M[i] == 0) {
        double d = std::abs(X_cpu[i] - X_gpu[i]);
        total_diff += d;
        if (d > max_diff)
          max_diff = d;
        count++;
      }
    }
    std::cout << "Avg Diff: " << (total_diff / count)
              << ", Max Diff: " << max_diff << std::endl;

    if (max_diff > 1.0) {
      std::cout << "FAIL: Large Divergence." << std::endl;
      return false;
    }
    return true;

  } catch (std::exception &e) {
    std::cout << "Exception: " << e.what() << std::endl;
    return false;
  }
}

int main() {
  if (!validate_bgs(32, 32, 10))
    return 1;
  if (!validate_bgs(100, 200, 20))
    return 1;
  std::cout << "BGS Parity Tests Passed." << std::endl;
  return 0;
}
