#include "../i_imputer.hpp"
#include "cpu_knn.cpp"
#include "cpu_svd.cpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>


namespace impute {

class CpuLinCmbImputer : public IImputer {
public:
  CpuLinCmbImputer(int k, int rank) : K(k), Rank(rank) {}

  std::string name() const override {
    return "CpuLinCmb (K=" + std::to_string(K) + ", R=" + std::to_string(Rank) +
           ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    // 1. Run SVD on a copy
    std::vector<float> X_G(X, X + (N * D));
    CpuSvdImputer svd(Rank, 25, 0.01f); // Using ALS approximation
    svd.impute(X_G.data(), Mask, N, D);

    // 2. Run KNN on a copy
    std::vector<float> X_L(X, X + (N * D));
    CpuKnnImputer knn(K);
    knn.impute(X_L.data(), Mask, N, D);

    // 3. Fusion
    for (int g = 0; g < D; g++) {
      double sum_ed = 0.0;
      double sum_d2 = 0.0;
      int count = 0;

      for (int i = 0; i < N; i++) {
        if (Mask[i * D + g] == 1) { // Observed
          float y = X[i * D + g];
          float g_val = X_G[i * D + g];
          float l_val = X_L[i * D + g];

          double e = (double)y - l_val;
          double d = (double)g_val - l_val;

          sum_ed += e * d;
          sum_d2 += d * d;
          count++;
        }
      }

      double w = 0.5;
      if (count > 0 && sum_d2 > 1e-9) {
        w = sum_ed / sum_d2;
        if (w < 0.0)
          w = 0.0;
        if (w > 1.0)
          w = 1.0;
      }

      // Apply to missing
      for (int i = 0; i < N; i++) {
        if (Mask[i * D + g] == 0) {
          X[i * D + g] =
              (float)(w * X_G[i * D + g] + (1.0 - w) * X_L[i * D + g]);
        }
      }
    }
  }

private:
  int K;
  int Rank;
};

} // namespace impute
