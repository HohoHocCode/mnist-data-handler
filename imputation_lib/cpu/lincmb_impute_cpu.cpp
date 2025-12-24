#include "../i_imputer.hpp"
#include "knn_impute_cpu.cpp"
#include "svd_impute_cpu.cpp"
#include <algorithm>
#include <cmath>
#include <vector>


namespace impute {

class LinCmbImputerCpu : public IImputer {
public:
  LinCmbImputerCpu(int K = 10, int rank = 5) : K_(K), Rank_(rank) {}

  std::string name() const override {
    return "LinCmbImputerCpu (K=" + std::to_string(K_) +
           ", R=" + std::to_string(Rank_) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    // 1. Run SVD on a copy
    std::vector<float> X_G(X, X + (size_t)N * D);
    SvdImputerCpu svd(Rank_);
    svd.impute(X_G.data(), Mask, N, D);

    // 2. Run KNN on a copy
    std::vector<float> X_L(X, X + (size_t)N * D);
    KnnImputerCpu knn(K_);
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
  int K_;
  int Rank_;
};

} // namespace impute
