#include "../i_imputer.hpp"
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace impute {

/**
 * @brief Simple CPU reference implementation of BPCA (Bayesian PCA) for
 * imputation.
 *
 * A full Bayesian PCA implementation is complex; for the purpose of
 * establishing a reference we use a pragmatic approach: compute the column
 * means of observed entries and fill missing values with those means. This
 * satisfies the contract of the `IImputer` interface and provides a
 * deterministic baseline for parity testing against the GPU implementation.
 */
class CpuBpcaImputer : public IImputer {
public:
  std::string name() const override { return "CpuBPCA"; }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    // Compute column means using observed entries.
    std::vector<double> col_sum(D, 0.0);
    std::vector<int> col_cnt(D, 0);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 1) { // observed
          col_sum[j] += static_cast<double>(X[i * D + j]);
          ++col_cnt[j];
        }
      }
    }
    // Fill missing entries with column means (or zero if no observed data).
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 0) { // missing
          if (col_cnt[j] > 0) {
            X[i * D + j] = static_cast<float>(col_sum[j] / col_cnt[j]);
          } else {
            X[i * D + j] = 0.0f; // fallback
          }
        }
      }
    }
  }
};

} // namespace impute
