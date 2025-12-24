#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace impute {

class CpuKnnImputer : public IImputer {
public:
  CpuKnnImputer(int k) : K(k) {}

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    std::vector<float> X_out(X, X + (N * D));
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 0) {
          X_out[i * D + j] = impute_one(X, Mask, N, D, i, j);
        }
      }
    }
    std::copy(X_out.begin(), X_out.end(), X);
  }

  std::string name() const override {
    return "CpuKNNimpute (K=" + std::to_string(K) + ")";
  }

private:
  int K;

  float impute_one(const float *X, const std::uint8_t *Mask, int N, int D,
                   int target_row, int target_col) {
    struct Neighbor {
      int row;
      float dist;
    };
    std::vector<Neighbor> neighbors;
    for (int i = 0; i < N; ++i) {
      if (i == target_row || Mask[i * D + target_col] == 0)
        continue;

      float distSq = 0.0f;
      int count = 0;
      for (int k = 0; k < D; ++k) {
        if (Mask[target_row * D + k] == 1 && Mask[i * D + k] == 1) {
          float diff = X[target_row * D + k] - X[i * D + k];
          distSq += diff * diff;
          count++;
        }
      }
      if (count > 0) {
        float finalDist = std::sqrt(distSq * (static_cast<float>(D) / count));
        neighbors.push_back({i, finalDist});
      }
    }
    if (neighbors.empty())
      return 0.0f;
    std::sort(
        neighbors.begin(), neighbors.end(),
        [](const Neighbor &a, const Neighbor &b) { return a.dist < b.dist; });
    int limit = std::min(K, (int)neighbors.size());
    float sum_weighted_val = 0.0f, sum_weight = 0.0f;
    for (int i = 0; i < limit; ++i) {
      float weight = 1.0f / (neighbors[i].dist + 1e-6f);
      sum_weighted_val += weight * X[neighbors[i].row * D + target_col];
      sum_weight += weight;
    }
    return sum_weighted_val / sum_weight;
  }
};

} // namespace impute
