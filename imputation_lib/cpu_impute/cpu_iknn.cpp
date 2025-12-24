#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>


namespace impute {

/**
 * @brief CPU reference implementation of Iterative KNN (IKNNimpute).
 *
 * The algorithm performs KNN imputation iteratively, updating missing values
 * after each pass. For simplicity we run a fixed number of iterations (default
 * 5).
 */
class IkNNImputer : public IImputer {
public:
  IkNNImputer(int k = 2, int iterations = 5) : K(k), max_iters(iterations) {}

  std::string name() const override {
    return "IkNNImpute (K=" + std::to_string(K) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    // Work on a copy that we update iteratively.
    std::vector<float> X_cur(X, X + N * D);
    for (int iter = 0; iter < max_iters; ++iter) {
      std::vector<float> X_next = X_cur; // start from current state
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
          if (Mask[i * D + j] == 0) { // missing
            X_next[i * D + j] = impute_one(X_cur.data(), Mask, N, D, i, j);
          }
        }
      }
      X_cur.swap(X_next);
    }
    // Copy final result back to original matrix.
    std::copy(X_cur.begin(), X_cur.end(), X);
  }

private:
  int K;
  int max_iters;

  float impute_one(const float *Xcur, const std::uint8_t *Mask, int N, int D,
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
          float diff = Xcur[target_row * D + k] - Xcur[i * D + k];
          distSq += diff * diff;
          ++count;
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
    int limit = std::min(K, static_cast<int>(neighbors.size()));
    float sum_weighted = 0.0f, sum_weight = 0.0f;
    for (int idx = 0; idx < limit; ++idx) {
      float weight = 1.0f / (neighbors[idx].dist + 1e-6f);
      sum_weighted += weight * Xcur[neighbors[idx].row * D + target_col];
      sum_weight += weight;
    }
    return sum_weighted / sum_weight;
  }
};

} // namespace impute
