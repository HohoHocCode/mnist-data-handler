#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace impute {

class SknnImputerCpu : public IImputer {
public:
  SknnImputerCpu(int k = 10) : K(k) {}

  std::string name() const override {
    return "SknnImputerCpu (K=" + std::to_string(K) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    if (N <= 0 || D <= 0)
      return;

    std::vector<uint8_t> M(Mask, Mask + (size_t)N * D);

    // 1. Separate reference (complete) and target (incomplete) sets
    std::vector<int> ref; // base genes (no missing at current step)
    std::vector<int> tgt; // target genes (missing at current step)

    for (int i = 0; i < N; ++i) {
      bool has_missing = false;
      for (int c = 0; c < D; ++c) {
        if (M[i * D + c] == 0) {
          has_missing = true;
          break;
        }
      }
      if (has_missing)
        tgt.push_back(i);
      else
        ref.push_back(i);
    }

    // 2. Fallback: If no reference genes, use those with least missing values
    if (ref.empty()) {
      std::vector<std::pair<int, int>> counts;
      for (int i = 0; i < N; i++) {
        int m = 0;
        for (int c = 0; c < D; c++)
          if (M[i * D + c] == 0)
            m++;
        counts.push_back({m, i});
      }
      std::sort(counts.begin(), counts.end());
      int min_miss = counts[0].first;
      for (auto &p : counts) {
        if (p.first == min_miss)
          ref.push_back(p.second);
        else
          break;
      }
      // Remove these from tgt
      for (int r : ref) {
        tgt.erase(std::remove(tgt.begin(), tgt.end(), r), tgt.end());
      }
    }

    if (tgt.empty())
      return;

    // 3. Sort target genes by missingness (Sequential part)
    std::sort(tgt.begin(), tgt.end(), [&](int a, int b) {
      int ma = 0, mb = 0;
      for (int c = 0; c < D; ++c) {
        if (M[a * D + c] == 0)
          ma++;
        if (M[b * D + c] == 0)
          mb++;
      }
      return ma < mb;
    });

    // 4. Main SKNN Loop
    for (int g : tgt) {
      std::vector<std::pair<double, int>> neighbors;

      for (int r : ref) {
        double distSq = 0;
        int count = 0;
        for (int c = 0; c < D; ++c) {
          if (M[g * D + c] == 1 && M[r * D + c] == 1) { // Both observed
            double d = X[g * D + c] - X[r * D + c];
            distSq += d * d;
            count++;
          }
        }
        if (count > 0) {
          double finalDist =
              std::sqrt(distSq * (static_cast<double>(D) / count));
          neighbors.push_back({finalDist, r});
        }
      }

      if (neighbors.empty()) {
        // Fallback to mean imputation if no neighbors found
        for (int c = 0; c < D; c++) {
          if (M[g * D + c] == 0) {
            double sum = 0;
            int cnt = 0;
            for (int i = 0; i < N; i++)
              if (M[i * D + c] == 1) {
                sum += X[i * D + c];
                cnt++;
              }
            X[g * D + c] = cnt > 0 ? (float)(sum / cnt) : 0.0f;
            M[g * D + c] = 1;
          }
        }
        ref.push_back(g);
        continue;
      }

      std::sort(neighbors.begin(), neighbors.end());
      int K_eff = std::min(K, (int)neighbors.size());

      for (int c = 0; c < D; ++c) {
        if (M[g * D + c] == 1)
          continue; // Already observed

        double num = 0, den = 0;
        for (int k = 0; k < K_eff; ++k) {
          int r = neighbors[k].second;
          if (M[r * D + c] == 1) {
            double w = 1.0 / (neighbors[k].first + 1e-6);
            num += w * X[r * D + c];
            den += w;
          }
        }

        if (den > 0) {
          X[g * D + c] = (float)(num / den);
          M[g * D + c] = 1;
        }
      }

      // Add newly imputed gene to reference set for future target genes
      ref.push_back(g);
    }
  }

private:
  int K;
};

} // namespace impute
