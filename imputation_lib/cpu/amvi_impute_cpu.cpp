#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace impute {

class AmviImputerCpu : public IImputer {
public:
  AmviImputerCpu(int max_K = 20) : max_K_(max_K) {}

  std::string name() const override {
    return "AmviImputerCpu (maxK=" + std::to_string(max_K_) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    // 1. Initial Mean Fill & Stats (For Correlation)
    std::vector<float> X_fill(X, X + (size_t)N * D);
    std::vector<float> means(D, 0.0f);
    std::vector<float> vars(D, 1.0f);

    for (int c = 0; c < D; ++c) {
      double sum = 0, sq_sum = 0;
      int count = 0;
      for (int i = 0; i < N; ++i) {
        if (Mask[i * D + c]) {
          float val = X[i * D + c];
          sum += val;
          sq_sum += val * val;
          count++;
        }
      }
      if (count > 0)
        means[c] = (float)(sum / count);
      if (count > 1) {
        float v = (float)((sq_sum - sum * sum / count) / (count - 1));
        vars[c] = (v > 1e-9f) ? v : 1e-9f;
      }

      for (int i = 0; i < N; i++)
        if (!Mask[i * D + c])
          X_fill[i * D + c] = means[c];
    }

    // 2. Correlation Matrix
    std::vector<float> Cov(D * D, 0.0f);
    std::vector<float> X_center = X_fill;
    for (int i = 0; i < N; i++)
      for (int c = 0; c < D; c++)
        X_center[i * D + c] -= means[c];

    for (int i = 0; i < D; i++) {
      for (int j = i; j < D; j++) {
        double dot = 0;
        for (int k = 0; k < N; k++)
          dot += X_center[k * D + i] * X_center[k * D + j];
        Cov[i * D + j] = (float)dot;
        Cov[j * D + i] = (float)dot;
      }
    }

    // 3. Imputation Loop
    for (int g = 0; g < D; g++) {
      std::vector<int> miss;
      for (int i = 0; i < N; i++)
        if (!Mask[i * D + g])
          miss.push_back(i);
      if (miss.empty())
        continue;

      // Sort potential collateral genes
      std::vector<std::pair<float, int>> cand;
      for (int c = 0; c < D; c++) {
        if (g == c)
          continue;
        float dot_g = Cov[g * D + g];
        float dot_c = Cov[c * D + c];
        float cov = Cov[g * D + c];
        float r2 = (cov * cov) / (dot_g * dot_c + 1e-9f);
        cand.push_back({r2, c});
      }
      std::sort(cand.rbegin(), cand.rend());

      struct RegModel {
        float alpha;
        float beta;
        float mse;
        int c;
      };
      std::vector<RegModel> all_models;
      double best_score = 1e30;
      int best_K = 0;

      for (int k = 1; k <= std::min(max_K_, (int)cand.size()); k++) {
        int c = cand[k - 1].second;
        double sx = 0, sy = 0, sxy = 0, sxx = 0;
        int n_pair = 0;
        for (int i = 0; i < N; i++) {
          if (Mask[i * D + g] && Mask[i * D + c]) {
            float y = X[i * D + g], x = X[i * D + c];
            sx += x;
            sy += y;
            sxy += x * y;
            sxx += x * x;
            n_pair++;
          }
        }
        if (n_pair < 5)
          continue;
        double denom = n_pair * sxx - sx * sx;
        if (std::abs(denom) < 1e-9)
          continue;
        double beta = (n_pair * sxy - sx * sy) / denom;
        double alpha = (sy - beta * sx) / n_pair;

        double sse = 0;
        for (int i = 0; i < N; i++) {
          if (Mask[i * D + g] && Mask[i * D + c]) {
            double err = X[i * D + g] - (alpha + beta * X[i * D + c]);
            sse += err * err;
          }
        }
        float mse = (float)(sse / (n_pair - 2));
        if (mse < 1e-6f)
          mse = 1e-6f;

        all_models.push_back({(float)alpha, (float)beta, mse, c});

        double total_se = 0;
        int total_cnt = 0;
        for (int i = 0; i < N; i++) {
          if (Mask[i * D + g]) {
            double num = 0, den = 0;
            for (const auto &m : all_models) {
              if (Mask[i * D + m.c]) {
                num += (m.alpha + m.beta * X[i * D + m.c]) / m.mse;
                den += 1.0 / m.mse;
              }
            }
            if (den > 0) {
              double err = X[i * D + g] - (num / den);
              total_se += err * err;
              total_cnt++;
            }
          }
        }
        if (total_cnt > 0) {
          double current_rmse = std::sqrt(total_se / total_cnt);
          if (current_rmse < best_score) {
            best_score = current_rmse;
            best_K = (int)all_models.size();
          }
        }
      }

      // Final Impute using best_K
      for (int i : miss) {
        double num = 0, den = 0;
        for (int k = 0; k < best_K; k++) {
          const auto &m = all_models[k];
          if (Mask[i * D + m.c]) {
            num += (m.alpha + m.beta * X[i * D + m.c]) / m.mse;
            den += 1.0 / m.mse;
          }
        }
        if (den > 0)
          X[i * D + g] = (float)(num / den);
        else
          X[i * D + g] = means[g];
      }
    }
  }

private:
  int max_K_;
};

} // namespace impute
