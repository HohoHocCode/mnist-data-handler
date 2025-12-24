#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>


namespace impute {

class CmveImputerCpu : public IImputer {
public:
  int K_;
  CmveImputerCpu(int K = 10) : K_(K) {}

  std::string name() const override { return "CMVE-CPU"; }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    // 1. Initial Mean Fill & Stats (For Correlation)
    std::vector<float> X_fill(X, X + N * D);
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
      if (count > 1)
        vars[c] = (float)((sq_sum - sum * sum / count) / (count - 1));

      // Fill missing with mean for full matrix correlation
      for (int i = 0; i < N; i++)
        if (!Mask[i * D + c])
          X_fill[i * D + c] = means[c];
    }

    // 2. Correlation Matrix
    std::vector<float> Cov(D * D, 0.0f);
    // Center X_fill
    std::vector<float> X_center = X_fill;
    for (int i = 0; i < N; i++)
      for (int c = 0; c < D; c++)
        X_center[i * D + c] -= means[c];

    // Cov = X^T * X (assuming already divided by N-1 or we just use DotProd for
    // Corr) Corr_sq = Cov_sq / (Var * Var). Scaling doesn't matter as long as
    // consistent. Let's compute Dot Products.
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
      std::vector<int> missing_rows;
      for (int i = 0; i < N; i++)
        if (Mask[i * D + g] == 0)
          missing_rows.push_back(i);
      if (missing_rows.empty())
        continue;

      // Select Candidates
      std::vector<std::pair<float, int>> candidates;
      for (int c = 0; c < D; c++) {
        if (g == c)
          continue;
        float cov = Cov[g * D + c]; // Dot product
        // Corr approx = Dot / sqrt(Dot_ii * Dot_jj) ?
        // Using previously calc Vars is better if N-1 scaling matches.
        // Here Cov is unscaled dot. Vars was scaled.
        // Normalized metric: (Dot_ij)^2 / (Dot_ii * Dot_jj)
        float dot_g = Cov[g * D + g];
        float dot_c = Cov[c * D + c];
        float corr_sq = (cov * cov) / (dot_g * dot_c + 1e-9f);
        candidates.push_back({corr_sq, c});
      }
      std::sort(candidates.rbegin(), candidates.rend());

      int K_eff = std::min(K_, (int)candidates.size());

      // Build models
      struct Model {
        float alpha;
        float beta;
        float weight;
        int c_idx;
      };
      std::vector<Model> models;

      for (int k = 0; k < K_eff; k++) {
        int c = candidates[k].second;
        double sx = 0, sy = 0, sxy = 0, sxx = 0;
        int n_pair = 0;
        for (int i = 0; i < N; i++) {
          if (Mask[i * D + g] && Mask[i * D + c]) {
            float y = X[i * D + g];
            float x = X[i * D + c];
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
        double mse = sse / (n_pair - 2);
        if (mse < 1e-6)
          mse = 1e-6;
        models.push_back({(float)alpha, (float)beta, (float)(1.0f / mse), c});
      }

      if (models.empty()) {
        for (int i : missing_rows)
          X[i * D + g] = means[g];
        continue;
      }

      // Predict
      for (int i : missing_rows) {
        double num = 0, den = 0;
        for (const auto &m : models) {
          if (Mask[i * D + m.c_idx] == 0)
            continue; // Skip missing predictors
          float val = X[i * D + m.c_idx];
          num += (m.alpha + m.beta * val) * m.weight;
          den += m.weight;
        }
        if (den > 0)
          X[i * D + g] = (float)(num / den);
        else
          X[i * D + g] = means[g];
      }
    }
  }
};

} // namespace impute
