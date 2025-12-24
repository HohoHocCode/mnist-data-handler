#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace impute {

class ArlsImputerCpu : public IImputer {
public:
  ArlsImputerCpu(int K = 10, float ridge = 0.1f) : K_(K), Ridge_(ridge) {}

  std::string name() const override {
    return "ArlsImputerCpu (K=" + std::to_string(K_) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    std::vector<float> means(D, 0.0f);
    for (int c = 0; c < D; c++) {
      double sum = 0;
      int count = 0;
      for (int i = 0; i < N; i++)
        if (Mask[i * D + c]) {
          sum += X[i * D + c];
          count++;
        }
      if (count > 0)
        means[c] = (float)(sum / count);
    }

    // Correlation for neighbor selection (same as CMVE)
    std::vector<float> X_fill(X, X + (size_t)N * D);
    for (int c = 0; c < D; c++)
      for (int i = 0; i < N; i++)
        if (!Mask[i * D + c])
          X_fill[i * D + c] = means[c];

    std::vector<float> Cov(D * D, 0.0f);
    for (int i = 0; i < N; i++) {
      for (int c = 0; c < D; c++) {
        float xi = X_fill[i * D + c] - means[c];
        for (int k = 0; k < D; k++) {
          Cov[c * D + k] += xi * (X_fill[i * D + k] - means[k]);
        }
      }
    }

    for (int g = 0; g < D; g++) {
      std::vector<int> miss;
      for (int i = 0; i < N; i++)
        if (!Mask[i * D + g])
          miss.push_back(i);
      if (miss.empty())
        continue;

      std::vector<std::pair<float, int>> cand;
      for (int c = 0; c < D; c++) {
        if (g == c)
          continue;
        float r2 = (Cov[g * D + c] * Cov[g * D + c]) /
                   (Cov[g * D + g] * Cov[c * D + c] + 1e-9f);
        cand.push_back({r2, c});
      }
      std::sort(cand.rbegin(), cand.rend());

      int num_preds = std::min(K_, (int)cand.size());
      std::vector<int> preds;
      for (int k = 0; k < num_preds; k++)
        preds.push_back(cand[k].second);

      // Solve ridge regression: y = beta0 + beta1*x1 + ...
      int S = num_preds + 1;
      std::vector<double> A(S * S, 0.0);
      std::vector<double> y_sol(S, 0.0);

      int n_pair = 0;
      for (int i = 0; i < N; i++) {
        if (Mask[i * D + g]) {
          n_pair++;
          std::vector<float> row_preds(num_preds);
          for (int k = 0; k < num_preds; k++)
            row_preds[k] =
                Mask[i * D + preds[k]] ? X[i * D + preds[k]] : means[preds[k]];

          y_sol[0] += X[i * D + g];
          A[0 * S + 0] += 1.0;
          for (int k = 0; k < num_preds; k++) {
            A[0 * S + (k + 1)] += row_preds[k];
            A[(k + 1) * S + 0] += row_preds[k];
            for (int j = 0; j < num_preds; j++) {
              A[(k + 1) * S + (j + 1)] += (double)row_preds[k] * row_preds[j];
            }
            y_sol[k + 1] += (double)X[i * D + g] * row_preds[k];
          }
        }
      }

      if (n_pair < 5) {
        for (int i : miss)
          X[i * D + g] = means[g];
        continue;
      }

      // Ridge regularization
      for (int k = 0; k < S; k++)
        A[k * S + k] += Ridge_ * n_pair;

      // Solve A*beta = y_sol using simple Gaussian elimination
      std::vector<double> beta(S, 0.0);
      bool solved = true;
      for (int i = 0; i < S; i++) {
        int pivot = i;
        for (int j = i + 1; j < S; j++)
          if (std::abs(A[j * S + i]) > std::abs(A[pivot * S + i]))
            pivot = j;

        for (int k = 0; k < S; k++)
          std::swap(A[i * S + k], A[pivot * S + k]);
        std::swap(y_sol[i], y_sol[pivot]);

        if (std::abs(A[i * S + i]) < 1e-12) {
          solved = false;
          break;
        }

        for (int j = i + 1; j < S; j++) {
          double factor = A[j * S + i] / A[i * S + i];
          y_sol[j] -= factor * y_sol[i];
          for (int k = i; k < S; k++)
            A[j * S + k] -= factor * A[i * S + k];
        }
      }

      if (solved) {
        for (int i = S - 1; i >= 0; i--) {
          double sum = 0;
          for (int j = i + 1; j < S; j++)
            sum += A[i * S + j] * beta[j];
          beta[i] = (y_sol[i] - sum) / A[i * S + i];
        }
      }

      for (int i : miss) {
        if (!solved)
          X[i * D + g] = means[g];
        else {
          double pred_val = beta[0];
          for (int k = 0; k < num_preds; k++) {
            float val =
                Mask[i * D + preds[k]] ? X[i * D + preds[k]] : means[preds[k]];
            pred_val += beta[k + 1] * val;
          }
          X[i * D + g] = (float)pred_val;
        }
      }
    }
  }

private:
  int K_;
  float Ridge_;
};

} // namespace impute
