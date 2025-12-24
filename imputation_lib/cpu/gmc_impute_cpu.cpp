#pragma once
#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace impute {

class GmcImputerCpu : public IImputer {
public:
  GmcImputerCpu(int K = 5, int max_iter = 20) : K_(K), MaxIter_(max_iter) {}

  std::string name() const override {
    return "GmcImputerCpu (K=" + std::to_string(K_) + ")";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    if (N <= 0 || D <= 0)
      return;

    // This logic MUST match gmc_impute_gpu.cu host side exactly.

    std::vector<std::vector<double>> mu(K_, std::vector<double>(D));
    std::vector<std::vector<double>> var(K_, std::vector<double>(D, 1.0));
    std::vector<double> pi(K_, 1.0 / K_);

    std::mt19937 rng(42);

    std::vector<double> col_means(D, 0.0);
    for (int c = 0; c < D; c++) {
      double sum = 0;
      int cnt = 0;
      for (int i = 0; i < N; i++) {
        if (Mask[i * D + c] == 1) {
          sum += X[i * D + c];
          cnt++;
        }
      }
      col_means[c] = (cnt > 0) ? (sum / cnt) : 0.0;
    }

    std::uniform_int_distribution<int> dist(0, N - 1);
    for (int k = 0; k < K_; k++) {
      int idx = dist(rng);
      for (int c = 0; c < D; c++) {
        if (Mask[idx * D + c] == 1)
          mu[k][c] = X[idx * D + c];
        else
          mu[k][c] = col_means[c] + ((double)dist(rng) / N - 0.5) * 0.1;
      }
    }

    std::vector<std::vector<double>> gamma(N, std::vector<double>(K_));

    for (int iter = 0; iter < MaxIter_; iter++) {
      // E-Step
      for (int i = 0; i < N; i++) {
        std::vector<double> log_probs(K_);
        for (int k = 0; k < K_; k++) {
          double lp = std::log(pi[k] + 1e-9);
          for (int c = 0; c < D; c++) {
            if (Mask[i * D + c] == 1) {
              double diff = X[i * D + c] - mu[k][c];
              lp += -0.5 * (std::log(2 * 3.141592653589793 * var[k][c]) +
                            (diff * diff) / var[k][c]);
            }
          }
          log_probs[k] = lp;
        }

        double max_log = log_probs[0];
        for (int k = 1; k < K_; k++)
          if (log_probs[k] > max_log)
            max_log = log_probs[k];

        double lse = 0;
        if (max_log != -std::numeric_limits<double>::infinity()) {
          double s = 0;
          for (double lp : log_probs)
            s += std::exp(lp - max_log);
          lse = max_log + std::log(s);
        }

        for (int k = 0; k < K_; k++) {
          gamma[i][k] = std::exp(log_probs[k] - lse);
        }
      }

      // M-Step
      std::vector<double> Nk(K_, 0.0);
      for (int k = 0; k < K_; k++) {
        for (int i = 0; i < N; i++)
          Nk[k] += gamma[i][k];
        pi[k] = Nk[k] / N;

        for (int c = 0; c < D; c++) {
          double w_sum_x = 0, w_sum_r = 0;
          for (int i = 0; i < N; i++) {
            if (Mask[i * D + c] == 1) {
              w_sum_x += gamma[i][k] * X[i * D + c];
              w_sum_r += gamma[i][k];
            }
          }
          if (w_sum_r > 1e-6)
            mu[k][c] = w_sum_x / w_sum_r;

          double w_sum_diff = 0;
          for (int i = 0; i < N; i++) {
            if (Mask[i * D + c] == 1) {
              double d = X[i * D + c] - mu[k][c];
              w_sum_diff += gamma[i][k] * (d * d);
            }
          }
          if (w_sum_r > 1e-6)
            var[k][c] = w_sum_diff / w_sum_r;
          if (var[k][c] < 1e-4)
            var[k][c] = 1e-4;
        }
      }
    }

    // Impute
    for (int i = 0; i < N; i++) {
      for (int c = 0; c < D; c++) {
        if (Mask[i * D + c] == 0) {
          double val = 0;
          for (int k = 0; k < K_; k++)
            val += gamma[i][k] * mu[k][c];
          X[i * D + c] = (float)val;
        }
      }
    }
  }

private:
  int K_;
  int MaxIter_;
};

} // namespace impute
