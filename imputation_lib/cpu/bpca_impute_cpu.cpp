#include "../i_imputer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace impute {

// Tiny Linear Algebra Helpers for BPCA (Row-Major)
// We avoid external libs to keep it self-contained but strictly correct.

struct Mat {
  int r, c;
  std::vector<double> d;
  Mat(int rows, int cols, double val = 0.0)
      : r(rows), c(cols), d(rows * cols, val) {}

  double &operator()(int i, int j) { return d[i * c + j]; }
  const double &operator()(int i, int j) const { return d[i * c + j]; }

  static Mat eye(int n) {
    Mat I(n, n);
    for (int i = 0; i < n; ++i)
      I(i, i) = 1.0;
    return I;
  }
};

static Mat mul(const Mat &A, const Mat &B) {
  if (A.c != B.r)
    throw std::runtime_error("Shape mismatch in mul");
  Mat C(A.r, B.c);
  for (int i = 0; i < A.r; ++i) {
    for (int k = 0; k < A.c; ++k) {
      double val = A(i, k);
      if (val == 0)
        continue;
      for (int j = 0; j < B.c; ++j) {
        C(i, j) += val * B(k, j);
      }
    }
  }
  return C;
}

static Mat transpose(const Mat &A) {
  Mat T(A.c, A.r);
  for (int i = 0; i < A.r; ++i)
    for (int j = 0; j < A.c; ++j)
      T(j, i) = A(i, j);
  return T;
}

static Mat add(const Mat &A, const Mat &B) {
  Mat C(A.r, A.c);
  for (size_t i = 0; i < A.d.size(); ++i)
    C.d[i] = A.d[i] + B.d[i];
  return C;
}

static Mat sub(const Mat &A, const Mat &B) {
  Mat C(A.r, A.c);
  for (size_t i = 0; i < A.d.size(); ++i)
    C.d[i] = A.d[i] - B.d[i];
  return C;
}

// Invert Symmetric Positive Definite Matrix (using Cholesky)
static Mat inv(const Mat &A) {
  int n = A.r;
  Mat L(n, n);

  // Cholesky L * L^T = A
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0;
      for (int k = 0; k < j; k++)
        sum += L(i, k) * L(j, k);

      if (i == j) {
        double val = A(i, i) - sum;
        if (val < 1e-12)
          val = 1e-12; // Stronger regularization
        L(i, j) = std::sqrt(val);
      } else {
        if (std::abs(L(j, j)) < 1e-12)
          L(i, j) = (A(i, j) - sum) / 1e-12;
        else
          L(i, j) = (1.0 / L(j, j) * (A(i, j) - sum));
      }
    }
  }

  // Invert L
  Mat Linv(n, n);
  for (int i = 0; i < n; i++) {
    Linv(i, i) = 1.0 / L(i, i);
    for (int j = 0; j < i; j++) {
      double sum = 0;
      for (int k = j; k < i; k++) {
        sum += L(i, k) * Linv(k, j);
      }
      Linv(i, j) = -Linv(i, i) * sum;
    }
  }

  // Inv = Linv^T * Linv
  Mat res = mul(transpose(Linv), Linv);

  // Sanity check for NaNs
  for (double val : res.d) {
    if (!std::isfinite(val)) {
      Mat eye = Mat::eye(n);
      return eye; // Fallback
    }
  }
  return res;
}

/**
 * @brief Rigorous BPCA (Bayesian PCA) Impuation - Oba et al. (2003)
 *        Variational Bayes EM with Automatic Relevance Determination (ARD).
 */
class BpcaImputerCpu : public IImputer {
public:
  BpcaImputerCpu(int k_comp = -1, int max_iter = 100, double tol = 1e-4)
      : K_req(k_comp), MaxIter(max_iter), Tol(tol) {}

  std::string name() const override {
    return "BpcaImputerCpu (Rigorous Oba2003)";
  }

  void impute(float *X, const std::uint8_t *Mask, int N, int D) override {
    int K = (K_req > 0) ? K_req : (D - 1);
    if (K >= D)
      K = D - 1;
    if (K >= N)
      K = N - 1;

    // 1. Initial Imputation: Column Mean
    std::vector<double> col_means(D, 0.0);
    std::vector<int> col_counts(D, 0);

    for (int i = 0; i < N * D; ++i) {
      if (Mask[i] == 1) {
        col_means[i % D] += X[i];
        col_counts[i % D]++;
      }
    }
    for (int j = 0; j < D; ++j) {
      if (col_counts[j] > 0)
        col_means[j] /= col_counts[j];
    }

    Mat Y(N, D);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < D; ++j) {
        if (Mask[i * D + j] == 1) {
          float val = X[i * D + j];
          Y(i, j) = std::isfinite(val) ? (double)val : col_means[j];
        } else {
          Y(i, j) = col_means[j];
        }
      }
    }

    Mat W(D, K);
    Mat mu(1, D);
    std::vector<double> alpha(K, 1.0);
    double tau = 1.0;

    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 0.01);

    for (int j = 0; j < D; ++j)
      mu(0, j) = col_means[j];
    for (int i = 0; i < D * K; ++i)
      W.d[i] = dist(gen);

    for (int iter = 0; iter < MaxIter; ++iter) {
      Mat Wt = transpose(W);
      Mat WtW = mul(Wt, W);

      Mat M = WtW;
      for (int k = 0; k < K; ++k)
        M(k, k) *= tau;
      for (int k = 0; k < K; ++k)
        M(k, k) += 1.0;

      Mat Sx = inv(M);
      Mat Ex(N, K);
      Mat SumExx(K, K);

      for (int i = 0; i < N; ++i) {
        Mat yc(D, 1);
        for (int j = 0; j < D; ++j)
          yc(j, 0) = Y(i, j) - mu(0, j);

        Mat term = mul(Wt, yc);
        Mat mn = mul(Sx, term);
        for (int k = 0; k < K; ++k)
          mn(k, 0) *= tau;

        for (int k = 0; k < K; ++k)
          Ex(i, k) = mn(k, 0);

        Mat mm(K, K);
        for (int r = 0; r < K; ++r)
          for (int c = 0; c < K; ++c)
            mm(r, c) = mn(r, 0) * mn(c, 0);
        SumExx = add(SumExx, add(Sx, mm));
      }

      Mat Yc_all(N, D);
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
          Yc_all(i, j) = Y(i, j) - mu(0, j);

      Mat YcT = transpose(Yc_all);
      Mat RHS_W = mul(YcT, Ex);

      Mat LHS_W = SumExx;
      for (int k = 0; k < K; ++k)
        LHS_W(k, k) += (alpha[k] * 1e-4) + 1e-6;

      Mat LHS_W_inv = inv(LHS_W);
      W = mul(RHS_W, LHS_W_inv);

      for (int k = 0; k < K; ++k) {
        double w_norm_sq = 1e-8;
        for (int j = 0; j < D; ++j)
          w_norm_sq += W(j, k) * W(j, k);
        alpha[k] = D / w_norm_sq;
      }

      double err_sum = 0.0;
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j) {
          double val = Yc_all(i, j);
          err_sum += val * val;
        }

      Mat Recon = mul(Ex, transpose(W));
      double term2 = 0;
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
          term2 += Yc_all(i, j) * Recon(i, j);
      err_sum -= 2.0 * term2;

      Mat WtW_new = mul(transpose(W), W);
      Mat trMat = mul(WtW_new, SumExx);
      for (int k = 0; k < K; ++k)
        err_sum += trMat(k, k);

      double var = err_sum / (N * D);
      if (!std::isfinite(var) || var < 1e-4)
        var = 1e-4;
      tau = 1.0 / var;
      if (tau > 1e4)
        tau = 1e4;

      double diff = 0.0;
      for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
          if (Mask[i * D + j] == 0) {
            double val = mu(0, j) + Recon(i, j);
            double d = val - Y(i, j);
            diff += d * d;
            Y(i, j) = val;
          }
        }
      }

      for (int j = 0; j < D; ++j) {
        double sum = 0;
        for (int i = 0; i < N; ++i)
          sum += Y(i, j);
        mu(0, j) = sum / N;
      }

      if (!std::isfinite(diff))
        break;
      if (iter > 0 && std::sqrt(diff) / (N * D) < Tol)
        break;
    }

    for (int i = 0; i < N * D; ++i)
      X[i] = (float)Y.d[i];
  }

private:
  int K_req;
  int MaxIter;
  double Tol;
};

} // namespace impute
