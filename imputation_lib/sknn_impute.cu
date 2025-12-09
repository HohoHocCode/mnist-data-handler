#include "sknn_impute.cuh"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>

namespace impute {

    void sknn_impute_cuda(
        float* d_X,
        uint8_t* d_mask,
        int N, int D,
        int K,
        cudaStream_t /*stream*/)
    {
        if (N <= 0 || D <= 0) return;

        // --- Copy từ GPU → CPU ---
        std::vector<float> X(N * D);
        std::vector<uint8_t> M(N * D);

        cudaMemcpy(X.data(), d_X, sizeof(float) * N * D, cudaMemcpyDeviceToHost);
        cudaMemcpy(M.data(), d_mask, sizeof(uint8_t) * N * D, cudaMemcpyDeviceToHost);

        // --- Tách reference / target ---
        std::vector<int> ref;
        std::vector<int> tgt;

        for (int i = 0; i < N; ++i) {
            bool missing = false;
            for (int c = 0; c < D; ++c)
                if (M[i * D + c] == 1) missing = true;

            if (missing) tgt.push_back(i);
            else ref.push_back(i);
        }

        if (ref.empty()) return;
        if (tgt.empty()) return;

        // --- Sort target theo missing-rate ---
        std::sort(tgt.begin(), tgt.end(),
            [&](int a, int b) {
                int ma = 0, mb = 0;
                for (int c = 0; c < D; ++c) {
                    ma += M[a * D + c];
                    mb += M[b * D + c];
                }
                return ma < mb;
            });

        // --- Function tính khoảng cách ---
        auto dist = [&](int i, int j) {
            double s = 0;
            int cnt = 0;
            for (int c = 0; c < D; ++c) {
                if (M[i * D + c] == 0 && M[j * D + c] == 0) {
                    double d = X[i * D + c] - X[j * D + c];
                    s += d * d;
                    cnt++;
                }
            }
            if (cnt == 0) return std::numeric_limits<double>::infinity();
            return s;
            };

        // --- SKNN MAIN LOOP ---
        for (int g : tgt) {

            // 1. Compute distance to reference set
            std::vector<std::pair<double, int>> dist_ref;

            for (int r : ref) {
                double d2 = dist(g, r);
                if (std::isfinite(d2))
                    dist_ref.emplace_back(d2, r);
            }

            if (dist_ref.empty()) continue;

            // 2. Sort neighbors
            std::sort(dist_ref.begin(), dist_ref.end(),
                [](auto& a, auto& b) { return a.first < b.first; });

            int K_eff = std::min(K, (int)dist_ref.size());

            // 3. Impute từng missing
            for (int c = 0; c < D; ++c) {
                if (M[g * D + c] == 0) continue;

                double num = 0, den = 0;

                for (int kk = 0; kk < K_eff; ++kk) {
                    int r = dist_ref[kk].second;
                    if (M[r * D + c] == 1) continue;

                    double w = 1.0 / (std::sqrt(dist_ref[kk].first) + 1e-6);
                    num += w * X[r * D + c];
                    den += w;
                }

                if (den > 0) {
                    X[g * D + c] = float(num / den);
                    M[g * D + c] = 0;
                }
            }

            // 4. Promote gene to reference set
            ref.push_back(g);
        }

        // --- Copy về GPU ---
        cudaMemcpy(d_X, X.data(), sizeof(float) * N * D, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask, M.data(), sizeof(uint8_t) * N * D, cudaMemcpyHostToDevice);
    }

} // namespace impute
