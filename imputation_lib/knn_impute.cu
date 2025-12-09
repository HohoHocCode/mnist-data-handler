#include "knn_impute.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__)    \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        throw std::runtime_error("CUDA error");                     \
    }                                                               \
} while(0)

namespace {

    constexpr int BLOCK = 256;
    constexpr float BIG = 1e9f;

    // ----------------------------------------------------
    // 1. Compute distance(i,j) using only shared non-missing features
    // ----------------------------------------------------
    __global__
        void compute_distances_kernel(const float* __restrict__ X,
            const uint8_t* __restrict__ mask,
            int N, int D,
            float* __restrict__ dist)
    {
        int i = blockIdx.x;      // mỗi block xử lý hàng i
        int j = threadIdx.x;     // mỗi thread xử lý hàng j

        if (i >= N || j >= N) return;

        if (i == j) {
            dist[i * N + j] = BIG;
            return;
        }

        float sum = 0.0f;
        int count = 0;

        for (int c = 0; c < D; c++) {
            int idx_i = i * D + c;
            int idx_j = j * D + c;

            if (mask[idx_i] == 0 && mask[idx_j] == 0) {
                float a = X[idx_i];
                float b = X[idx_j];
                float d = a - b;
                sum += d * d;
                count++;
            }
        }

        dist[i * N + j] = (count == 0) ? BIG : sqrtf(sum / count);
    }

    // ----------------------------------------------------
    // 2. Impute missing entries using K nearest neighbors
    // ----------------------------------------------------
    __global__
        void knn_fill_kernel(float* __restrict__ X,
            const uint8_t* __restrict__ mask,
            const float* __restrict__ dist,
            int N, int D, int K)
    {
        int i = blockIdx.x;   // sample
        int c = threadIdx.x;  // feature

        if (i >= N || c >= D) return;

        int idx = i * D + c;

        if (mask[idx] == 0) return;  // không missing

        // Tìm K hàng gần nhất
        const int MAXK = 32;
        K = min(K, MAXK);

        float best_d[MAXK];
        int   best_j[MAXK];

        for (int k = 0; k < K; k++) {
            best_d[k] = BIG;
            best_j[k] = -1;
        }

        // tìm KNN
        for (int j = 0; j < N; j++) {
            float d = dist[i * N + j];

            for (int k = 0; k < K; k++) {
                if (d < best_d[k]) {
                    // shift xuống
                    for (int t = K - 1; t > k; t--) {
                        best_d[t] = best_d[t - 1];
                        best_j[t] = best_j[t - 1];
                    }
                    best_d[k] = d;
                    best_j[k] = j;
                    break;
                }
            }
        }

        // Weighted average cho feature c
        float num = 0.0f;
        float den = 0.0f;

        for (int k = 0; k < K; k++) {
            int j = best_j[k];
            if (j < 0) continue;

            int idx_j = j * D + c;
            if (mask[idx_j] == 0) {
                float w = 1.0f / (best_d[k] + 1e-6f);
                num += w * X[idx_j];
                den += w;
            }
        }

        X[idx] = (den > 0 ? num / den : 0.0f);
    }

} // namespace


// ----------------------------------------------------
// PUBLIC FUNCTION
// ----------------------------------------------------
namespace impute {

    void knn_impute(float* d_X_rowmajor,
        const uint8_t* d_mask,
        int N,
        int D,
        int K,
        cudaStream_t stream)
    {
        if (!d_X_rowmajor || !d_mask)
            throw std::runtime_error("knn_impute: null pointer");

        if (K <= 0)
            throw std::runtime_error("knn_impute: invalid K");

        // 1. Allocate distance matrix NxN
        float* d_dist = nullptr;
        CUDA_CHECK(cudaMalloc(&d_dist, sizeof(float) * N * N));

        // 2. Compute distances
        {
            dim3 grid(N);
            dim3 block((N > BLOCK) ? BLOCK : N);
            compute_distances_kernel << <grid, block, 0, stream >> > (
                d_X_rowmajor, d_mask, N, D, d_dist
                );
        }

        // 3. Fill missing with KNN
        {
            dim3 grid(N);
            dim3 block((D > BLOCK) ? BLOCK : D);
            knn_fill_kernel << <grid, block, 0, stream >> > (
                d_X_rowmajor, d_mask, d_dist, N, D, K
                );
        }

        CUDA_CHECK(cudaFree(d_dist));
    }

} // namespace impute
