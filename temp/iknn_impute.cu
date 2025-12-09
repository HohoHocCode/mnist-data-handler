#include "iknn_impute.cuh"
#include "knn_impute.cuh"
#include "metrics.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err__ = (call);                                    \
    if (err__ != cudaSuccess) {                                    \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__)   \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";\
        throw std::runtime_error("CUDA error");                    \
    }                                                              \
} while(0)

namespace {

    constexpr int BLOCK = 256;

    // Tính L2 diff giữa hai bản X_prev và X_curr chỉ trên vùng missing
    __global__
        void diff_masked_kernel(const float* __restrict__ X_prev,
            const float* __restrict__ X_curr,
            const uint8_t* __restrict__ mask,
            float* __restrict__ diff,
            int N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;

        if (mask[idx] == 1) {
            float d = X_prev[idx] - X_curr[idx];
            diff[idx] = d * d;
        }
        else {
            diff[idx] = 0.0f;
        }
    }

} // namespace


// -------------------------------------------------
// PUBLIC FUNCTION: IKNNimpute
// -------------------------------------------------
namespace impute {

    void iknn_impute(float* d_X,
        const uint8_t* d_mask,
        int N,
        int D,
        int K,
        int max_iter,
        float tol,
        cudaStream_t stream)
    {
        if (!d_X || !d_mask)
            throw std::runtime_error("iknn_impute: null pointer");

        int size = N * D;

        // Bộ nhớ để lưu X_prev
        float* d_prev = nullptr;
        CUDA_CHECK(cudaMalloc(&d_prev, sizeof(float) * size));

        // Buffer tính diff
        float* d_diff = nullptr;
        CUDA_CHECK(cudaMalloc(&d_diff, sizeof(float) * size));

        // ---------------------------
        // Trước khi vào vòng lặp:
        // khởi tạo bằng KNN 1 vòng hoặc mean
        // => dùng lại KNN luôn
        // ---------------------------
        impute::knn_impute(d_X, d_mask, N, D, K, stream);

        // ---------------------------
        // Iterative refinement loop
        // ---------------------------
        for (int it = 0; it < max_iter; it++) {

            // Copy d_X -> d_prev
            CUDA_CHECK(cudaMemcpyAsync(
                d_prev, d_X,
                sizeof(float) * size,
                cudaMemcpyDeviceToDevice,
                stream
            ));

            // 1 vòng KNN mới
            impute::knn_impute(d_X, d_mask, N, D, K, stream);

            // Compute diff
            int blocks = (size + BLOCK - 1) / BLOCK;
            diff_masked_kernel << <blocks, BLOCK, 0, stream >> > (
                d_prev, d_X, d_mask, d_diff, size
                );

            float diff_value = metrics::sum_reduction(d_diff, size);
            float mse = diff_value / N;

            if (mse < tol) {
                // hội tụ
                break;
            }
        }

        CUDA_CHECK(cudaFree(d_prev));
        CUDA_CHECK(cudaFree(d_diff));
    }

} // namespace impute
