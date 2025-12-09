#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace metrics {

    // --------- Unmasked metrics (tính trên toàn bộ n phần tử) ---------

    // Mean Absolute Error
    float mae(const float* d_truth,
        const float* d_pred,
        std::size_t n,
        cudaStream_t stream = 0);

    // Root Mean Squared Error
    float rmse(const float* d_truth,
        const float* d_pred,
        std::size_t n,
        cudaStream_t stream = 0);

    // Normalized RMSE: sqrt(sum((y-ŷ)^2) / sum(y^2))
    float nrmse(const float* d_truth,
        const float* d_pred,
        std::size_t n,
        cudaStream_t stream = 0);

    // --------- Masked metrics (chỉ tính với mask[i] != 0) ---------

    // MAE trên các phần tử có mask[i] != 0
    float mae_masked(const float* d_truth,
        const float* d_pred,
        const std::uint8_t* d_mask,
        std::size_t n,
        cudaStream_t stream = 0);

    // RMSE trên các phần tử có mask[i] != 0
    float rmse_masked(const float* d_truth,
        const float* d_pred,
        const std::uint8_t* d_mask,
        std::size_t n,
        cudaStream_t stream = 0);

    // NRMSE trên các phần tử có mask[i] != 0
    float nrmse_masked(const float* d_truth,
        const float* d_pred,
        const std::uint8_t* d_mask,
        std::size_t n,
        cudaStream_t stream = 0);

    // Sum reduction — cần cho SVD, KNN và iKNN
    float sum_reduction(const float* d_data,
        std::size_t n,
        cudaStream_t stream = 0);
    // Sum of squares: Σ(x^2)

    float sumsq_reduction(const float* d_data,
        std::size_t n,
        cudaStream_t stream = 0);

    float sum_abs_masked(const float* d_truth,
        const float* d_pred,
        const uint8_t* d_mask,
        std::size_t n,
        cudaStream_t stream = 0);

    float sum_sq_masked(const float* d_truth,
        const float* d_pred,
        const uint8_t* d_mask,
        std::size_t n,
        cudaStream_t stream = 0);

    float sum_sq_truth_masked(const float* d_truth,
        const uint8_t* d_mask,
        std::size_t n,
        cudaStream_t stream = 0);

} // namespace metrics
