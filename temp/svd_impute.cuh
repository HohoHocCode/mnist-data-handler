#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace impute {

    // d_X_rowmajor: N x D, NaN hoặc giá trị bất kỳ ở vị trí missing
    // d_mask      : N x D, 1 = missing, 0 = observed
    // rank        : số thành phần SVD giữ lại (k <= min(N, D))
    // max_iter    : số vòng lặp imputation (ví dụ 5)
    // stream      : CUDA stream (0 = default)
    void svd_impute(float* d_X_rowmajor,
        const std::uint8_t* d_mask,
        int n_samples,
        int n_features,
        int rank,
        int max_iter,
        cudaStream_t stream = 0);

} // namespace impute
