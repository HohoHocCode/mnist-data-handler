#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace impute {

    // d_X_rowmajor:     N x D (row-major), chứa NaN tại missing
    // d_mask:           N x D, 1 = missing, 0 = observed
    // K:                số neighbors
    // stream:           CUDA stream nếu cần
    void knn_impute(float* d_X_rowmajor,
        const std::uint8_t* d_mask,
        int N,
        int D,
        int K,
        cudaStream_t stream = 0);

}
