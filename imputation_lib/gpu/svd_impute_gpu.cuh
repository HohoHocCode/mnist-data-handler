#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace impute {

    void svd_impute(float* d_X,
                uint8_t* d_mask,
                int N, int D, int rank,
                int max_iter, float tol,
                cudaStream_t stream);

}  // namespace impute
