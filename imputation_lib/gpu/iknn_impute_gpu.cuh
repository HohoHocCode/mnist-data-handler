#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace impute {

    void iknn_impute(float* d_X_rowmajor,
        const uint8_t* d_mask,
        int N,
        int D,
        int K,
        int max_iter = 5,
        float tol = 1e-4f,
        cudaStream_t stream = 0);

}
