#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace impute {

    void sknn_impute_cuda(
        float* d_X,
        uint8_t* d_mask,
        int N, int D,
        int K,
        cudaStream_t stream);

}
