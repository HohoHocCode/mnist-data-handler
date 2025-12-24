#pragma once
#include <cstdint>
#include <cuda_runtime.h>


namespace impute {

void ills_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                      int max_iter, cudaStream_t stream);

} // namespace impute
