#pragma once
#include <cstdint>
#include <cuda_runtime.h>


namespace impute {

void rlsp_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                      int n_pc, cudaStream_t stream);

} // namespace impute
