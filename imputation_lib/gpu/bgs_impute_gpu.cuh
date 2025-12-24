#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace impute {

void bgs_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int max_genes,
                     float ridge, cudaStream_t stream);

} // namespace impute
