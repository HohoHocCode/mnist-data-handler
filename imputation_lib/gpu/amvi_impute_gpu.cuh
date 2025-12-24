#pragma once
#include <cstdint>
#include <cuda_runtime.h>


namespace impute {

// AMVI: Adaptive Multiple Value Imputation
// Extends CMVE by automatically selecting K for each gene.
void amvi_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D,
                      int max_K, // Maximum allowable collateral genes
                      cudaStream_t stream = 0);

} // namespace impute
