#pragma once
#include <cstdint>
#include <cuda_runtime.h>


namespace impute {

// Sequential LLS Imputation
// Runs sequentially on Host (CPU) to satisfy the dependency chain.
// Copy D2H -> Impute -> H2D.
void slls_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D, int K,
                      cudaStream_t stream = 0);

} // namespace impute
