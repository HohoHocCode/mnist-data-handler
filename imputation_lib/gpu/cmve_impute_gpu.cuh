#pragma once
#include <cstdint>
#include <cuda_runtime.h>


namespace impute {

// CMVE: Collateral Missing Value Estimation
// Hybrid approach: GPU for Correlation Matrix, Host for Regression logic.
void cmve_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D,
                      int K, // Number of collateral genes
                      cudaStream_t stream = 0);

} // namespace impute
