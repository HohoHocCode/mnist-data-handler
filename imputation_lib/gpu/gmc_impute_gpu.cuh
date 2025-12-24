#pragma once
#include <cstdint>
#include <cuda_runtime.h>


namespace impute {

// Gaussian Mixture Clustering Imputation
// Runs on Host (CPU) for algorithmic stability with complex missing patterns.
// Diagonal Covariance Model.
void gmc_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D,
                     int K,        // Number of components
                     int max_iter, // EM iterations
                     cudaStream_t stream = 0);

} // namespace impute
