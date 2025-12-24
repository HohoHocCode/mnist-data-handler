#pragma once
#include <cstdint>
#include <cuda_runtime.h>


namespace impute {

// LinCmb: Hybrid Imputation combining Global (SVD) and Local (KNN)
void lincmb_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D,
                        int k_knn = 10, int rank_svd = 5,
                        cudaStream_t stream = 0);

} // namespace impute
