#pragma once
#include <cstdint>
#include <cuda_runtime.h>


namespace impute {

// ARLSimpute: Autoregressive Least Squares
// Predicts missing values using a multiple regression model on P predictors.
void arls_impute_cuda(float *d_X, uint8_t *d_mask, int N, int D,
                      int P, // Number of AR predictors (lag)
                      cudaStream_t stream = 0);

} // namespace impute
