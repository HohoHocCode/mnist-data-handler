#pragma once

#include <cstdint>
#include <string>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace impute {

class IImputer {
public:
  virtual ~IImputer() = default;
  virtual void impute(float *X, const uint8_t *Mask, int N, int D) = 0;
  virtual std::string name() const = 0;

  // GPU Streaming specialized method
#ifdef __CUDACC__
  virtual void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                           cudaStream_t stream) {
    // Default: Not implemented for CPU algorithms
  }
#else
  virtual void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                           void *stream) {
    // Non-CUDA fallback
  }
#endif
};

} // namespace impute
