#pragma once
#include <cuda_runtime.h>
#include <iostream>

namespace impute {

/**
 * @brief RAII-based high-precision GPU timer using CUDA Events.
 * Equivalent to NVBench's core timing functionality.
 */
class GpuTimer {
public:
  GpuTimer(cudaStream_t stream = 0) : stream_(stream), started_(false) {
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
  }

  ~GpuTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
  }

  void start() {
    cudaEventRecord(start_event_, stream_);
    started_ = true;
  }

  float stop() {
    if (!started_)
      return 0.0f;
    cudaEventRecord(stop_event_, stream_);
    cudaEventSynchronize(stop_event_);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
    started_ = false;
    return milliseconds;
  }

  /**
   * @brief Shorthand to print kernel time.
   */
  void report(const std::string &name, float time_ms) {
    std::cout << "[PROF] " << name << " Kernel: " << time_ms << " ms\n";
  }

private:
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
  cudaStream_t stream_;
  bool started_;
};

} // namespace impute
