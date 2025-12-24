#include "ImputingLibrary/logger.hpp"
#include "imputation_lib/gpu/gpu_wrappers.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>


using namespace impute;

// Simple test framework
#define TEST_ASSERT(cond, msg)                                                 \
  if (!(cond)) {                                                               \
    Logger::error("Test Failed: " + std::string(msg));                         \
    return false;                                                              \
  }

// 1. Verify Warp-Shuffle Reductions
__global__ void test_warp_shuffle_reduction(float *d_out) {
  float val = (float)threadIdx.x; // 0, 1, 2, ..., 31
  // Warp reduce sum (0+1+...+31) = (31*32)/2 = 496
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);

  if (threadIdx.x == 0)
    *d_out = val;
}

bool run_reduction_test() {
  float *d_out;
  float h_out = 0;
  cudaMalloc(&d_out, sizeof(float));
  test_warp_shuffle_reduction<<<1, 32>>>(d_out);
  cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_out);

  TEST_ASSERT(std::abs(h_out - 496.0f) < 1e-5f,
              "Warp reduction sum incorrect (Expected 496, got " +
                  std::to_string(h_out) + ")");
  Logger::info("Warp reduction test passed.");
  return true;
}

// 2. Verify Jacobi Solver (Small Matrix)
// Handled by run_benchmarks parity, but adding a specific unit test for
// stability
bool run_solver_test() {
  // We'll use a small RLSP execution on tiny synthetic data
  int N = 10, D = 4, K = 3, pc = 2;
  std::vector<float> X(N * D, 1.0f);
  std::vector<uint8_t> M(N * D, 1);
  X[0] = 0.0f;
  M[0] = 0; // One missing value

  RlspImputerGpu imputer(K, pc);
  try {
    imputer.impute(X.data(), M.data(), N, D);
    TEST_ASSERT(!std::isnan(X[0]), "Solver produced NaN");
    Logger::info("Jacobi/RLSP solver stability test passed.");
  } catch (...) {
    TEST_ASSERT(false, "Solver threw exception");
  }
  return true;
}

int main() {
  Logger::info("========== STARTING UNIT TESTS ==========");

  bool all_passed = true;
  all_passed &= run_reduction_test();
  all_passed &= run_solver_test();

  if (all_passed) {
    Logger::info("========== ALL TESTS PASSED SUCCESSFULLY ==========");
    return 0;
  } else {
    Logger::error("========== SOME TESTS FAILED ==========");
    return 1;
  }
}
