#include "imputation_lib/parity_checker.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>


using namespace impute;

int main() {
  std::cout << "Testing Parity Checker..." << std::endl;

  const int N = 2, D = 2;
  // Missing at [0,1] and [1,0]
  std::uint8_t Mask[N * D] = {1, 0, 0, 1};

  float X1[N * D] = {1.0f, 2.0f, 3.0f, 4.0f}; // CPU result
  float X2[N * D] = {1.0f, 2.05f, 3.0f,
                     4.0f}; // GPU result (simulated, small diff)

  // Test 1: Diff < Epsilon
  auto res1 = ParityChecker::compare(X1, X2, Mask, N, D, 0.1f);
  assert(res1.passed == true);
  assert(abs(res1.max_diff - 0.05f) < 1e-6);
  std::cout << "Test 1 Passed (Small Diff)\n";

  // Test 2: Diff > Epsilon
  auto res2 = ParityChecker::compare(X1, X2, Mask, N, D, 0.01f);
  assert(res2.passed == false);
  assert(res2.fail_count == 1);
  std::cout << "Test 2 Passed (Large Diff detected)\n";

  // Test 3: Ignore observed values
  float X3[N * D] = {99.0f, 2.0f, 3.0f,
                     4.0f}; // X3[0] differs largely but is OBSERVED (Mask=1)
  auto res3 = ParityChecker::compare(X1, X3, Mask, N, D, 0.01f);
  // Should pass because we only check where Mask == 0 (index 1 and 2)
  // index 1: 2.0 vs 2.0 (diff 0)
  // index 2: 3.0 vs 3.0 (diff 0)
  assert(res3.passed == true);
  std::cout << "Test 3 Passed (Observed values ignored)\n";

  std::cout << "SUCCESS: Parity Checker verified.\n";
  return 0;
}
