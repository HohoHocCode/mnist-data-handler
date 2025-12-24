#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(err)                                                        \
  if (err != cudaSuccess) {                                                    \
    std::cout << "CUDA Error: " << cudaGetErrorString(err) << " at line "      \
              << __LINE__ << std::endl;                                        \
    exit(-1);                                                                  \
  }
#define CUSOLVER_CHECK(err)                                                    \
  if (err != CUSOLVER_STATUS_SUCCESS) {                                        \
    std::cout << "cuSOLVER Error: " << err << " at line " << __LINE__          \
              << std::endl;                                                    \
    exit(-1);                                                                  \
  }

int main() {
  int M = 6;
  int N = 4;
  std::vector<float> h_A(M * N, 1.0f);
  for (int i = 0; i < M * N; ++i)
    h_A[i] = (float)rand() / RAND_MAX;

  float *d_A, *d_S, *d_U, *d_VT, *d_work, *d_rwork;
  int *d_info;

  CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * M * N));
  CUDA_CHECK(cudaMalloc(&d_S, sizeof(float) * M));
  CUDA_CHECK(cudaMalloc(&d_U, sizeof(float) * M * M));
  CUDA_CHECK(cudaMalloc(&d_VT, sizeof(float) * M * N));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeof(float) * M * N,
                        cudaMemcpyHostToDevice));

  cusolverDnHandle_t handle;
  CUSOLVER_CHECK(cusolverDnCreate(&handle));

  int lwork = 0;
  CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(handle, M, N, &lwork));
  std::cout << "Required lwork: " << lwork << std::endl;
  CUDA_CHECK(cudaMalloc(&d_work, sizeof(float) * lwork));
  CUDA_CHECK(cudaMalloc(&d_rwork, sizeof(float) * M * 5));

  signed char jobu = 'S';
  signed char jobvt = 'S';

  // Test 1: M < N
  std::cout << "Testing SVD with M < N (4x6)..." << std::endl;
  cusolverStatus_t stat =
      cusolverDnSgesvd(handle, jobu, jobvt, M, N, d_A, M, d_S, d_U, M, d_VT, M,
                       d_work, lwork, d_rwork, d_info);
  std::cout << "Result: " << stat << std::endl;

  int info = 0;
  CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "Info: " << info << std::endl;

  CUSOLVER_CHECK(cusolverDnDestroy(handle));
  return 0;
}
