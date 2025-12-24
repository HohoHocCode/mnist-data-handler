#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

struct SvdWorkspace {
  int N, D, R;

  float *d_A_col = nullptr;
  float *d_U = nullptr;
  float *d_S = nullptr;
  float *d_VT = nullptr;
  float *d_recon_col = nullptr;
  float *d_means = nullptr; // Dedicated buffer for col means
  float *d_rwork = nullptr; // Required for 16-arg signature

  int *d_info = nullptr;

  float *d_work = nullptr;
  int work_size = 0;

  cublasHandle_t cublas = nullptr;
  cusolverDnHandle_t solver = nullptr;

  SvdWorkspace(int N, int D, int R);
  ~SvdWorkspace();

  void allocate();
};
