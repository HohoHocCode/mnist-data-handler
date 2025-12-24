#include "ls_workspace.cuh"
#include <cstdio>

#include "cuda_utils.cuh"

LsWorkspace::LsWorkspace(int N_, int D_, int K_) : N(N_), D(D_), K(K_) {}

LsWorkspace::~LsWorkspace() {
  if (d_A)
    cudaFree(d_A);
  if (d_y)
    cudaFree(d_y);
  if (d_w)
    cudaFree(d_w);
  if (d_ATA)
    cudaFree(d_ATA);
  if (d_ATy)
    cudaFree(d_ATy);
  if (d_info)
    cudaFree(d_info);
  if (d_work)
    cudaFree(d_work);

  if (cublas)
    cublasDestroy(cublas);
  if (solver)
    cusolverDnDestroy(solver);
}

void LsWorkspace::allocate() {
  cublasCreate(&cublas);
  cusolverDnCreate(&solver);

  cudaMalloc(&d_A, sizeof(float) * K * D);
  cudaMalloc(&d_y, sizeof(float) * K);
  cudaMalloc(&d_w, sizeof(float) * D);

  cudaMalloc(&d_ATA, sizeof(float) * D * D);
  cudaMalloc(&d_ATy, sizeof(float) * D);

  cudaMalloc(&d_info, sizeof(int));

  cusolverDnSpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, D, d_ATA, D,
                              &work_size);

  cudaMalloc(&d_work, sizeof(float) * work_size);
}
