#include "svd_workspace.cuh"
#include <cstdio>
#include <stdexcept>

#include "cuda_utils.cuh"

SvdWorkspace::SvdWorkspace(int N_, int D_, int R_) : N(N_), D(D_), R(R_) {}

SvdWorkspace::~SvdWorkspace() {
  if (d_A_col)
    CUDA_CHECK(cudaFree(d_A_col));
  if (d_U)
    CUDA_CHECK(cudaFree(d_U));
  if (d_S)
    CUDA_CHECK(cudaFree(d_S));
  if (d_VT)
    CUDA_CHECK(cudaFree(d_VT));
  if (d_recon_col)
    CUDA_CHECK(cudaFree(d_recon_col));
  if (d_means)
    CUDA_CHECK(cudaFree(d_means));
  if (d_rwork)
    CUDA_CHECK(cudaFree(d_rwork));
  if (d_info)
    CUDA_CHECK(cudaFree(d_info));
  if (d_work)
    CUDA_CHECK(cudaFree(d_work));

  if (cublas)
    cublasDestroy(cublas);
  if (solver)
    cusolverDnDestroy(solver);
}

void SvdWorkspace::allocate() {
  CUBLAS_CHECK(cublasCreate(&cublas));
  CUSOLVER_CHECK(cusolverDnCreate(&solver));

  int min_dim = (N < D) ? N : D;

  // Allocate logic: Solver writes min(N,D) components.
  // We must allocate full buffer.

  CUDA_CHECK(cudaMalloc(&d_A_col, sizeof(float) * N * D));

  // U is N x min_dim
  CUDA_CHECK(cudaMalloc(&d_U, sizeof(float) * N * min_dim));

  // S is min_dim
  CUDA_CHECK(cudaMalloc(&d_S, sizeof(float) * min_dim));

  // VT MUST have LDVT >= min(N,D). Since VT is min(N,D) x D in column-major.
  // We specify LDVT = min_dim. Size min_dim * D.
  CUDA_CHECK(cudaMalloc(&d_VT, sizeof(float) * min_dim * D));

  CUDA_CHECK(cudaMalloc(&d_recon_col, sizeof(float) * N * D));
  CUDA_CHECK(cudaMalloc(&d_means, sizeof(float) * D));
  CUDA_CHECK(cudaMalloc(&d_rwork, sizeof(float) * min_dim * 10)); // Safe buffer
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(solver, N, D, &work_size));
  CUDA_CHECK(cudaMalloc(&d_work, sizeof(float) * work_size));
}
