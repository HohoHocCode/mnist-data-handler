#include "svd_workspace.cuh"
#include <stdexcept>
#include <cstdio>

#define CUDA_CHECK(e) do{ cudaError_t err=(e); if(err!=cudaSuccess){ \
    printf("CUDA ERR: %s\n", cudaGetErrorString(err)); \
}}while(0)

#define CUBLAS_CHECK(e) do{ if((e)!=CUBLAS_STATUS_SUCCESS){ \
    printf("cuBLAS ERR\n"); }}while(0)

#define CUSOLVER_CHECK(e) do{ if((e)!=CUSOLVER_STATUS_SUCCESS){ \
    printf("cuSOLVER ERR\n"); }}while(0)


SvdWorkspace::SvdWorkspace(int N_, int D_, int R_)
    : N(N_), D(D_), R(R_) {
}

SvdWorkspace::~SvdWorkspace()
{
    if (d_A_col) CUDA_CHECK(cudaFree(d_A_col));
    if (d_U) CUDA_CHECK(cudaFree(d_U));
    if (d_S) CUDA_CHECK(cudaFree(d_S));
    if (d_VT) CUDA_CHECK(cudaFree(d_VT));
    if (d_recon_col) CUDA_CHECK(cudaFree(d_recon_col));
    if (d_info) CUDA_CHECK(cudaFree(d_info));
    if (d_work) CUDA_CHECK(cudaFree(d_work));

    if (cublas) cublasDestroy(cublas);
    if (solver) cusolverDnDestroy(solver);
}

void SvdWorkspace::allocate()
{
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    CUDA_CHECK(cudaMalloc(&d_A_col, sizeof(float) * N * D));
    CUDA_CHECK(cudaMalloc(&d_U, sizeof(float) * N * R));
    CUDA_CHECK(cudaMalloc(&d_S, sizeof(float) * R));
    CUDA_CHECK(cudaMalloc(&d_VT, sizeof(float) * R * D));
    CUDA_CHECK(cudaMalloc(&d_recon_col, sizeof(float) * N * D));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(
        solver, N, D, &work_size
    ));
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(float) * work_size));
}
