#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

struct SvdWorkspace {
    int N, D, R;

    float* d_A_col = nullptr;
    float* d_U = nullptr;
    float* d_S = nullptr;
    float* d_VT = nullptr;
    float* d_recon_col = nullptr;

    int* d_info = nullptr;

    float* d_work = nullptr;
    int work_size = 0;

    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t solver = nullptr;

    SvdWorkspace(int N, int D, int R);
    ~SvdWorkspace();

    void allocate();
};
