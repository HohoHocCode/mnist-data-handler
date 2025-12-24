#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

struct LsWorkspace {
    int N, D, K;

    float* d_A = nullptr;
    float* d_y = nullptr;
    float* d_w = nullptr;

    float* d_ATA = nullptr;
    float* d_ATy = nullptr;
    int* d_info = nullptr;
    float* d_work = nullptr;
    int work_size = 0;

    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t solver = nullptr;

    LsWorkspace(int N, int D, int K);
    ~LsWorkspace();

    void allocate();
};
