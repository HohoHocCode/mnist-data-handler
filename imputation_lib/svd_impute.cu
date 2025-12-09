#include "svd_impute.cuh"
#include "svd_workspace.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cmath>
#include <cstdio>
#include <cstdint>

#define CUDA_CHECK(e) do{ cudaError_t err=(e); if(err!=cudaSuccess){ \
    printf("CUDA ERR: %s\n", cudaGetErrorString(err)); \
    return; }}while(0)

#define CUBLAS_CHECK(e) do{ if((e)!=CUBLAS_STATUS_SUCCESS){ \
    printf("cuBLAS ERR\n"); return; }}while(0)

#define CUSOLVER_CHECK(e) do{ if((e)!=CUSOLVER_STATUS_SUCCESS){ \
    printf("cuSOLVER ERR\n"); return; }}while(0)


// ---------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------

__global__ void col_mean_kernel(const float* X,
    const uint8_t* mask,
    int N, int D,
    float* mean)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;

    double sum = 0.0;
    int cnt = 0;
    for (int r = 0; r < N; r++) {
        int idx = r * D + c;
        if (mask[idx] == 0) {
            sum += X[idx];
            cnt++;
        }
    }
    mean[c] = cnt ? (float)(sum / cnt) : 0.f;
}

__global__ void fill_mean_kernel(float* X,
    const uint8_t* mask,
    const float* mean,
    int N, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    if (mask[idx] == 1) {
        int c = idx % D;
        X[idx] = mean[c];
    }
}


__global__ void row_to_col_kernel(const float* src,
    float* dst,
    int N, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;

    int r = idx / D;
    int c = idx % D;
    dst[c * N + r] = src[idx];
}

__global__ void col_to_row_kernel(const float* src,
    float* dst,
    int N, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;

    int r = idx / D;
    int c = idx % D;
    dst[idx] = src[c * N + r];
}

__global__ void update_missing_kernel(float* X,
    const float* Xhat,
    const uint8_t* mask,
    int N, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    if (mask[idx] == 1) X[idx] = Xhat[idx];
}

__global__ void scale_columns_kernel(float* U,
    const float* S,
    int N, int R)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * R) return;

    int r = idx % N;
    int c = idx / N;

    U[c * N + r] *= S[c];
}


// ---------------------------------------------------------------------
// Main SVD-impute
// ---------------------------------------------------------------------

namespace impute {

    void svd_impute(float* d_X, uint8_t* d_mask,
        int N, int D, int R,
        int max_iter, float tol,
        cudaStream_t stream)
    {
        SvdWorkspace ws(N, D, R);
        ws.allocate();

        for (int it = 0; it < max_iter; it++)
        {
            // --- compute column means
            col_mean_kernel << <(D + 255) / 256, 256, 0, stream >> > (
                d_X, d_mask, N, D, ws.d_S
                );

            // --- fill missing by mean
            fill_mean_kernel << <(N * D + 255) / 256, 256, 0, stream >> > (
                d_X, d_mask, ws.d_S, N, D
                );

            // --- convert row-major → col-major
            row_to_col_kernel << <(N * D + 255) / 256, 256, 0, stream >> > (
                d_X, ws.d_A_col, N, D
                );

            // --- SVD: A = U S V^T
            signed char jobu = 'S';
            signed char jobvt = 'S';

            CUSOLVER_CHECK(cusolverDnSgesvd(
                ws.solver,
                jobu, jobvt,
                N, D,
                ws.d_A_col, N,
                ws.d_S,
                ws.d_U, N,
                ws.d_VT, R,
                ws.d_work, ws.work_size,
                nullptr,
                ws.d_info
            ));

            // --- B = U * diag(S)
            scale_columns_kernel << <(N * R + 255) / 256, 256, 0, stream >> > (
                ws.d_U, ws.d_S, N, R
                );

            // --- reconstruct Â = B * V
            float a = 1.f, b = 0.f;
            CUBLAS_CHECK(cublasSgemm(
                ws.cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                D, N, R,
                &a,
                ws.d_VT, R,
                ws.d_U, N,
                &b,
                ws.d_recon_col, D
            ));

            // --- col-major → row-major
            col_to_row_kernel << <(N * D + 255) / 256, 256, 0, stream >> > (
                ws.d_recon_col, d_X, N, D
                );

            // --- update missing
            update_missing_kernel << <(N * D + 255) / 256, 256, 0, stream >> > (
                d_X, ws.d_recon_col, d_mask, N, D
                );
        }

        cudaStreamSynchronize(stream);
    }

} // namespace impute
