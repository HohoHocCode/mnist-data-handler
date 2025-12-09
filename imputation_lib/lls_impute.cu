#include "lls_impute.cuh"
#include "ls_workspace.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

// Các kernel giống hệt code cũ của bạn
__global__
void build_A_y_kernel(float* A, float* y,
    const float* X, const uint8_t* mask,
    int row, int col, int N, int D, int K)
{
    int k = threadIdx.x;
    if (k >= K) return;

    int r = (row + k + 1) % N;   // hiện tại chọn K hàng "hàng xóm" đơn giản

    y[k] = X[r * D + col];

    for (int c = 0; c < D; c++)
        A[k * D + c] = X[r * D + c];
}

__global__
void apply_prediction_kernel(float* X, const float* w,
    int row, int col, int D)
{
    float num = 0.f, den = 0.f;
    for (int c = 0; c < D; c++) {
        num += X[row * D + c] * w[c];
        den += fabsf(w[c]) + 1e-6f;
    }
    X[row * D + col] = num / den;
}

namespace impute {

    void lls_impute_cuda(float* d_X, uint8_t* d_mask,
        int N, int D, int K,
        cudaStream_t stream)
    {
        if (N <= 0 || D <= 0 || K <= 0) return;

        LsWorkspace ws(N, D, K);
        ws.allocate();

        // ❗ Copy mask sang host để dùng trong if (missing?)
        std::vector<uint8_t> h_mask(N * D);
        cudaMemcpy(h_mask.data(), d_mask,
            sizeof(uint8_t) * N * D,
            cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < D; j++)
            {
                // Chỉ impute nếu vị trí (i,j) đang missing
                if (h_mask[i * D + j] == 0)
                    continue;

                // 1) Xây A (K × D) và y (K) trên GPU
                build_A_y_kernel << <1, K, 0, stream >> > (
                    ws.d_A, ws.d_y,
                    d_X, d_mask,
                    i, j, N, D, K
                    );

                float a = 1.f, b = 0.f;

                // 2) ATA = Aᵀ A  (D × D)
                cublasSgemm(ws.cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    D, D, K,
                    &a,
                    ws.d_A, K,
                    ws.d_A, K,
                    &b,
                    ws.d_ATA, D
                );

                // 3) ATy = Aᵀ y  (D)
                cublasSgemv(ws.cublas,
                    CUBLAS_OP_T,
                    K, D,
                    &a,
                    ws.d_A, K,
                    ws.d_y, 1,
                    &b,
                    ws.d_ATy, 1
                );

                // 4) Giải hệ ATA w = ATy bằng Cholesky
                cusolverDnSpotrf(ws.solver,
                    CUBLAS_FILL_MODE_UPPER,
                    D, ws.d_ATA, D,
                    ws.d_work, ws.work_size,
                    ws.d_info
                );

                cusolverDnSpotrs(ws.solver,
                    CUBLAS_FILL_MODE_UPPER,
                    D, 1,
                    ws.d_ATA, D,
                    ws.d_ATy, D,
                    ws.d_info
                );

                // 5) Copy nghiệm w tạm vào d_w
                cudaMemcpyAsync(ws.d_w, ws.d_ATy,
                    sizeof(float) * D,
                    cudaMemcpyDeviceToDevice,
                    stream);

                // 6) Dùng w để dự đoán X[i, j]
                apply_prediction_kernel << <1, 1, 0, stream >> > (
                    d_X, ws.d_w, i, j, D
                    );
            }
        }

        cudaStreamSynchronize(stream);
    }

} // namespace impute
