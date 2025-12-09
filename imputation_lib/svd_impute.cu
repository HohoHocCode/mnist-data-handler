#include "svd_impute.cuh"

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <iostream>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err__ = (call);                                    \
    if (err__ != cudaSuccess) {                                    \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__)   \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";\
        throw std::runtime_error("CUDA error");                    \
    }                                                              \
} while(0)

#define CUSOLVER_CHECK(call) do {                                  \
    cusolverStatus_t st__ = (call);                                \
    if (st__ != CUSOLVER_STATUS_SUCCESS) {                         \
        std::cerr << "cuSolver error at "                          \
                  << __FILE__ << ":" << __LINE__ << "\n";          \
        throw std::runtime_error("cuSolver error");                \
    }                                                              \
} while(0)

#define CUBLAS_CHECK(call) do {                                    \
    cublasStatus_t st__ = (call);                                  \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                           \
        std::cerr << "cuBLAS error at "                            \
                  << __FILE__ << ":" << __LINE__ << "\n";          \
        throw std::runtime_error("cuBLAS error");                  \
    }                                                              \
} while(0)

namespace {

    constexpr int BLOCK = 256;

    // --------- 1. Tìm column mean (bỏ qua missing) ----------
    // mỗi thread xử lý 1 cột (OK nếu số cột không quá lớn)
    __global__
        void col_mean_kernel(const float* __restrict__ X,
            const std::uint8_t* __restrict__ mask,
            int N, int D,
            float* __restrict__ col_mean)
    {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= D) return;

        double sum = 0.0;
        int cnt = 0;
        for (int r = 0; r < N; ++r) {
            int idx = r * D + c;
            if (mask[idx] == 0) { // observed
                float v = X[idx];
                if (!isnan(v)) {
                    sum += v;
                    ++cnt;
                }
            }
        }
        col_mean[c] = (cnt == 0) ? 0.0f : static_cast<float>(sum / cnt);
    }

    // điền missing bằng column mean
    __global__
        void fill_mean_kernel(float* __restrict__ X,
            const std::uint8_t* __restrict__ mask,
            const float* __restrict__ col_mean,
            int N, int D)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = N * D;
        if (idx >= size) return;

        if (mask[idx] == 1) {
            int c = idx % D;
            X[idx] = col_mean[c];
        }
    }

    // --------- 2. row-major <-> column-major ----------

    // X_row: [r * D + c], X_col: [c * N + r]
    __global__
        void row_to_col_kernel(const float* __restrict__ X_row,
            float* __restrict__ X_col,
            int N, int D)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = N * D;
        if (idx >= size) return;

        int r = idx / D;
        int c = idx % D;
        int col_idx = c * N + r;
        X_col[col_idx] = X_row[idx];
    }

    __global__
        void col_to_row_kernel(const float* __restrict__ X_col,
            float* __restrict__ X_row,
            int N, int D)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = N * D;
        if (idx >= size) return;

        int r = idx / D;
        int c = idx % D;
        int col_idx = c * N + r;
        X_row[idx] = X_col[col_idx];
    }

    // --------- 3. update missing từ X_hat -----------

    __global__
        void update_missing_kernel(float* __restrict__ X,
            const float* __restrict__ X_hat,
            const std::uint8_t* __restrict__ mask,
            int N, int D)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = N * D;
        if (idx >= size) return;

        if (mask[idx] == 1) {
            X[idx] = X_hat[idx];
        }
    }

    // --------- 4. scale cột U theo S để tạo B = U_k * diag(S_k) ---------
    __global__
        void scale_columns_kernel(float* __restrict__ U,
            const float* __restrict__ S,
            int N, int k)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int size = N * k;
        if (idx >= size) return;

        int r = idx % N;
        int c = idx / N;  // vì U col-major: U[r + c*N]
        float s = S[c];
        U[r + c * N] *= s;
    }

} // anonymous namespace

namespace impute {

    void svd_impute(float* d_X_rowmajor,
        const std::uint8_t* d_mask,
        int N,
        int D,
        int rank,
        int max_iter,
        cudaStream_t stream)
    {
        if (!d_X_rowmajor || !d_mask)
            throw std::runtime_error("svd_impute: null pointer");

        if (rank <= 0 || rank > std::min(N, D))
            throw std::runtime_error("svd_impute: invalid rank");

        if (max_iter <= 0) max_iter = 1;

        // 1) Khởi tạo: điền missing = column mean
        float* d_col_mean = nullptr;
        CUDA_CHECK(cudaMalloc(&d_col_mean, D * sizeof(float)));

        dim3 grid_mean((D + BLOCK - 1) / BLOCK);
        dim3 block_mean(BLOCK);

        col_mean_kernel << <grid_mean, block_mean, 0, stream >> > (
            d_X_rowmajor, d_mask, N, D, d_col_mean
            );

        dim3 grid_fill((N * D + BLOCK - 1) / BLOCK);
        dim3 block_fill(BLOCK);

        fill_mean_kernel << <grid_fill, block_fill, 0, stream >> > (
            d_X_rowmajor, d_mask, d_col_mean, N, D
            );

        CUDA_CHECK(cudaFree(d_col_mean));

        // 2) Chuẩn bị cuSolver & cuBLAS
        cusolverDnHandle_t cusolverH = nullptr;
        cublasHandle_t     cublasH = nullptr;

        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
        CUBLAS_CHECK(cublasCreate(&cublasH));

        // đồng bộ stream
        CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
        CUBLAS_CHECK(cublasSetStream(cublasH, stream));

        // 3) Allocate matrix ở dạng column-major
        float* d_A_col = nullptr;   // N x D
        float* d_U = nullptr;   // N x min(N,D)
        float* d_VT = nullptr;   // min(N,D) x D
        float* d_S = nullptr;   // min(N,D)
        float* d_work = nullptr;   // workspace SVD
        int* d_info = nullptr;

        int lda = N;
        int ldu = N;
        int ldvt = std::min(N, D);
        int min_nd = std::min(N, D);

        CUDA_CHECK(cudaMalloc(&d_A_col, sizeof(float) * N * D));
        CUDA_CHECK(cudaMalloc(&d_U, sizeof(float) * N * min_nd));
        CUDA_CHECK(cudaMalloc(&d_VT, sizeof(float) * min_nd * D));
        CUDA_CHECK(cudaMalloc(&d_S, sizeof(float) * min_nd));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        // workspace query
        int lwork = 0;
        CUSOLVER_CHECK(
            cusolverDnSgesvd_bufferSize(
                cusolverH, N, D, &lwork
            )
        );
        CUDA_CHECK(cudaMalloc(&d_work, sizeof(float) * lwork));

        // buffer cho X_hat (row-major) và B = U_k * diag(S_k)
        float* d_X_hat_row = nullptr;
        float* d_B = nullptr;    // N x k (col-major)
        CUDA_CHECK(cudaMalloc(&d_X_hat_row, sizeof(float) * N * D));
        CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * N * rank));

        // 4) Vòng lặp SVD impute
        for (int it = 0; it < max_iter; ++it) {

            // 4.1 copy X_rowmajor -> A_col (column-major)
            int size = N * D;
            dim3 grid_r2c((size + BLOCK - 1) / BLOCK);
            dim3 block_r2c(BLOCK);

            row_to_col_kernel << <grid_r2c, block_r2c, 0, stream >> > (
                d_X_rowmajor, d_A_col, N, D
                );

            // 4.2 SVD: A_col = U * diag(S) * VT
            // jobu = 'S', jobvt = 'S' => U: N x min_nd, VT: min_nd x D
            signed char jobu = 'S';
            signed char jobvt = 'S';

            CUSOLVER_CHECK(
                cusolverDnSgesvd(
                    cusolverH,
                    jobu,
                    jobvt,
                    N,       // m
                    D,       // n
                    d_A_col, // A, lda x n
                    lda,
                    d_S,
                    d_U,     // U
                    ldu,
                    d_VT,    // VT
                    ldvt,
                    d_work,
                    lwork,
                    nullptr, // rwork (float không cần)
                    d_info
                )
            );

            // 4.3 Giữ rank thành phần đầu (k = rank)
            int k = std::min(rank, min_nd);

            // B = U_k * diag(S_k)  (N x k)
            // U đang N x min_nd, col-major, ta dùng cột 0..k-1
            // Scale từng cột U theo S.
            dim3 grid_scale((N * k + BLOCK - 1) / BLOCK);
            dim3 block_scale(BLOCK);

            scale_columns_kernel << <grid_scale, block_scale, 0, stream >> > (
                d_U, d_S, N, k
                );
            // Sau scale, cột 0..k-1 của d_U là B; ta copy sang d_B (N x k)
            // Có thể gemm trực tiếp dùng leading pointers, nhưng copy cho rõ.

            // copy B = U(:,0:k-1)
            // dùng cublasScopy trên từng cột
            for (int col = 0; col < k; ++col) {
                const float* src = d_U + col * N;
                float* dst = d_B + col * N;
                CUBLAS_CHECK(
                    cublasScopy(
                        cublasH,
                        N,
                        src, 1,
                        dst, 1
                    )
                );
            }

            // 4.4 X_hat_col = B (N x k) * V_k^T (k x D)  => N x D (col-major)
            // V_k^T: lấy k hàng đầu của d_VT (min_nd x D)
            // => gemm: (N x k) * (k x D) = (N x D)
            const float alpha = 1.0f;
            const float beta = 0.0f;

            // pointer tới V_k^T (k hàng đầu)
            // d_VT layout: (min_nd x D), col-major → cột j: d_VT + j*ldvt
            // ta chỉ dùng k hàng đầu => coi như ma trận k x D với ldvt = min_nd

            CUBLAS_CHECK(
                cublasSgemm(
                    cublasH,
                    CUBLAS_OP_N,   // B: N x k
                    CUBLAS_OP_N,   // V_k^T: k x D
                    N,             // m
                    D,             // n
                    k,             // k
                    &alpha,
                    d_B, N,      // B
                    d_VT, ldvt,    // V_k^T (hàng 0..k-1)
                    &beta,
                    d_A_col, N     // output: X_hat_col (reuse d_A_col)
                )
            );

            // 4.5 Convert X_hat_col -> row-major
            col_to_row_kernel << <grid_r2c, block_r2c, 0, stream >> > (
                d_A_col, d_X_hat_row, N, D
                );

            // 4.6 Update chỉ các ô missing
            update_missing_kernel << <grid_fill, block_fill, 0, stream >> > (
                d_X_rowmajor, d_X_hat_row, d_mask, N, D
                );

            // (optional) có thể check hội tụ bằng RMSE trên masked nếu muốn
        }

        // 5) free
        CUDA_CHECK(cudaFree(d_A_col));
        CUDA_CHECK(cudaFree(d_U));
        CUDA_CHECK(cudaFree(d_VT));
        CUDA_CHECK(cudaFree(d_S));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_X_hat_row));
        CUDA_CHECK(cudaFree(d_B));

        CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
        CUBLAS_CHECK(cublasDestroy(cublasH));
    }

} // namespace impute
