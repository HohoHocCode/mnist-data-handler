#include "ls_impute.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess){ \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA failure"); }} while(0)

#define CUBLAS_CHECK(call) \
    do { cublasStatus_t err = call; if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuBLAS failure"); }} while(0)

#define CUSOLVER_CHECK(call) \
    do { cusolverStatus_t err = call; if (err != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuSOLVER failure"); }} while(0)


namespace impute_ls {

    static void solve_least_squares(
        const float* d_A, int K, int d,
        const float* d_y,
        float* d_w,
        cublasHandle_t cublas,
        cusolverDnHandle_t solver,
        cudaStream_t stream
    ) {
        // Create ATA (d×d)
        float alpha = 1.0f, beta = 0.0f;

        float* d_ATA = nullptr;
        float* d_ATy = nullptr;

        CUDA_CHECK(cudaMalloc(&d_ATA, d * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ATy, d * sizeof(float)));

        // ATA = Aᵀ A
        CUBLAS_CHECK(cublasSgemm(
            cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            d, d, K,
            &alpha,
            d_A, K,
            d_A, K,
            &beta,
            d_ATA, d
        ));

        // ATy = Aᵀ y
        CUBLAS_CHECK(cublasSgemv(
            cublas,
            CUBLAS_OP_T,
            K, d,
            &alpha,
            d_A, K,
            d_y,
            1,
            &beta,
            d_ATy,
            1
        ));

        // Solve ATA w = ATy
        int work_size = 0;
        int* dev_info = nullptr;
        CUDA_CHECK(cudaMalloc(&dev_info, sizeof(int)));

        // Workspace query
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(
            solver,
            CUBLAS_FILL_MODE_UPPER,
            d,
            d_ATA,
            d,
            &work_size
        ));

        float* d_work = nullptr;
        CUDA_CHECK(cudaMalloc(&d_work, work_size * sizeof(float)));

        // Cholesky factorization ATA -> Uᵀ U
        CUSOLVER_CHECK(cusolverDnSpotrf(
            solver,
            CUBLAS_FILL_MODE_UPPER,
            d,
            d_ATA,
            d,
            d_work,
            work_size,
            dev_info
        ));

        // Solve Uᵀ U w = ATy
        CUSOLVER_CHECK(cusolverDnSpotrs(
            solver,
            CUBLAS_FILL_MODE_UPPER,
            d, 1,
            d_ATA,
            d,
            d_ATy,
            d,
            dev_info
        ));

        CUDA_CHECK(cudaMemcpyAsync(d_w, d_ATy, d * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));

        // Cleanup
        CUDA_CHECK(cudaFree(d_ATA));
        CUDA_CHECK(cudaFree(d_ATy));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFree(dev_info));
    }


    void ls_impute(
        float* d_data,
        const uint8_t* d_mask,
        int N, int D,
        int K,
        cudaStream_t stream
    ) {
        cublasHandle_t cublas;
        cusolverDnHandle_t solver;

        CUBLAS_CHECK(cublasCreate(&cublas));
        CUSOLVER_CHECK(cusolverDnCreate(&solver));

        cublasSetStream(cublas, stream);
        cusolverDnSetStream(solver, stream);

        // Allocate workspace for A (K×D), y (K), w (D)
        float* d_A;
        float* d_y;
        float* d_w;
        CUDA_CHECK(cudaMalloc(&d_A, K * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, D * sizeof(float)));

        // Copy data to host temporarily to check missing positions
        std::vector<uint8_t> h_mask(N * D);
        CUDA_CHECK(cudaMemcpy(h_mask.data(), d_mask, N * D, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                if (h_mask[i * D + j] == 0)
                    continue;

                // (1) chọn K hàng gần nhất (ở đây chọn đơn giản: lấy K hàng đầu đủ dữ liệu)
                std::vector<int> neighbors;
                neighbors.reserve(K);

                for (int r = 0; r < N && (int)neighbors.size() < K; r++) {
                    if (h_mask[r * D + j] == 0)
                        neighbors.push_back(r);
                }

                if (neighbors.size() < K)
                    continue;

                // Duyệt xây dựng A (host -> device)
                std::vector<float> h_A(K * D);
                std::vector<float> h_y(K);

                for (int k = 0; k < K; k++) {
                    int r = neighbors[k];

                    CUDA_CHECK(cudaMemcpy(
                        &h_A[k * D],
                        &d_data[r * D],
                        D * sizeof(float),
                        cudaMemcpyDeviceToHost
                    ));

                    CUDA_CHECK(cudaMemcpy(
                        &h_y[k],
                        &d_data[r * D + j],
                        sizeof(float),
                        cudaMemcpyDeviceToHost
                    ));
                }

                CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), K * D * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), K * sizeof(float), cudaMemcpyHostToDevice));

                // Solve LS
                solve_least_squares(
                    d_A, K, D,
                    d_y,
                    d_w,
                    cublas, solver,
                    stream
                );

                // compute predicted = dot(a_i, w)
                std::vector<float> h_ai(D), h_w(D);
                CUDA_CHECK(cudaMemcpy(h_ai.data(), &d_data[i * D], D * sizeof(float),
                    cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_w.data(), d_w, D * sizeof(float),
                    cudaMemcpyDeviceToHost));

                float pred = 0.0f;
                for (int t = 0; t < D; t++)
                    pred += h_ai[t] * h_w[t];

                // write back
                CUDA_CHECK(cudaMemcpy(&d_data[i * D + j], &pred,
                    sizeof(float), cudaMemcpyHostToDevice));
            }
        }

        cudaFree(d_A);
        cudaFree(d_y);
        cudaFree(d_w);

        cublasDestroy(cublas);
        cusolverDnDestroy(solver);
    }

}
