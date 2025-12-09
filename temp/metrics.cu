#include "metrics.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <iostream>

namespace {

    constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            throw std::runtime_error("CUDA error");                            \
        }                                                                      \
    } while (0)

    // -------------------- Kernels cho MAE / MSE unmasked --------------------

    __global__
        void mae_kernel(const float* __restrict__ truth,
            const float* __restrict__ pred,
            std::size_t n,
            double* __restrict__ block_sums)
    {
        __shared__ double sdata[BLOCK_SIZE];

        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        double local = 0.0;

        // tính partial sum
        for (; idx < n; idx += blockDim.x * gridDim.x) {
            float diff = truth[idx] - pred[idx];
            local += fabsf(diff);
        }

        sdata[threadIdx.x] = local;
        __syncthreads();

        // reduce trong block
        for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                sdata[threadIdx.x] += sdata[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            block_sums[blockIdx.x] = sdata[0];
        }
    }

    __global__
        void mse_kernel(const float* __restrict__ truth,
            const float* __restrict__ pred,
            std::size_t n,
            double* __restrict__ block_sums)
    {
        __shared__ double sdata[BLOCK_SIZE];

        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        double local = 0.0;

        for (; idx < n; idx += blockDim.x * gridDim.x) {
            float diff = truth[idx] - pred[idx];
            local += static_cast<double>(diff) * static_cast<double>(diff);
        }

        sdata[threadIdx.x] = local;
        __syncthreads();

        for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                sdata[threadIdx.x] += sdata[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            block_sums[blockIdx.x] = sdata[0];
        }
    }

    // -------------------- Kernels cho masked (sum & count) -----------------

    __global__
        void mae_masked_kernel(const float* __restrict__ truth,
            const float* __restrict__ pred,
            const std::uint8_t* __restrict__ mask,
            std::size_t n,
            double* __restrict__ block_sums,
            unsigned int* __restrict__ block_counts)
    {
        __shared__ double sdata[BLOCK_SIZE];
        __shared__ unsigned int scount[BLOCK_SIZE];

        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        double local_sum = 0.0;
        unsigned int local_count = 0;

        for (; idx < n; idx += blockDim.x * gridDim.x) {
            if (mask[idx]) {
                float diff = truth[idx] - pred[idx];
                local_sum += fabsf(diff);
                local_count += 1;
            }
        }

        sdata[threadIdx.x] = local_sum;
        scount[threadIdx.x] = local_count;
        __syncthreads();

        for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                sdata[threadIdx.x] += sdata[threadIdx.x + offset];
                scount[threadIdx.x] += scount[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            block_sums[blockIdx.x] = sdata[0];
            block_counts[blockIdx.x] = scount[0];
        }
    }

    __global__
        void mse_masked_kernel(const float* __restrict__ truth,
            const float* __restrict__ pred,
            const std::uint8_t* __restrict__ mask,
            std::size_t n,
            double* __restrict__ block_sums,
            unsigned int* __restrict__ block_counts)
    {
        __shared__ double sdata[BLOCK_SIZE];
        __shared__ unsigned int scount[BLOCK_SIZE];

        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        double local_sum = 0.0;
        unsigned int local_count = 0;

        for (; idx < n; idx += blockDim.x * gridDim.x) {
            if (mask[idx]) {
                float diff = truth[idx] - pred[idx];
                local_sum += static_cast<double>(diff) * static_cast<double>(diff);
                local_count += 1;
            }
        }

        sdata[threadIdx.x] = local_sum;
        scount[threadIdx.x] = local_count;
        __syncthreads();

        for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                sdata[threadIdx.x] += sdata[threadIdx.x + offset];
                scount[threadIdx.x] += scount[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            block_sums[blockIdx.x] = sdata[0];
            block_counts[blockIdx.x] = scount[0];
        }
    }

    // ----------------- Kernel cho NRMSE: num = sum(diff^2), den = sum(y^2) ---

    __global__
        void nrmse_kernel(const float* __restrict__ truth,
            const float* __restrict__ pred,
            std::size_t n,
            double* __restrict__ block_num,
            double* __restrict__ block_den)
    {
        __shared__ double snum[BLOCK_SIZE];
        __shared__ double sden[BLOCK_SIZE];

        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        double local_num = 0.0;
        double local_den = 0.0;

        for (; idx < n; idx += blockDim.x * gridDim.x) {
            float y = truth[idx];
            float yhat = pred[idx];
            float diff = y - yhat;
            local_num += static_cast<double>(diff) * static_cast<double>(diff);
            local_den += static_cast<double>(y) * static_cast<double>(y);
        }

        snum[threadIdx.x] = local_num;
        sden[threadIdx.x] = local_den;
        __syncthreads();

        for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                snum[threadIdx.x] += snum[threadIdx.x + offset];
                sden[threadIdx.x] += sden[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            block_num[blockIdx.x] = snum[0];
            block_den[blockIdx.x] = sden[0];
        }
    }

    __global__
        void nrmse_masked_kernel(const float* __restrict__ truth,
            const float* __restrict__ pred,
            const std::uint8_t* __restrict__ mask,
            std::size_t n,
            double* __restrict__ block_num,
            double* __restrict__ block_den,
            unsigned int* __restrict__ block_counts)
    {
        __shared__ double snum[BLOCK_SIZE];
        __shared__ double sden[BLOCK_SIZE];
        __shared__ unsigned int scount[BLOCK_SIZE];

        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        double local_num = 0.0;
        double local_den = 0.0;
        unsigned int local_count = 0;

        for (; idx < n; idx += blockDim.x * gridDim.x) {
            if (mask[idx]) {
                float y = truth[idx];
                float yhat = pred[idx];
                float diff = y - yhat;
                local_num += static_cast<double>(diff) * static_cast<double>(diff);
                local_den += static_cast<double>(y) * static_cast<double>(y);
                local_count += 1;
            }
        }

        snum[threadIdx.x] = local_num;
        sden[threadIdx.x] = local_den;
        scount[threadIdx.x] = local_count;
        __syncthreads();

        for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                snum[threadIdx.x] += snum[threadIdx.x + offset];
                sden[threadIdx.x] += sden[threadIdx.x + offset];
                scount[threadIdx.x] += scount[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            block_num[blockIdx.x] = snum[0];
            block_den[blockIdx.x] = sden[0];
            block_counts[blockIdx.x] = scount[0];
        }
    }

} // anonymous namespace

// ================= HOST WRAPPERS ===================

namespace metrics {

    float mae(const float* d_truth,
        const float* d_pred,
        std::size_t n,
        cudaStream_t stream)
    {
        if (n == 0) return 0.0f;

        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        // giới hạn để tránh quá nhiều blocks
        blocks = std::min(blocks, 65535);

        double* d_block_sums = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(double)));

        mae_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (d_truth, d_pred, n, d_block_sums);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h_sums(blocks);
        CUDA_CHECK(cudaMemcpyAsync(h_sums.data(), d_block_sums,
            blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block_sums));

        double total = 0.0;
        for (double v : h_sums) total += v;

        return static_cast<float>(total / static_cast<double>(n));
    }

    float rmse(const float* d_truth,
        const float* d_pred,
        std::size_t n,
        cudaStream_t stream)
    {
        if (n == 0) return 0.0f;

        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        blocks = std::min(blocks, 65535);

        double* d_block_sums = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(double)));

        mse_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (d_truth, d_pred, n, d_block_sums);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h_sums(blocks);
        CUDA_CHECK(cudaMemcpyAsync(h_sums.data(), d_block_sums,
            blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block_sums));

        double total = 0.0;
        for (double v : h_sums) total += v;

        double mse = total / static_cast<double>(n);
        return static_cast<float>(std::sqrt(mse));
    }

    float nrmse(const float* d_truth,
        const float* d_pred,
        std::size_t n,
        cudaStream_t stream)
    {
        if (n == 0) return 0.0f;

        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        blocks = std::min(blocks, 65535);

        double* d_block_num = nullptr;
        double* d_block_den = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block_num, blocks * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_block_den, blocks * sizeof(double)));

        nrmse_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (d_truth, d_pred, n,
            d_block_num, d_block_den);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h_num(blocks), h_den(blocks);
        CUDA_CHECK(cudaMemcpyAsync(h_num.data(), d_block_num,
            blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_den.data(), d_block_den,
            blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block_num));
        CUDA_CHECK(cudaFree(d_block_den));

        double num = 0.0, den = 0.0;
        for (int i = 0; i < blocks; ++i) {
            num += h_num[i];
            den += h_den[i];
        }
        if (den == 0.0) {
            return 0.0f; // hoặc NAN, tùy bạn
        }
        return static_cast<float>(std::sqrt(num / den));
    }

    // -------------- Masked metrics ---------------------

    float mae_masked(const float* d_truth,
        const float* d_pred,
        const std::uint8_t* d_mask,
        std::size_t n,
        cudaStream_t stream)
    {
        if (n == 0) return 0.0f;

        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        blocks = std::min(blocks, 65535);

        double* d_block_sums = nullptr;
        unsigned int* d_block_counts = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_block_counts, blocks * sizeof(unsigned int)));

        mae_masked_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (
            d_truth, d_pred, d_mask, n, d_block_sums, d_block_counts);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h_sums(blocks);
        std::vector<unsigned int> h_counts(blocks);
        CUDA_CHECK(cudaMemcpyAsync(h_sums.data(), d_block_sums,
            blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_counts.data(), d_block_counts,
            blocks * sizeof(unsigned int),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block_sums));
        CUDA_CHECK(cudaFree(d_block_counts));

        double total = 0.0;
        unsigned int count = 0;
        for (int i = 0; i < blocks; ++i) {
            total += h_sums[i];
            count += h_counts[i];
        }
        if (count == 0) return 0.0f;

        return static_cast<float>(total / static_cast<double>(count));
    }

    float rmse_masked(const float* d_truth,
        const float* d_pred,
        const std::uint8_t* d_mask,
        std::size_t n,
        cudaStream_t stream)
    {
        if (n == 0) return 0.0f;

        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        blocks = std::min(blocks, 65535);

        double* d_block_sums = nullptr;
        unsigned int* d_block_counts = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_block_counts, blocks * sizeof(unsigned int)));

        mse_masked_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (
            d_truth, d_pred, d_mask, n, d_block_sums, d_block_counts);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h_sums(blocks);
        std::vector<unsigned int> h_counts(blocks);
        CUDA_CHECK(cudaMemcpyAsync(h_sums.data(), d_block_sums,
            blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_counts.data(), d_block_counts,
            blocks * sizeof(unsigned int),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block_sums));
        CUDA_CHECK(cudaFree(d_block_counts));

        double total = 0.0;
        unsigned int count = 0;
        for (int i = 0; i < blocks; ++i) {
            total += h_sums[i];
            count += h_counts[i];
        }
        if (count == 0) return 0.0f;

        double mse = total / static_cast<double>(count);
        return static_cast<float>(std::sqrt(mse));
    }

    float nrmse_masked(const float* d_truth,
        const float* d_pred,
        const std::uint8_t* d_mask,
        std::size_t n,
        cudaStream_t stream)
    {
        if (n == 0) return 0.0f;

        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        blocks = std::min(blocks, 65535);

        double* d_block_num = nullptr;
        double* d_block_den = nullptr;
        unsigned int* d_block_counts = nullptr;

        CUDA_CHECK(cudaMalloc(&d_block_num, blocks * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_block_den, blocks * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_block_counts, blocks * sizeof(unsigned int)));

        nrmse_masked_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (
            d_truth, d_pred, d_mask, n, d_block_num, d_block_den, d_block_counts);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h_num(blocks), h_den(blocks);
        std::vector<unsigned int> h_counts(blocks);

        CUDA_CHECK(cudaMemcpyAsync(h_num.data(), d_block_num,
            blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_den.data(), d_block_den,
            blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_counts.data(), d_block_counts,
            blocks * sizeof(unsigned int),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_block_num));
        CUDA_CHECK(cudaFree(d_block_den));
        CUDA_CHECK(cudaFree(d_block_counts));

        double num = 0.0, den = 0.0;
        unsigned int count = 0;
        for (int i = 0; i < blocks; ++i) {
            num += h_num[i];
            den += h_den[i];
            count += h_counts[i];
        }
        if (count == 0 || den == 0.0) {
            return 0.0f;
        }

        return static_cast<float>(std::sqrt(num / den));
    }

    // =============================================================
    // EXTRA REDUCTIONS (sum, sumsq, masked variants)
    // =============================================================

    namespace {

        // Kernel: sum(x)
        __global__
            void sum_kernel(const float* __restrict__ data,
                std::size_t n,
                double* __restrict__ block_sums)
        {
            __shared__ double sdata[BLOCK_SIZE];

            std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            double local = 0.0;

            for (; idx < n; idx += gridDim.x * blockDim.x) {
                local += static_cast<double>(data[idx]);
            }

            sdata[threadIdx.x] = local;
            __syncthreads();

            for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
                if (threadIdx.x < offset)
                    sdata[threadIdx.x] += sdata[threadIdx.x + offset];
                __syncthreads();
            }

            if (threadIdx.x == 0)
                block_sums[blockIdx.x] = sdata[0];
        }

        // Kernel: sum(x^2)
        __global__
            void sumsq_kernel(const float* __restrict__ data,
                std::size_t n,
                double* __restrict__ block_sums)
        {
            __shared__ double sdata[BLOCK_SIZE];

            std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            double local = 0.0;

            for (; idx < n; idx += gridDim.x * blockDim.x) {
                float v = data[idx];
                local += static_cast<double>(v) * static_cast<double>(v);
            }

            sdata[threadIdx.x] = local;
            __syncthreads();

            for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
                if (threadIdx.x < offset)
                    sdata[threadIdx.x] += sdata[threadIdx.x + offset];
                __syncthreads();
            }

            if (threadIdx.x == 0)
                block_sums[blockIdx.x] = sdata[0];
        }

        // Kernel: sum(|y - yhat| * mask)
        __global__
            void sum_abs_mask_kernel(const float* __restrict__ y,
                const float* __restrict__ yhat,
                const uint8_t* __restrict__ mask,
                std::size_t n,
                double* __restrict__ block_sums)
        {
            __shared__ double sdata[BLOCK_SIZE];

            std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            double local = 0.0;

            for (; idx < n; idx += gridDim.x * blockDim.x) {
                if (mask[idx]) {
                    float diff = y[idx] - yhat[idx];
                    local += fabsf(diff);
                }
            }

            sdata[threadIdx.x] = local;
            __syncthreads();

            for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
                if (threadIdx.x < offset)
                    sdata[threadIdx.x] += sdata[threadIdx.x + offset];
                __syncthreads();
            }

            if (threadIdx.x == 0)
                block_sums[blockIdx.x] = sdata[0];
        }

        // Kernel: sum((y - yhat)^2 * mask)
        __global__
            void sum_sq_mask_kernel(const float* __restrict__ y,
                const float* __restrict__ yhat,
                const uint8_t* __restrict__ mask,
                std::size_t n,
                double* __restrict__ block_sums)
        {
            __shared__ double sdata[BLOCK_SIZE];

            std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            double local = 0.0;

            for (; idx < n; idx += gridDim.x * blockDim.x) {
                if (mask[idx]) {
                    float diff = y[idx] - yhat[idx];
                    local += static_cast<double>(diff) * static_cast<double>(diff);
                }
            }

            sdata[threadIdx.x] = local;
            __syncthreads();

            for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
                if (threadIdx.x < offset)
                    sdata[threadIdx.x] += sdata[threadIdx.x + offset];
                __syncthreads();
            }

            if (threadIdx.x == 0)
                block_sums[blockIdx.x] = sdata[0];
        }

        // Kernel: sum(y^2 * mask)
        __global__
            void sum_sq_truth_mask_kernel(const float* __restrict__ y,
                const uint8_t* __restrict__ mask,
                std::size_t n,
                double* __restrict__ block_sums)
        {
            __shared__ double sdata[BLOCK_SIZE];

            std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            double local = 0.0;

            for (; idx < n; idx += gridDim.x * blockDim.x) {
                if (mask[idx]) {
                    float v = y[idx];
                    local += static_cast<double>(v) * static_cast<double>(v);
                }
            }

            sdata[threadIdx.x] = local;
            __syncthreads();

            for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
                if (threadIdx.x < offset)
                    sdata[threadIdx.x] += sdata[threadIdx.x + offset];
                __syncthreads();
            }

            if (threadIdx.x == 0)
                block_sums[blockIdx.x] = sdata[0];
        }

    } // anonymous namespace

    // =========================================================
    // HOST WRAPPERS FOR EXTRA REDUCTIONS
    // =========================================================

    float sum_reduction(const float* d_data,
        std::size_t n,
        cudaStream_t stream)
    {
        int blocks = std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 65535);
        double* d_block = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block, blocks * sizeof(double)));

        sum_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (d_data, n, d_block);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h(blocks);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), d_block, blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block));

        double total = 0.0;
        for (double v : h) total += v;
        return static_cast<float>(total);
    }

    float sumsq_reduction(const float* d_data,
        std::size_t n,
        cudaStream_t stream)
    {
        int blocks = std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 65535);
        double* d_block = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block, blocks * sizeof(double)));

        sumsq_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (d_data, n, d_block);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h(blocks);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), d_block, blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block));

        double total = 0.0;
        for (double v : h) total += v;
        return static_cast<float>(total);
    }

    float sum_abs_masked(const float* y,
        const float* yhat,
        const uint8_t* mask,
        std::size_t n,
        cudaStream_t stream)
    {
        int blocks = std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 65535);
        double* d_block = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block, blocks * sizeof(double)));

        sum_abs_mask_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (y, yhat, mask, n, d_block);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h(blocks);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), d_block, blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block));

        double total = 0.0;
        for (double v : h) total += v;
        return static_cast<float>(total);
    }

    float sum_sq_masked(const float* y,
        const float* yhat,
        const uint8_t* mask,
        std::size_t n,
        cudaStream_t stream)
    {
        int blocks = std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 65535);
        double* d_block = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block, blocks * sizeof(double)));

        sum_sq_mask_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (y, yhat, mask, n, d_block);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h(blocks);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), d_block, blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block));

        double total = 0.0;
        for (double v : h) total += v;
        return static_cast<float>(total);
    }

    float sum_sq_truth_masked(const float* y,
        const uint8_t* mask,
        std::size_t n,
        cudaStream_t stream)
    {
        int blocks = std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 65535);
        double* d_block = nullptr;
        CUDA_CHECK(cudaMalloc(&d_block, blocks * sizeof(double)));

        sum_sq_truth_mask_kernel << <blocks, BLOCK_SIZE, 0, stream >> > (y, mask, n, d_block);
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h(blocks);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), d_block, blocks * sizeof(double),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_block));

        double total = 0.0;
        for (double v : h) total += v;
        return static_cast<float>(total);
    }

} // namespace metrics
