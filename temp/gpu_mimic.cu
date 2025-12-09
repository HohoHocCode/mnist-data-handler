// gpu_mimic.cu
#include "gpu_mimic.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>

#define BLOCK_SIZE 256

// =====================
// kernels
// =====================

__global__ void convert_fp32_fp16(const float* in, __half* out, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        out[idx] = __float2half(in[idx]);
}

// Tính mean & std trên FP16 (mỗi block = 1 feature)
__global__ void compute_mean_std_fp16(
    const __half* __restrict__ X,
    float* __restrict__ means,
    float* __restrict__ stds,
    int N, int D)
{
    extern __shared__ float s[];
    float* s_sum = s;
    float* s_sumsq = s + blockDim.x;

    int feat = blockIdx.x;
    int tid = threadIdx.x;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        int idx = i * D + feat;
        float v = __half2float(X[idx]);
        local_sum += v;
        local_sumsq += v * v;
    }

    s_sum[tid] = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mu = s_sum[0] / N;
        float var = s_sumsq[0] / N - mu * mu;
        if (var < 1e-6f) var = 1e-6f;
        means[feat] = mu;
        stds[feat] = sqrtf(var);
    }
}

// Chuẩn hóa FP16 theo mean/std (mỗi thread = 1 phần tử)
__global__ void normalize_fp16(
    __half* __restrict__ X,
    const float* __restrict__ means,
    const float* __restrict__ stds,
    int N, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D;
    if (idx >= total) return;

    int feat = idx % D;
    float v = __half2float(X[idx]);
    float nv = (v - means[feat]) / stds[feat];
    X[idx] = __float2half(nv);
}

// =====================
// GpuMimicMatrix impl
// =====================

GpuMimicMatrix::GpuMimicMatrix()
{
    cudaStreamCreate(&stream);
}

GpuMimicMatrix::~GpuMimicMatrix()
{
    if (d_truth_fp16) cudaFree(d_truth_fp16);
    if (d_miss_fp16)  cudaFree(d_miss_fp16);
    if (d_mask_u8)    cudaFree(d_mask_u8);
    if (d_means)      cudaFree(d_means);
    if (d_stds)       cudaFree(d_stds);
    if (stream)       cudaStreamDestroy(stream);
}

GpuMimicMatrix::GpuMimicMatrix(GpuMimicMatrix&& other) noexcept
{
    *this = std::move(other);
}

GpuMimicMatrix& GpuMimicMatrix::operator=(GpuMimicMatrix&& other) noexcept
{
    if (this != &other) {
        this->~GpuMimicMatrix();
        n_rows = other.n_rows;
        n_cols = other.n_cols;
        d_truth_fp16 = other.d_truth_fp16;
        d_miss_fp16 = other.d_miss_fp16;
        d_mask_u8 = other.d_mask_u8;
        d_means = other.d_means;
        d_stds = other.d_stds;
        stream = other.stream;

        other.n_rows = 0;
        other.n_cols = 0;
        other.d_truth_fp16 = nullptr;
        other.d_miss_fp16 = nullptr;
        other.d_mask_u8 = nullptr;
        other.d_means = nullptr;
        other.d_stds = nullptr;
        other.stream = nullptr;
    }
    return *this;
}

void GpuMimicMatrix::upload(const MimicDataset& ds)
{
    n_rows = ds.rows;
    n_cols = ds.cols;

    int total = n_rows * n_cols;
    size_t bytes_fp16 = sizeof(__half) * (size_t)total;
    size_t bytes_u8 = sizeof(uint8_t) * (size_t)total;

    if (!d_truth_fp16) cudaMalloc(&d_truth_fp16, bytes_fp16);
    if (!d_miss_fp16)  cudaMalloc(&d_miss_fp16, bytes_fp16);
    if (!d_mask_u8)    cudaMalloc(&d_mask_u8, bytes_u8);

    // CPU tmp: float32
    float* d_tmp_truth = nullptr;
    float* d_tmp_miss = nullptr;

    size_t bytes_f32 = sizeof(float) * (size_t)total;
    cudaMalloc(&d_tmp_truth, bytes_f32);
    cudaMalloc(&d_tmp_miss, bytes_f32);

    // copy host → device (float32)
    cudaMemcpyAsync(d_tmp_truth, ds.truth.data(), bytes_f32,
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tmp_miss, ds.miss.data(), bytes_f32,
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_mask_u8, ds.mask.data(), bytes_u8,
        cudaMemcpyHostToDevice, stream);

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // float32 → fp16
    convert_fp32_fp16 << <blocks, BLOCK_SIZE, 0, stream >> > (d_tmp_truth, d_truth_fp16, total);
    convert_fp32_fp16 << <blocks, BLOCK_SIZE, 0, stream >> > (d_tmp_miss, d_miss_fp16, total);

    cudaStreamSynchronize(stream);

    cudaFree(d_tmp_truth);
    cudaFree(d_tmp_miss);

    std::cout << "[GPU] Uploaded MIMIC: " << n_rows << " x " << n_cols << "\n";
}

void GpuMimicMatrix::normalize_miss()
{
    if (!d_miss_fp16 || n_rows == 0 || n_cols == 0) {
        std::cout << "[GPU] normalize_miss skipped\n";
        return;
    }

    if (!d_means) cudaMalloc(&d_means, sizeof(float) * n_cols);
    if (!d_stds)  cudaMalloc(&d_stds, sizeof(float) * n_cols);

    dim3 grid_feat(n_cols);
    dim3 block(BLOCK_SIZE);
    size_t shmem = 2 * BLOCK_SIZE * sizeof(float);

    compute_mean_std_fp16 << <grid_feat, block, shmem, stream >> > (
        d_miss_fp16, d_means, d_stds, n_rows, n_cols
        );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "[CUDA] compute_mean_std_fp16: "
            << cudaGetErrorString(err) << "\n";
    }

    int total = n_rows * n_cols;
    int blocksVal = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    normalize_fp16 << <blocksVal, BLOCK_SIZE, 0, stream >> > (
        d_miss_fp16, d_means, d_stds, n_rows, n_cols
        );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "[CUDA] normalize_fp16: "
            << cudaGetErrorString(err) << "\n";
    }

    cudaStreamSynchronize(stream);
    std::cout << "[GPU] normalize_miss done\n";
}
