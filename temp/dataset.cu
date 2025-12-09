#include "dataset.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <npy.hpp>

template <typename T>
T* load_npy(const std::string& path, int& rows, int& cols) {
    std::vector<unsigned long> shape;
    std::vector<T> data;

    npy::LoadArrayFromNumpy(path, shape, data);
    rows = (int)shape[0];
    cols = (int)shape[1];

    T* ptr = new T[rows * cols];
    memcpy(ptr, data.data(), sizeof(T) * rows * cols);
    return ptr;
}

void ImputeDataset::load(const std::string& truth_path,
    const std::string& miss_path,
    const std::string& mask_path)
{
    int r1, c1;
    int r2, c2;
    int r3, c3;

    truth_cpu = load_npy<float>(truth_path, r1, c1);
    miss_cpu = load_npy<float>(miss_path, r2, c2);
    mask_cpu = load_npy<float>(mask_path, r3, c3);

    if (r1 != r2 || r1 != r3 || c1 != c2 || c1 != c3) {
        throw std::runtime_error("Dataset shape mismatch!");
    }

    rows = r1;
    cols = c1;

    printf("[dataset] Loaded: %d x %d\n", rows, cols);
}

__global__ void float_to_half(__half* out, const float* in, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = __float2half(in[idx]);
}

void ImputeDataset::load_to_gpu()
{
    int N = rows * cols;

    cudaMalloc(&truth_gpu, sizeof(__half) * N);
    cudaMalloc(&miss_gpu, sizeof(__half) * N);
    cudaMalloc(&mask_gpu, sizeof(__half) * N);

    float* tmp_truth; cudaMalloc(&tmp_truth, sizeof(float) * N);
    float* tmp_miss;  cudaMalloc(&tmp_miss, sizeof(float) * N);
    float* tmp_mask;  cudaMalloc(&tmp_mask, sizeof(float) * N);

    cudaMemcpy(tmp_truth, truth_cpu, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_miss, miss_cpu, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_mask, mask_cpu, sizeof(float) * N, cudaMemcpyHostToDevice);

    dim3 blk(256);
    dim3 grd((N + blk.x - 1) / blk.x);

    float_to_half << <grd, blk >> > (truth_gpu, tmp_truth, N);
    float_to_half << <grd, blk >> > (miss_gpu, tmp_miss, N);
    float_to_half << <grd, blk >> > (mask_gpu, tmp_mask, N);

    cudaFree(tmp_truth);
    cudaFree(tmp_miss);
    cudaFree(tmp_mask);

    printf("[dataset] Copied to GPU.\n");
}
