#include "dataset.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <npy.hpp>
#include <vector>
#include <cstring>
#include <cstdio>
#include <stdexcept>

// Simple CUDA error check helper for this translation unit
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error: %s at %s:%d\n",                 \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            throw std::runtime_error("CUDA error in ImputeDataset");          \
        }                                                                     \
    } while (0)

template <typename T>
T* load_npy(const std::string& path, int& rows, int& cols) {
    std::vector<unsigned long> shape;
    std::vector<T> data;

    npy::LoadArrayFromNumpy(path, shape, data);
    rows = static_cast<int>(shape[0]);
    cols = static_cast<int>(shape[1]);

    T* ptr = new T[rows * cols];
    std::memcpy(ptr, data.data(), sizeof(T) * rows * cols);
    return ptr;
}

// ---------------- ImputeDataset: RAII implementation ----------------

ImputeDataset::~ImputeDataset()
{
    // Free host buffers
    delete[] truth_cpu;
    delete[] miss_cpu;
    delete[] mask_cpu;
    truth_cpu = miss_cpu = mask_cpu = nullptr;

    // Free device buffers
    if (truth_gpu) CUDA_CHECK(cudaFree(truth_gpu));
    if (miss_gpu)  CUDA_CHECK(cudaFree(miss_gpu));
    if (mask_gpu)  CUDA_CHECK(cudaFree(mask_gpu));
    truth_gpu = miss_gpu = mask_gpu = nullptr;
}

ImputeDataset::ImputeDataset(ImputeDataset&& other) noexcept
{
    rows = other.rows;
    cols = other.cols;
    truth_cpu = other.truth_cpu;
    miss_cpu = other.miss_cpu;
    mask_cpu = other.mask_cpu;
    truth_gpu = other.truth_gpu;
    miss_gpu = other.miss_gpu;
    mask_gpu = other.mask_gpu;

    other.rows = other.cols = 0;
    other.truth_cpu = other.miss_cpu = other.mask_cpu = nullptr;
    other.truth_gpu = other.miss_gpu = other.mask_gpu = nullptr;
}

ImputeDataset& ImputeDataset::operator=(ImputeDataset&& other) noexcept
{
    if (this != &other) {
        // Clean up current resources
        this->~ImputeDataset();

        rows = other.rows;
        cols = other.cols;
        truth_cpu = other.truth_cpu;
        miss_cpu = other.miss_cpu;
        mask_cpu = other.mask_cpu;
        truth_gpu = other.truth_gpu;
        miss_gpu = other.miss_gpu;
        mask_gpu = other.mask_gpu;

        other.rows = other.cols = 0;
        other.truth_cpu = other.miss_cpu = other.mask_cpu = nullptr;
        other.truth_gpu = other.miss_gpu = other.mask_gpu = nullptr;
    }
    return *this;
}

void ImputeDataset::load(const std::string& truth_path,
    const std::string& miss_path,
    const std::string& mask_path)
{
    int r1, c1;
    int r2, c2;
    int r3, c3;

    // Clean up any existing CPU data first
    delete[] truth_cpu;
    delete[] miss_cpu;
    delete[] mask_cpu;
    truth_cpu = miss_cpu = mask_cpu = nullptr;

    truth_cpu = load_npy<float>(truth_path, r1, c1);
    miss_cpu = load_npy<float>(miss_path, r2, c2);
    mask_cpu = load_npy<float>(mask_path, r3, c3);

    if (r1 != r2 || r1 != r3 || c1 != c2 || c1 != c3) {
        throw std::runtime_error("Dataset shape mismatch!");
    }

    rows = r1;
    cols = c1;

    std::printf("[dataset] Loaded: %d x %d\n", rows, cols);
}

__global__ void float_to_half(__half* out, const float* in, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = __float2half(in[idx]);
}

void ImputeDataset::load_to_gpu()
{
    const int N = rows * cols;
    if (N <= 0) {
        throw std::runtime_error("ImputeDataset::load_to_gpu: dataset is empty");
    }

    // Free any existing device buffers before reallocating
    if (truth_gpu) CUDA_CHECK(cudaFree(truth_gpu));
    if (miss_gpu)  CUDA_CHECK(cudaFree(miss_gpu));
    if (mask_gpu)  CUDA_CHECK(cudaFree(mask_gpu));
    truth_gpu = miss_gpu = mask_gpu = nullptr;

    CUDA_CHECK(cudaMalloc(&truth_gpu, sizeof(__half) * N));
    CUDA_CHECK(cudaMalloc(&miss_gpu, sizeof(__half) * N));
    CUDA_CHECK(cudaMalloc(&mask_gpu, sizeof(__half) * N));

    float* tmp_truth = nullptr;
    float* tmp_miss = nullptr;
    float* tmp_mask = nullptr;

    CUDA_CHECK(cudaMalloc(&tmp_truth, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&tmp_miss, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&tmp_mask, sizeof(float) * N));

    CUDA_CHECK(cudaMemcpy(tmp_truth, truth_cpu,
        sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_miss, miss_cpu,
        sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_mask, mask_cpu,
        sizeof(float) * N, cudaMemcpyHostToDevice));

    dim3 blk(256);
    dim3 grd((N + blk.x - 1) / blk.x);

    float_to_half << <grd, blk >> > (truth_gpu, tmp_truth, N);
    CUDA_CHECK(cudaGetLastError());

    float_to_half << <grd, blk >> > (miss_gpu, tmp_miss, N);
    CUDA_CHECK(cudaGetLastError());

    float_to_half << <grd, blk >> > (mask_gpu, tmp_mask, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(tmp_truth));
    CUDA_CHECK(cudaFree(tmp_miss));
    CUDA_CHECK(cudaFree(tmp_mask));

    std::printf("[dataset] Copied to GPU (FP16).\n");
}
