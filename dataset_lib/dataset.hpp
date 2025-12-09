#pragma once
#include <string>
#include <vector>
#include <cuda_fp16.h>

// Simple holder for an imputation dataset on both host (float32) and device (FP16).
// This struct now owns its memory and frees it in the destructor (RAII-style)
// to avoid leaks on both CPU and GPU.
struct ImputeDataset {

    int rows = 0;
    int cols = 0;

    // CPU buffers
    float* truth_cpu = nullptr;
    float* miss_cpu = nullptr;
    float* mask_cpu = nullptr;

    // GPU buffers (half precision)
    __half* truth_gpu = nullptr;
    __half* miss_gpu = nullptr;
    __half* mask_gpu = nullptr;

    ImputeDataset() = default;

    // Non-copyable to avoid double-free
    ImputeDataset(const ImputeDataset&) = delete;
    ImputeDataset& operator=(const ImputeDataset&) = delete;

    // Movable
    ImputeDataset(ImputeDataset&& other) noexcept;
    ImputeDataset& operator=(ImputeDataset&& other) noexcept;

    ~ImputeDataset();

    void load(const std::string& truth_path,
        const std::string& miss_path,
        const std::string& mask_path);

    void load_to_gpu();
};
