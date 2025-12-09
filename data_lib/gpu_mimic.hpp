// gpu_mimic.hpp
#pragma once

#include "mimic_dataset.hpp"
#include <cuda_fp16.h>
#include <cstdint>

// Quản lý dữ liệu MIMIC trên GPU (FP16)
class GpuMimicMatrix {
public:
    GpuMimicMatrix();
    ~GpuMimicMatrix();

    // không copy
    GpuMimicMatrix(const GpuMimicMatrix&) = delete;
    GpuMimicMatrix& operator=(const GpuMimicMatrix&) = delete;

    // cho phép move nếu cần
    GpuMimicMatrix(GpuMimicMatrix&&) noexcept;
    GpuMimicMatrix& operator=(GpuMimicMatrix&&) noexcept;

    void upload(const MimicDataset& ds);  // CPU → GPU
    void normalize_miss();                // chuẩn hóa d_miss inplace

    int rows() const { return n_rows; }
    int cols() const { return n_cols; }

    // truy cập buffer GPU (cho imputation_lib)
    __half*  d_truth() { return d_truth_fp16; }
    __half*  d_miss()  { return d_miss_fp16; }
    uint8_t* d_mask()  { return d_mask_u8;   }

private:
    int n_rows = 0;
    int n_cols = 0;

    __half*  d_truth_fp16 = nullptr;
    __half*  d_miss_fp16  = nullptr;
    uint8_t* d_mask_u8    = nullptr;

    float* d_means = nullptr;
    float* d_stds  = nullptr;

    cudaStream_t stream = nullptr;
};
