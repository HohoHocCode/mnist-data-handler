#pragma once
#include <string>
#include <vector>

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

    void load(const std::string& truth_path,
        const std::string& miss_path,
        const std::string& mask_path);

    void load_to_gpu();
};
