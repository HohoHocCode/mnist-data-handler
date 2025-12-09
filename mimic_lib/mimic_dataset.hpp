// mimic_dataset.hpp
#pragma once
#include <vector>
#include <cstdint>
#include <string>

struct MimicDataset {
    int rows = 0;
    int cols = 0;

    // N * D, row-major
    std::vector<float> truth;   // matrix_truth.npy
    std::vector<float> miss;    // matrix_miss_X.npy
    std::vector<uint8_t> mask;  // mask_X.npy
};

// Loader chính: đọc 3 file npy trong 1 thư mục
class MimicLoader {
public:
    // dir: thư mục chứa file npy
    // miss_name: ví dụ "matrix_miss_10.npy"
    // mask_name: ví dụ "mask_10.npy"
    static MimicDataset load_from_npy(
        const std::string& dir,
        const std::string& truth_name,
        const std::string& miss_name,
        const std::string& mask_name
    );
};
