// mimic_dataset.cpp
#include "mimic_dataset.hpp"
#include "npy_reader.hpp"

#include <iostream>
#include <stdexcept>

MimicDataset MimicLoader::load_from_npy(
    const std::string& dir,
    const std::string& truth_name,
    const std::string& miss_name,
    const std::string& mask_name)
{
    MimicDataset ds;

    std::string path_truth = dir + "/" + truth_name;
    std::string path_miss = dir + "/" + miss_name;
    std::string path_mask = dir + "/" + mask_name;

    int r1, c1, r2, c2, r3, c3;

    if (!load_npy_float32(path_truth, ds.truth, r1, c1))
        throw std::runtime_error("Failed to load " + path_truth);

    if (!load_npy_float32(path_miss, ds.miss, r2, c2))
        throw std::runtime_error("Failed to load " + path_miss);

    if (!load_npy_uint8(path_mask, ds.mask, r3, c3))
        throw std::runtime_error("Failed to load " + path_mask);

    if (r1 != r2 || c1 != c2 || r1 != r3 || c1 != c3) {
        throw std::runtime_error("Mismatched shapes in MIMIC npy files");
    }

    ds.rows = r1;
    ds.cols = c1;

    std::cout << "[MIMIC] Loaded NPY: " << ds.rows << " rows x "
        << ds.cols << " cols\n";

    return ds;
}
