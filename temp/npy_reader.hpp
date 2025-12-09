// npy_reader.hpp
#pragma once
#include <string>
#include <vector>
#include <cstdint>

bool load_npy_float32(
    const std::string& path,
    std::vector<float>& out,
    int& rows,
    int& cols
);

bool load_npy_uint8(
    const std::string& path,
    std::vector<uint8_t>& out,
    int& rows,
    int& cols
);
