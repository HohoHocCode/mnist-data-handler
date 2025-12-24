#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <optional>

struct BinMeta {
    int rows = 0;
    int cols = 0;
    std::string row_id;                 // "stay_id" hoặc ""
    std::vector<std::string> features;  // list feature columns
    std::string x_file;                 // e.g. "X_float32.bin"
    std::string m_file;                 // e.g. "M_uint8.bin"
    std::optional<std::string> row_ids_file; // null nếu không có
};

std::vector<uint8_t> read_u8(const std::string& path, size_t nbytes_expected = 0);
std::vector<int32_t> read_i32(const std::string& path, size_t nbytes_expected = 0);
std::vector<int64_t> read_i64(const std::string& path, size_t nbytes_expected = 0);
std::vector<float>   read_f32(const std::string& path, size_t nbytes_expected = 0);

// Đọc meta.json (rows/cols/features + tên file) bằng nlohmann/json
BinMeta read_meta_json(const std::string& meta_json_path);
