#include "cuda_bin_io.hpp"
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstring> // memcpy
#include <nlohmann/json.hpp>

static std::vector<uint8_t> read_all_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t n = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(n);
    f.read((char*)buf.data(), (std::streamsize)n);
    return buf;
}

std::vector<uint8_t> read_u8(const std::string& path, size_t nbytes_expected) {
    auto b = read_all_bytes(path);
    if (nbytes_expected && b.size() != nbytes_expected)
        throw std::runtime_error("Size mismatch: " + path);
    return b;
}

std::vector<int32_t> read_i32(const std::string& path, size_t nbytes_expected) {
    auto b = read_all_bytes(path);
    if (nbytes_expected && b.size() != nbytes_expected)
        throw std::runtime_error("Size mismatch: " + path);
    if (b.size() % sizeof(int32_t)) throw std::runtime_error("Bad i32 size: " + path);
    std::vector<int32_t> v(b.size() / sizeof(int32_t));
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

std::vector<int64_t> read_i64(const std::string& path, size_t nbytes_expected) {
    auto b = read_all_bytes(path);
    if (nbytes_expected && b.size() != nbytes_expected)
        throw std::runtime_error("Size mismatch: " + path);
    if (b.size() % sizeof(int64_t)) throw std::runtime_error("Bad i64 size: " + path);
    std::vector<int64_t> v(b.size() / sizeof(int64_t));
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

std::vector<float> read_f32(const std::string& path, size_t nbytes_expected) {
    auto b = read_all_bytes(path);
    if (nbytes_expected && b.size() != nbytes_expected)
        throw std::runtime_error("Size mismatch: " + path);
    if (b.size() % sizeof(float)) throw std::runtime_error("Bad f32 size: " + path);
    std::vector<float> v(b.size() / sizeof(float));
    std::memcpy(v.data(), b.data(), b.size());
    return v;
}

static nlohmann::json load_json_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open json: " + path);
    nlohmann::json j;
    f >> j;
    return j;
}

BinMeta read_meta_json(const std::string& meta_json_path) {
    auto j = load_json_file(meta_json_path);

    BinMeta m;
    m.rows = j.at("rows").get<int>();
    m.cols = j.at("cols").get<int>();

    if (j.contains("row_id") && !j["row_id"].is_null())
        m.row_id = j["row_id"].get<std::string>();

    if (j.contains("features"))
        m.features = j["features"].get<std::vector<std::string>>();

    // meta.json của bạn có block "files": {"X": "...", "M": "...", "row_ids": ...}
    if (j.contains("files")) {
        const auto& f = j["files"];
        if (f.contains("X")) m.x_file = f["X"].get<std::string>();
        if (f.contains("M")) m.m_file = f["M"].get<std::string>();
        if (f.contains("row_ids") && !f["row_ids"].is_null())
            m.row_ids_file = f["row_ids"].get<std::string>();
    }

    // fallback nếu meta cũ không có "files"
    if (m.x_file.empty() && j.contains("format")) m.x_file = "X_float32.bin";
    if (m.m_file.empty() && j.contains("format")) m.m_file = "M_uint8.bin";

    return m;
}
