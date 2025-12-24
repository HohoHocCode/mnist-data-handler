#pragma once
#include "json.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;

struct DatasetMeta {
  int rows;
  int cols;
  std::string x_file;
  std::string m_file;
  std::vector<std::string> features;
};

inline DatasetMeta read_meta_json(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "Cannot open meta file: " << path << "\n";
    return {};
  }
  json j;
  f >> j;
  DatasetMeta m;
  m.rows = j.value("rows", 0);
  m.cols = j.value("cols", 0);
  if (j.contains("files")) {
    m.x_file = j["files"].value("X", "");
    m.m_file = j["files"].value("M", "");
  } else {
    m.x_file = j.value("x_file", "");
    m.m_file = j.value("m_file", "");
  }
  if (j.contains("features")) {
    m.features = j["features"].get<std::vector<std::string>>();
  }
  return m;
}

inline std::vector<float> read_f32(const std::string &path, size_t bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "Cannot open binary file: " << path << "\n";
    return {};
  }
  size_t count = bytes / sizeof(float);
  std::vector<float> data(count);
  f.read(reinterpret_cast<char *>(data.data()), bytes);
  return data;
}

inline std::vector<int32_t> read_i32(const std::string &path, size_t bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "Cannot open binary file: " << path << "\n";
    return {};
  }
  size_t count = bytes / sizeof(int32_t);
  std::vector<int32_t> data(count);
  f.read(reinterpret_cast<char *>(data.data()), bytes);
  return data;
}

inline std::vector<uint8_t> read_u8(const std::string &path, size_t bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "Cannot open binary file: " << path << "\n";
    return {};
  }
  size_t count = bytes / sizeof(uint8_t);
  std::vector<uint8_t> data(count);
  f.read(reinterpret_cast<char *>(data.data()), bytes);
  return data;
}
