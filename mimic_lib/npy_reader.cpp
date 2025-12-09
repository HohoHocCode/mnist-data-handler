// npy_reader.cpp (bản sửa)

#define _CRT_SECURE_NO_WARNINGS
#include "npy_reader.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>   // memcpy
#include <cstdint>

namespace {

    // đọc toàn bộ file vào vector<char>
    std::vector<char> read_file(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("Cannot open npy file: " + path);
        }
        f.seekg(0, std::ios::end);
        std::streamsize size = f.tellg();
        f.seekg(0, std::ios::beg);

        std::vector<char> buffer((size_t)size);
        if (!f.read(buffer.data(), size)) {
            throw std::runtime_error("Error reading npy file: " + path);
        }
        return buffer;
    }

    struct NpyHeader {
        std::string descr;       // "<f4", "|u1", ...
        bool        fortran_order;
        int         rows;
        int         cols;
        std::size_t header_len;
        std::size_t header_start;
    };

    // parse header NPY (hỗ trợ v1, v2, v3)
    NpyHeader parse_header(const std::vector<char>& buf) {
        const char magic[] = "\x93NUMPY";
        for (int i = 0; i < 6; ++i) {
            if (buf[i] != magic[i]) {
                throw std::runtime_error("Not a npy file (magic mismatch)");
            }
        }

        uint8_t major = static_cast<uint8_t>(buf[6]);
        uint8_t minor = static_cast<uint8_t>(buf[7]);

        std::size_t header_start = 0;
        std::size_t header_len = 0;

        if (major == 1) {
            // v1.0: header_len là uint16 LE bắt đầu từ offset 8
            uint16_t hl16 = 0;
            std::memcpy(&hl16, &buf[8], sizeof(uint16_t));
            header_len = hl16;
            header_start = 10;
        }
        else if (major == 2 || major == 3) {
            // v2.0, v3.0: header_len là uint32 LE bắt đầu từ offset 8
            uint32_t hl32 = 0;
            std::memcpy(&hl32, &buf[8], sizeof(uint32_t));
            header_len = hl32;
            header_start = 12;
        }
        else {
            throw std::runtime_error("Unsupported npy version");
        }

        std::size_t header_end = header_start + header_len;
        if (header_end > buf.size()) {
            throw std::runtime_error("Invalid header length");
        }

        std::string header(buf.begin() + header_start, buf.begin() + header_end);

        NpyHeader h{};
        h.fortran_order = false;
        h.rows = h.cols = 0;
        h.header_len = header_len;
        h.header_start = header_start;

        // helper: lấy value dạng string cho key, vd "descr", "fortran_order"
        auto find_value_string = [&](const std::string& key) -> std::string {
            std::size_t pos = header.find(key);
            if (pos == std::string::npos) return {};

            // nhảy tới dấu ':' sau key
            pos = header.find(':', pos);
            if (pos == std::string::npos) return {};

            // tìm dấu nháy đơn đầu tiên sau dấu ':'
            pos = header.find('\'', pos);
            if (pos == std::string::npos) return {};

            std::size_t end = header.find('\'', pos + 1);
            if (end == std::string::npos) return {};

            return header.substr(pos + 1, end - (pos + 1));
            };

        // ---- descr ----
        h.descr = find_value_string("descr");
        if (h.descr.empty()) {
            throw std::runtime_error("npy header: missing 'descr'");
        }

        // ---- fortran_order ----
        {
            std::size_t pos = header.find("fortran_order");
            if (pos == std::string::npos)
                throw std::runtime_error("npy header: missing 'fortran_order'");

            std::size_t true_pos = header.find("True", pos);
            std::size_t false_pos = header.find("False", pos);

            if (false_pos != std::string::npos &&
                (true_pos == std::string::npos || false_pos < true_pos)) {
                h.fortran_order = false;
            }
            else {
                h.fortran_order = true;
            }
        }

        // ---- shape = (rows, cols) ----
        {
            std::size_t pos = header.find("shape");
            if (pos == std::string::npos)
                throw std::runtime_error("npy header: missing 'shape'");

            std::size_t p1 = header.find('(', pos);
            std::size_t p2 = header.find(')', p1);
            if (p1 == std::string::npos || p2 == std::string::npos)
                throw std::runtime_error("npy header: invalid shape format");

            std::string shape_str = header.substr(p1 + 1, p2 - p1 - 1);

            int r = 0, c = 0;
            if (std::sscanf(shape_str.c_str(), "%d, %d", &r, &c) != 2) {
                // trường hợp (N,) 1D
                if (std::sscanf(shape_str.c_str(), "%d,", &r) != 1) {
                    throw std::runtime_error("npy header: cannot parse shape");
                }
                c = 1;
            }

            h.rows = r;
            h.cols = c;
        }

        return h;
    }

} // anonymous namespace

// ======================================================
//  float32
// ======================================================
bool load_npy_float32(
    const std::string& path,
    std::vector<float>& out,
    int& rows,
    int& cols)
{
    try {
        auto buf = read_file(path);
        NpyHeader h = parse_header(buf);

        // kiểm tra dtype
        if (h.descr != "<f4" && h.descr != "|f4") {
            throw std::runtime_error("Expected float32 dtype in " + path +
                " (got '" + h.descr + "')");
        }

        std::size_t data_offset = h.header_start + h.header_len;
        std::size_t n_elem = static_cast<std::size_t>(h.rows) * h.cols;
        std::size_t bytes = n_elem * sizeof(float);

        if (data_offset + bytes > buf.size()) {
            throw std::runtime_error("npy file truncated: " + path);
        }

        // đọc raw data
        std::vector<float> tmp(n_elem);
        std::memcpy(tmp.data(), buf.data() + data_offset, bytes);

        out.resize(n_elem);

        if (!h.fortran_order) {
            // C-order: copy thẳng
            std::memcpy(out.data(), tmp.data(), bytes);
        }
        else {
            // Fortran-order: chuyển thành row-major
            int R = h.rows;
            int C = h.cols;
            for (int j = 0; j < C; ++j) {
                for (int i = 0; i < R; ++i) {
                    std::size_t src = static_cast<std::size_t>(j) * R + i;
                    std::size_t dst = static_cast<std::size_t>(i) * C + j;
                    out[dst] = tmp[src];
                }
            }
        }

        rows = h.rows;
        cols = h.cols;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[npy] " << e.what() << "\n";
        return false;
    }
}

// ======================================================
//  uint8
// ======================================================
bool load_npy_uint8(
    const std::string& path,
    std::vector<uint8_t>& out,
    int& rows,
    int& cols)
{
    try {
        auto buf = read_file(path);
        NpyHeader h = parse_header(buf);

        if (h.descr != "|u1" && h.descr != "<u1") {
            throw std::runtime_error("Expected uint8 dtype in " + path +
                " (got '" + h.descr + "')");
        }

        std::size_t data_offset = h.header_start + h.header_len;
        std::size_t n_elem = static_cast<std::size_t>(h.rows) * h.cols;
        std::size_t bytes = n_elem * sizeof(uint8_t);

        if (data_offset + bytes > buf.size()) {
            throw std::runtime_error("npy file truncated: " + path);
        }

        std::vector<uint8_t> tmp(n_elem);
        std::memcpy(tmp.data(), buf.data() + data_offset, bytes);

        out.resize(n_elem);

        if (!h.fortran_order) {
            std::memcpy(out.data(), tmp.data(), bytes);
        }
        else {
            int R = h.rows;
            int C = h.cols;
            for (int j = 0; j < C; ++j) {
                for (int i = 0; i < R; ++i) {
                    std::size_t src = static_cast<std::size_t>(j) * R + i;
                    std::size_t dst = static_cast<std::size_t>(i) * C + j;
                    out[dst] = tmp[src];
                }
            }
        }

        rows = h.rows;
        cols = h.cols;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[npy] " << e.what() << "\n";
        return false;
    }
}
