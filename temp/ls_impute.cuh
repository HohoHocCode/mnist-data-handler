#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

/**
 * LS-Imputation (Local Least Squares)
 * Theo paper:
 *   "Missing value imputation for gene expression data"
 *   Troyanskaya et al., 2001.
 *
 * Ý tưởng:
 * - Tìm K hàng gần nhất (dữ liệu không missing tại cột bị thiếu).
 * - Giải hệ least-squares để tìm trọng số w:
 *       w = (A^T A)^(-1) A^T y
 * - Dự đoán giá trị missing qua:
 *       x̂ = a_i ⋅ w
 */
namespace impute_ls {

    /**
     * @brief Thực hiện LS-imputation cho toàn bộ ma trận.
     *
     * @param d_data    N×D dữ liệu (float), chứa NaN tại vị trí missing
     * @param d_mask    N×D mask: 1 = missing, 0 = observed
     * @param N         số hàng
     * @param D         số cột
     * @param K         số hàng gần nhất dùng để ước lượng LS
     * @param stream    CUDA stream
     */
    void ls_impute(
        float* d_data,
        const uint8_t* d_mask,
        int N, int D,
        int K,
        cudaStream_t stream = 0
    );

}
