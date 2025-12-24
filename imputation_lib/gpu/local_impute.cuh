#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace impute {

    // =============== 1. K-NEAREST NEIGHBOR FAMILY ===============

    // KNNimpute (bạn đã có bản GPU trong project, sau này chỉ cần nối hàm này vào)
    void knn_impute(
        float* d_X,      // [N x D], row-major, trên GPU
        uint8_t* d_mask,   // [N x D], 0 = observed, 1 = missing
        int      N,        // số gene (rows)
        int      D,        // số điều kiện (columns)
        int      K,        // số láng giềng
        cudaStream_t stream
    );

    // Sequential KNN (SKNNimpute)
    void sknn_impute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      K,
        cudaStream_t stream
    );

    // Iterative KNN (IKNNimpute)
    void iknn_impute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      K,
        int      max_iter,
        float    tol,
        cudaStream_t stream
    );

    // =============== 2. LEAST SQUARES / LOCAL LEAST SQUARES FAMILY ===============

    // LSimpute – single regression (LSimpute_gene/array trong bài)
    void lsimpute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      K,          // số gene tham chiếu
        cudaStream_t stream
    );

    // LLSimpute – multiple regression (cái này gần với ls_impute hiện tại của bạn)
    void llsimpute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      K,
        cudaStream_t stream
    );

    // SLLSimpute – Sequential LLS (reuse gene đã impute nếu missing rate < threshold)
    void sllsimpute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      K,
        float    reuse_threshold,   // ví dụ 0.2 => chỉ reuse gene có missing rate < 20%
        cudaStream_t stream
    );

    // ILLSimpute – Iterated LLS, chọn K bằng ngưỡng khoảng cách
    void illsimpute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      max_iter,
        float    dist_factor,       // hệ số nhân với khoảng cách trung bình để chọn neighbor
        cudaStream_t stream
    );

    // RLSP – Robust LS với principal components (LLS + PCA)
    void rlsp_impute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      K,         // số neighbor
        int      n_pc,      // số principal components dùng
        cudaStream_t stream
    );

    // =============== 3. NHÓM LOCAL NÂNG CAO (HIỆN TẠI CHỈ LÀ SKELETON) ===============

    // Gaussian Mixture Clustering imputation (GMCimpute)
    void gmc_impute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      n_components,    // số mixture components
        cudaStream_t stream
    );

    // Regression with Bayesian Gene Selection (BGSregress)
    void bgs_regress_impute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      max_genes,      // tối đa số gene được chọn trong regression model
        cudaStream_t stream
    );

    // Collateral Missing Value Estimation (CMVE)
    void cmve_impute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      K,
        cudaStream_t stream
    );

    // Ameliorative Missing Value Imputation (AMVI)
    void amvi_impute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        cudaStream_t stream
    );

    // Autoregressive LS imputation (ARLSimpute) – dành cho time series
    void arls_impute(
        float* d_X,
        uint8_t* d_mask,
        int      N,
        int      D,
        int      K,           // số gene tương đồng (neighbors)
        int      ar_order,    // bậc AR(p)
        cudaStream_t stream
    );

} // namespace impute
