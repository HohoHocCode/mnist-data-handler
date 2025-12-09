#include "local_impute.cuh"

// Include các file gốc trong project của bạn
#include "knn_impute.cuh"
#include "iknn_impute.cuh"
#include "ls_impute.cuh"
#include "svd_impute.cuh"
#include "sknn_impute.cuh"
#include "lls_impute.cuh"

// ======================================
// IMPLEMENTATION CHO CÁC HÀM LOCAL
// ======================================

namespace impute {

    //
    // KNNimpute — bản gốc của bạn đã có trong knn_impute.cu
    //
    void knn_impute(float* d_X, uint8_t* d_mask,
        int N, int D, int K,
        cudaStream_t stream)
    {
        // Gọi đúng hàm hiện có trong project của bạn
        knn_impute(d_X, d_mask, N, D, K, stream);
    }



    //
    // IKNNimpute — bản gốc có trong iknn_impute.cu
    //
    void iknn_impute(float* d_X, uint8_t* d_mask,
        int N, int D, int K,
        int max_iter, float tol,
        cudaStream_t stream)
    {
        iknn_impute(d_X, d_mask, N, D, K, max_iter, tol, stream);
    }

    void sknn_impute(float* d_X, uint8_t* d_mask,
        int N, int D, int K,
        cudaStream_t stream)
    {
        sknn_impute_cuda(d_X, d_mask, N, D, K, stream);
    }

    void lsimpute(float* d_X, uint8_t* d_mask,
        int N, int D, int K,
        cudaStream_t stream)
    {
        lsimpute(d_X, d_mask, N, D, K, stream);
    }

    void llsimpute(float* d_X, uint8_t* d_mask,
        int N, int D, int K,
        cudaStream_t stream)
    {
        // gọi bản CUDA đã tối ưu của bạn
        lls_impute_cuda(d_X, d_mask, N, D, K, stream);
    }

    //
    // Các hàm khác chưa implement — stub để linker không báo lỗi
    //
    void sllsimpute(float*, uint8_t*, int, int, int, float, cudaStream_t) {}
    void illsimpute(float*, uint8_t*, int, int, int, float, cudaStream_t) {}
    void rlsp_impute(float*, uint8_t*, int, int, int, int, cudaStream_t) {}

    void gmc_impute(float*, uint8_t*, int, int, int, cudaStream_t) {}
    void bgs_regress_impute(float*, uint8_t*, int, int, int, cudaStream_t) {}
    void cmve_impute(float*, uint8_t*, int, int, int, cudaStream_t) {}
    void amvi_impute(float*, uint8_t*, int, int, cudaStream_t) {}
    void arls_impute(float*, uint8_t*, int, int, int, int, cudaStream_t) {}

} // namespace impute
