#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "gpu_mimic.hpp"
#include "mimic_dataset.hpp"

// Cổng thống nhất cho các thuật toán local (KNN, iKNN, SKNN, LLS, ...)
#include "local_impute.cuh"

// SVD-impute (global)
#include "svd_impute.cuh"

// Metrics
#include "metrics.cuh"

// FP16 → FP32
extern "C" void convert_half_to_float(float* out, const __half* in, int n);

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(err)          \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                        \
    }                                                                   \
} while(0)


// ======================
// Hàm in METRICS
// ======================
void print_metrics(
    const char* name,
    const float* d_truth,
    const float* d_pred,
    const uint8_t* d_mask_eval,  // mask gốc, 1 = missing
    int N, int D)
{
    size_t n = static_cast<size_t>(N) * static_cast<size_t>(D);

    float mae = metrics::mae_masked(d_truth, d_pred, d_mask_eval, n);
    float rmse = metrics::rmse_masked(d_truth, d_pred, d_mask_eval, n);
    float nrmse = metrics::nrmse_masked(d_truth, d_pred, d_mask_eval, n);

    std::cout << "== " << name << " METRICS ==\n";
    std::cout << "MAE   = " << mae << "\n";
    std::cout << "RMSE  = " << rmse << "\n";
    std::cout << "NRMSE = " << nrmse << "\n\n";
}


int main()
{
    std::cout << "Loading dataset...\n";

    // Load từ .npy
    MimicDataset ds_cpu =
        MimicLoader::load_from_npy(
            "data",
            "matrix_truth.npy",
            "matrix_miss_10.npy",
            "mask_10.npy"
        );

    // Upload GPU
    GpuMimicMatrix ds_gpu;
    ds_gpu.upload(ds_cpu);

    int N = ds_gpu.rows();
    int D = ds_gpu.cols();
    int total = N * D;

    std::cout << "Dataset loaded: " << N << " x " << D << "\n";

    __half* d_miss_fp16 = ds_gpu.d_miss();
    __half* d_truth_fp16 = ds_gpu.d_truth();
    uint8_t* d_mask_orig = ds_gpu.d_mask();   // mask gốc (không sửa để tính metric)

    // Convert truth & miss → FP32
    float* d_X32;
    float* d_truth32;
    CUDA_CHECK(cudaMalloc(&d_X32, sizeof(float) * total));
    CUDA_CHECK(cudaMalloc(&d_truth32, sizeof(float) * total));

    convert_half_to_float(d_truth32, d_truth_fp16, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Mask làm việc (mỗi thuật toán dùng bản copy riêng)
    uint8_t* d_mask_work;
    CUDA_CHECK(cudaMalloc(&d_mask_work, sizeof(uint8_t) * total));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int K = 10;
    const int max_iter = 5;

    // =====================================================
    // 1) KNNimpute
    // =====================================================
    convert_half_to_float(d_X32, d_miss_fp16, total);
    CUDA_CHECK(cudaMemcpy(d_mask_work, d_mask_orig,
        sizeof(uint8_t) * total,
        cudaMemcpyDeviceToDevice));

    std::cout << "\nRunning KNN...\n";
    impute::knn_impute(d_X32, d_mask_work, N, D, K, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_metrics("KNN", d_truth32, d_X32, d_mask_orig, N, D);


    // =====================================================
    // 2) iKNNimpute
    // =====================================================
    convert_half_to_float(d_X32, d_miss_fp16, total);
    CUDA_CHECK(cudaMemcpy(d_mask_work, d_mask_orig,
        sizeof(uint8_t) * total,
        cudaMemcpyDeviceToDevice));

    std::cout << "Running iKNN...\n";
    impute::iknn_impute(d_X32, d_mask_work, N, D,
        K,         // K
        max_iter,  // max_iter
        1e-4f,     // tol
        stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_metrics("iKNN", d_truth32, d_X32, d_mask_orig, N, D);


    // =====================================================
    // 3) SKNNimpute (Sequential KNN)
    // =====================================================
    convert_half_to_float(d_X32, d_miss_fp16, total);
    CUDA_CHECK(cudaMemcpy(d_mask_work, d_mask_orig,
        sizeof(uint8_t) * total,
        cudaMemcpyDeviceToDevice));

    std::cout << "Running SKNN...\n";
    impute::sknn_impute(d_X32, d_mask_work, N, D, K, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_metrics("SKNN", d_truth32, d_X32, d_mask_orig, N, D);


    // =====================================================
    // 4) LS-impute cũ của bạn (baseline, namespace impute_ls)
    // =====================================================
    convert_half_to_float(d_X32, d_miss_fp16, total);
    CUDA_CHECK(cudaMemcpy(d_mask_work, d_mask_orig,
        sizeof(uint8_t) * total,
        cudaMemcpyDeviceToDevice));

    std::cout << "Running LS (old implementation)...\n";
    impute_ls::ls_impute(d_X32, d_mask_work, N, D, K, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_metrics("LS-old", d_truth32, d_X32, d_mask_orig, N, D);


    // =====================================================
    // 5) LLSimpute (LLS-Impute mới dựa trên ls_workspace)
    // =====================================================
    convert_half_to_float(d_X32, d_miss_fp16, total);
    CUDA_CHECK(cudaMemcpy(d_mask_work, d_mask_orig,
        sizeof(uint8_t) * total,
        cudaMemcpyDeviceToDevice));

    std::cout << "Running LLSimpute...\n";
    impute::llsimpute(d_X32, d_mask_work, N, D, K, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_metrics("LLS", d_truth32, d_X32, d_mask_orig, N, D);


    // =====================================================
    // 6) SVD-impute (global method)
    // =====================================================
    convert_half_to_float(d_X32, d_miss_fp16, total);
    CUDA_CHECK(cudaMemcpy(d_mask_work, d_mask_orig,
        sizeof(uint8_t) * total,
        cudaMemcpyDeviceToDevice));

    std::cout << "Running SVD...\n";
    impute::svd_impute(d_X32, d_mask_work, N, D,
        16,     // rank
        max_iter,
        1e-5f,  // tol
        stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_metrics("SVD", d_truth32, d_X32, d_mask_orig, N, D);


    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_X32));
    CUDA_CHECK(cudaFree(d_truth32));
    CUDA_CHECK(cudaFree(d_mask_work));

    std::cout << "\nAll algorithms finished.\n";
    system("pause");

    return 0;
}
