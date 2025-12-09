#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "gpu_mimic.hpp"
#include "mimic_dataset.hpp"

// Các thuật toán impute
#include "knn_impute.cuh"
#include "iknn_impute.cuh"
#include "svd_impute.cuh"
#include "ls_impute.cuh"

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
// Hàm in METRICS đúng
// ======================
void print_metrics(
    const char* name,
    const float* d_truth,
    const float* d_pred,
    const uint8_t* d_mask,
    int N, int D)
{
    size_t n = (size_t)N * D;

    float mae = metrics::mae_masked(d_truth, d_pred, d_mask, n);
    float rmse = metrics::rmse_masked(d_truth, d_pred, d_mask, n);
    float nrmse = metrics::nrmse_masked(d_truth, d_pred, d_mask, n);

    std::cout << "== " << name << " METRICS ==\n";
    std::cout << "MAE   = " << mae << "\n";
    std::cout << "RMSE  = " << rmse << "\n";
    std::cout << "NRMSE = " << nrmse << "\n\n";
}


int main()
{
    std::cout << "Loading dataset...\n";

    // Load .npy
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

    __half* d_miss = ds_gpu.d_miss();
    __half* d_truth_fp16 = ds_gpu.d_truth();
    uint8_t* d_mask = ds_gpu.d_mask();

    // Convert truth & miss → FP32
    float* d_X32;
    float* d_truth32;

    CUDA_CHECK(cudaMalloc(&d_X32, sizeof(float) * total));
    CUDA_CHECK(cudaMalloc(&d_truth32, sizeof(float) * total));

    convert_half_to_float(d_truth32, d_truth_fp16, total);
    convert_half_to_float(d_X32, d_miss, total);
    CUDA_CHECK(cudaDeviceSynchronize());


    cudaStream_t stream = nullptr;


    // ===========================
    // 1) KNN
    // ===========================
    std::cout << "Running KNN...\n";

    impute::knn_impute(d_X32, d_mask, N, D, 10, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_metrics("KNN", d_truth32, d_X32, d_mask, N, D);


    // ===========================
    // 2) iKNN
    // ===========================
    convert_half_to_float(d_X32, d_miss, total);

    std::cout << "Running iKNN...\n";

    impute::iknn_impute(d_X32, d_mask, N, D,
        10,    // K
        5,     // max_iter
        1e-4f,
        stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_metrics("iKNN", d_truth32, d_X32, d_mask, N, D);


    // ===========================
    // 3) LS
    // ===========================
    convert_half_to_float(d_X32, d_miss, total);

    std::cout << "Running LS...\n";

    impute_ls::ls_impute(d_X32, d_mask, N, D, 10, stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_metrics("LS", d_truth32, d_X32, d_mask, N, D);


    // ===========================
    // 4) SVD
    // ===========================
    convert_half_to_float(d_X32, d_miss, total);

    std::cout << "Running SVD...\n";

    impute::svd_impute(d_X32, d_mask, N, D,
        16,    // rank
        5,     // max_iter
        stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    print_metrics("SVD", d_truth32, d_X32, d_mask, N, D);


    // Cleanup
    CUDA_CHECK(cudaFree(d_X32));
    CUDA_CHECK(cudaFree(d_truth32));

    std::cout << "\nAll algorithms finished.\n";
    
    system("pause");

    return 0;
}
