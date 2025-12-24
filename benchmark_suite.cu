#include "ImputingLibrary/cuda_bin_io.hpp"
#include "imputation_lib/gpu/gpu_timer.cuh"
#include "imputation_lib/gpu/gpu_wrappers.hpp"
#include <cmath>
#include <cub/cub.cuh>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace impute;

struct Result {
  std::string name;
  float mae;
  float nrmse;
  float time_ms;
};

// Specialized kernels for holdout metrics
__global__ void mae_holdout_kernel(const float *d_X, const int *d_holdout_idx,
                                   const float *d_holdout_y, int n_holdout,
                                   double *d_sum, int *d_count,
                                   size_t total_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double local_sum = 0;
  int local_count = 0;
  for (int i = idx; i < n_holdout; i += blockDim.x * gridDim.x) {
    int flat_idx = d_holdout_idx[i];
    if (flat_idx < total_size) {
      local_sum += abs((double)d_X[flat_idx] - (double)d_holdout_y[i]);
      local_count++;
    }
  }
  typedef cub::BlockReduce<double, 256> BlockReduceSum;
  typedef cub::BlockReduce<int, 256> BlockReduceCnt;
  __shared__ typename BlockReduceSum::TempStorage temp_s;
  __shared__ typename BlockReduceCnt::TempStorage temp_c;

  double total_s = BlockReduceSum(temp_s).Sum(local_sum);
  __syncthreads();
  int total_c = BlockReduceCnt(temp_c).Sum(local_count);

  if (threadIdx.x == 0) {
    atomicAdd(d_sum, total_s);
    atomicAdd(d_count, total_c);
  }
}

__global__ void nrmse_holdout_kernel(const float *d_X, const int *d_holdout_idx,
                                     const float *d_holdout_y, int n_holdout,
                                     double *d_num, double *d_den,
                                     size_t total_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double local_num = 0, local_den = 0;
  for (int i = idx; i < n_holdout; i += blockDim.x * gridDim.x) {
    int flat_idx = d_holdout_idx[i];
    if (flat_idx < total_size) {
      double diff = (double)d_X[flat_idx] - (double)d_holdout_y[i];
      local_num += diff * diff;
      local_den += (double)d_holdout_y[i] * (double)d_holdout_y[i];
    }
  }
  typedef cub::BlockReduce<double, 256> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_num;
  __shared__ typename BlockReduce::TempStorage temp_storage_den;

  double total_num = BlockReduce(temp_storage_num).Sum(local_num);
  __syncthreads();
  double total_den = BlockReduce(temp_storage_den).Sum(local_den);

  if (threadIdx.x == 0) {
    atomicAdd(d_num, total_num);
    atomicAdd(d_den, total_den);
  }
}

void run_benchmark() {
  int N = 11756, D = 200;
  size_t total_size = (size_t)N * D;
  int n_holdout_total = 46340;

  std::string data_dir = R"(D:\icu_dataset\processed\cuda_eval)";
  auto X_miss =
      read_f32(data_dir + "\\X_miss_float32.bin", total_size * sizeof(float));
  auto Mask =
      read_u8(data_dir + "\\M_train_uint8.bin", total_size * sizeof(uint8_t));
  auto holdout_idx = read_i32(data_dir + "\\holdout_idx_int32.bin",
                              n_holdout_total * sizeof(int32_t));
  auto holdout_y = read_f32(data_dir + "\\holdout_y_float32.bin",
                            n_holdout_total * sizeof(float));

  if (X_miss.empty() || Mask.empty() || holdout_idx.empty() ||
      holdout_y.empty()) {
    std::cerr << "Error: Could not load dataset." << std::endl;
    return;
  }

  float *d_X, *d_holdout_y;
  uint8_t *d_mask;
  int *d_holdout_idx;
  cudaMalloc(&d_X, total_size * sizeof(float));
  cudaMalloc(&d_mask, total_size * sizeof(uint8_t));
  cudaMalloc(&d_holdout_idx, n_holdout_total * sizeof(int));
  cudaMalloc(&d_holdout_y, n_holdout_total * sizeof(float));

  double *d_mae_sum, *d_nrmse_num, *d_nrmse_den;
  int *d_count;
  cudaMalloc(&d_mae_sum, sizeof(double));
  cudaMalloc(&d_nrmse_num, sizeof(double));
  cudaMalloc(&d_nrmse_den, sizeof(double));
  cudaMalloc(&d_count, sizeof(int));

  cudaMemcpy(d_holdout_idx, holdout_idx.data(), n_holdout_total * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_holdout_y, holdout_y.data(), n_holdout_total * sizeof(float),
             cudaMemcpyHostToDevice);

  std::vector<Result> results;
  GpuTimer timer;

  auto run_algo = [&](const std::string &name, auto imputer_call) {
    cudaMemcpy(d_X, X_miss.data(), total_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, Mask.data(), total_size * sizeof(uint8_t),
               cudaMemcpyHostToDevice);

    timer.start();
    imputer_call(d_X, d_mask, N, D);
    float ms = timer.stop();

    cudaMemset(d_mae_sum, 0, sizeof(double));
    cudaMemset(d_nrmse_num, 0, sizeof(double));
    cudaMemset(d_nrmse_den, 0, sizeof(double));
    cudaMemset(d_count, 0, sizeof(int));

    mae_holdout_kernel<<<(n_holdout_total + 255) / 256, 256>>>(
        d_X, d_holdout_idx, d_holdout_y, n_holdout_total, d_mae_sum, d_count,
        total_size);
    nrmse_holdout_kernel<<<(n_holdout_total + 255) / 256, 256>>>(
        d_X, d_holdout_idx, d_holdout_y, n_holdout_total, d_nrmse_num,
        d_nrmse_den, total_size);

    double mae_h, num_h, den_h;
    int count_h;
    cudaMemcpy(&mae_h, d_mae_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&num_h, d_nrmse_num, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&den_h, d_nrmse_den, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count_h, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    float nrmse_val = (den_h > 0) ? sqrt((float)(num_h / den_h)) : 0.0f;
    float mae_val = (count_h > 0) ? (float)(mae_h / count_h) : 0.0f;
    results.push_back({name, mae_val, nrmse_val, ms});
  };

  std::cout << "Benchmarking 16 Algorithms (N=" << N << ", D=" << D << ")..."
            << std::endl;

  auto run_and_print = [&](const std::string &name, auto imputer_call) {
    std::cout << "  - " << name << "... " << std::flush;
    run_algo(name, imputer_call);
    std::cout << "Done." << std::endl;
  };

  run_and_print("SVDImpute", [](float *x, uint8_t *m, int n, int d) {
    SvdImputerGpu(10, 3).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("BPCA", [](float *x, uint8_t *m, int n, int d) {
    BpcaImputerGpu(10, 3).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("KNN", [](float *x, uint8_t *m, int n, int d) {
    KnnImputerGpu(5).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("SKNN", [](float *x, uint8_t *m, int n, int d) {
    SknnImputerGpu(10).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("IKNN", [](float *x, uint8_t *m, int n, int d) {
    IknnImputerGpu(10, 3).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("GMC", [](float *x, uint8_t *m, int n, int d) {
    GmcImputerGpu(5, 3).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("LS", [](float *x, uint8_t *m, int n, int d) {
    LsImputerGpu(10).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("LLS", [](float *x, uint8_t *m, int n, int d) {
    LlsImputerGpu(10).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("SLLS", [](float *x, uint8_t *m, int n, int d) {
    SllsImputerGpu(10).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("ILLS", [](float *x, uint8_t *m, int n, int d) {
    IllsImputerGpu(5, 3).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("RLSP", [](float *x, uint8_t *m, int n, int d) {
    RlspImputerGpu(10, 5).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("BGS", [](float *x, uint8_t *m, int n, int d) {
    BgsImputerGpu(10, 0.1f).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("CMVE", [](float *x, uint8_t *m, int n, int d) {
    CmveImputerGpu(10).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("AMVI", [](float *x, uint8_t *m, int n, int d) {
    AmviImputerGpu(20).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("ARLS", [](float *x, uint8_t *m, int n, int d) {
    ArlsImputerGpu(10, 0.1f).impute_cuda(x, m, n, d, 0);
  });
  run_and_print("LinCmb", [](float *x, uint8_t *m, int n, int d) {
    LinCmbImputerGpu(5, 5).impute_cuda(x, m, n, d, 0);
  });

  std::cout << "\n| Algorithm | MAE | NRMSE | Speed (ms) |" << std::endl;
  std::cout << "| :--- | :--- | :--- | :--- |" << std::endl;
  for (const auto &r : results) {
    std::cout << "| " << r.name << " | " << std::fixed << std::setprecision(4)
              << r.mae << " | " << r.nrmse << " | " << r.time_ms << " |"
              << std::endl;
  }

  cudaFree(d_X);
  cudaFree(d_mask);
  cudaFree(d_holdout_idx);
  cudaFree(d_holdout_y);
  cudaFree(d_mae_sum);
  cudaFree(d_nrmse_num);
  cudaFree(d_nrmse_den);
  cudaFree(d_count);
}

int main() {
  run_benchmark();
  return 0;
}
