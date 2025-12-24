#pragma once
#include "../imputation_lib/i_imputer.hpp"
#include "cuda_bin_io.hpp"
#include "logger.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class ExperimentRunner {
public:
  ExperimentRunner(const std::string &data_dir,
                   const std::string &eval_dir = "") {
    std::string meta_path = data_dir + "\\meta.json";
    auto meta = read_meta_json(meta_path);
    if (meta.rows == 0) {
      std::cerr << "Failed to load meta.json from " << data_dir << "\n";
      return;
    }
    N = meta.rows;
    D = meta.cols;
    size_t n_total = (size_t)N * D;
    std::cout << "Loading dataset: " << N << "x" << D << "\n";
    X_miss = read_f32(data_dir + "\\" + meta.x_file, n_total * sizeof(float));
    Mask = read_u8(data_dir + "\\" + meta.m_file, n_total * sizeof(uint8_t));

    if (X_miss.empty() || Mask.empty()) {
      // Fallback for ICU eval structure
      std::cout << "Attempting fallback for ICU eval structure...\n";
      X_miss =
          read_f32(data_dir + "\\X_miss_float32.bin", n_total * sizeof(float));
      Mask =
          read_u8(data_dir + "\\M_train_uint8.bin", n_total * sizeof(uint8_t));
    }

    if (X_miss.empty() || Mask.empty()) {
      std::cerr << "Failed to load binary files! Check paths in meta.json. X="
                << meta.x_file << " M=" << meta.m_file << "\n";
    }
  }

  void set_evaluation_data(const std::vector<int> &indices,
                           const std::vector<float> &values) {
    holdout_idx = indices;
    holdout_y = values;
    has_eval_data = true;
  }

  void add_algorithm(std::shared_ptr<impute::IImputer> algo) {
    algorithms.push_back(algo);
  }

  void set_limit(int limit) { limit_rows = limit; }

  void run(const std::string &output_csv) {
    std::ofstream csv(output_csv);
    csv << "Algorithm,MAE,RMSE,TimeMs\n";

    int n_run = (limit_rows > 0 && limit_rows < N) ? limit_rows : N;
    if (limit_rows > 0)
      std::cout << "Running on first " << n_run << " rows.\n";

    for (auto &algo : algorithms) {
      std::cout << "Running " << algo->name() << "... ";

      size_t subset_size = (size_t)n_run * D;
      // Copy only needed subset
      if (X_miss.size() < subset_size) {
        std::cerr << "Data insufficient or load failed. Size=" << X_miss.size()
                  << " Req=" << subset_size << "\n";
        continue;
      }

      std::vector<float> X_sub(X_miss.begin(), X_miss.begin() + subset_size);
      std::vector<uint8_t> Mask_sub(Mask.begin(), Mask.begin() + subset_size);

      auto start = std::chrono::high_resolution_clock::now();
      try {
        algo->impute(X_sub.data(), Mask_sub.data(), n_run, D);
      } catch (const std::exception &e) {
        std::cerr << "Algorithm failed: " << e.what() << "\n";
        continue;
      }
      auto end = std::chrono::high_resolution_clock::now();

      double elapsed_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      std::cout << "Done in " << std::fixed << std::setprecision(2)
                << elapsed_ms << " ms. ";

      float score_mae = 0.0f;
      float score_rmse = 0.0f;

      if (has_eval_data) {
        evaluate(X_sub, score_mae, score_rmse, subset_size);
        std::cout << "MAE=" << score_mae << " RMSE=" << score_rmse << "\n";
      } else {
        std::cout << "(No eval data)\n";
      }

      csv << algo->name() << "," << score_mae << "," << score_rmse << ","
          << elapsed_ms << "\n";
    }
  }

  void tune(const std::vector<std::shared_ptr<impute::IImputer>> &variants,
            const std::string &name_tag) {
    std::cout << "Optimizing hyperparameters for " << name_tag << "...\n";
    float best_mae = 1e10f;
    std::shared_ptr<impute::IImputer> best_algo = nullptr;

    int n_run = (limit_rows > 0 && limit_rows < N) ? limit_rows : N;
    size_t subset_size = (size_t)n_run * D;
    std::vector<float> X_orig(X_miss.begin(), X_miss.begin() + subset_size);
    std::vector<uint8_t> Mask_sub(Mask.begin(), Mask.begin() + subset_size);

    for (auto &algo : variants) {
      std::vector<float> X_sub = X_orig;
      try {
        algo->impute(X_sub.data(), Mask_sub.data(), n_run, D);
        float mae = 0, rmse = 0;
        evaluate(X_sub, mae, rmse, subset_size);
        std::cout << "  - " << algo->name() << ": MAE=" << mae << "\n";
        if (mae < best_mae) {
          best_mae = mae;
          best_algo = algo;
        }
      } catch (...) {
        continue;
      }
    }

    if (best_algo) {
      std::cout << ">>> DONE. Best Parameters for " << name_tag << ": "
                << best_algo->name() << " (MAE=" << best_mae << ")\n";
    }
  }

  void run_streaming(const std::string &output_csv, int chunk_size = 4096) {
    impute::Logger::info(
        "Starting DOUBLE-BUFFERED STREAMING benchmark (Out-of-Core support).");
    std::ofstream csv(output_csv);
    csv << "Algorithm,MAE,RMSE,TimeMs\n";

    // Pinned memory is required for asynchronous overlap
    cudaHostRegister(X_miss.data(), (size_t)N * D * sizeof(float),
                     cudaHostRegisterDefault);
    cudaHostRegister(Mask.data(), (size_t)N * D * sizeof(uint8_t),
                     cudaHostRegisterDefault);

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    for (auto &algo : algorithms) {
      impute::Logger::info("Running " + algo->name());
      auto start = std::chrono::high_resolution_clock::now();

      // Allocate two sets of device buffers for double buffering
      float *d_X[2];
      uint8_t *d_Mask[2];
      for (int b = 0; b < 2; ++b) {
        cudaMalloc(&d_X[b], (size_t)chunk_size * D * sizeof(float));
        cudaMalloc(&d_Mask[b], (size_t)chunk_size * D * sizeof(uint8_t));
      }

      for (int i = 0; i < N; i += chunk_size) {
        int chunk_idx = (i / chunk_size) % 2;
        int n_curr = std::min(chunk_size, N - i);
        cudaStream_t s = streams[chunk_idx];

        // 1. Async Copy Host to Device
        cudaMemcpyAsync(d_X[chunk_idx], X_miss.data() + (size_t)i * D,
                        (size_t)n_curr * D * sizeof(float),
                        cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_Mask[chunk_idx], Mask.data() + (size_t)i * D,
                        (size_t)n_curr * D * sizeof(uint8_t),
                        cudaMemcpyHostToDevice, s);

        // 2. Async Compute (The kernel is now non-blocking)
        algo->impute_cuda(d_X[chunk_idx], d_Mask[chunk_idx], n_curr, D, s);

        // 3. Async Copy Device to Host
        cudaMemcpyAsync(X_miss.data() + (size_t)i * D, d_X[chunk_idx],
                        (size_t)n_curr * D * sizeof(float),
                        cudaMemcpyDeviceToHost, s);
      }

      cudaStreamSynchronize(streams[0]);
      cudaStreamSynchronize(streams[1]);
      auto end = std::chrono::high_resolution_clock::now();

      for (int b = 0; b < 2; ++b) {
        cudaFree(d_X[b]);
        cudaFree(d_Mask[b]);
      }

      double elapsed_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      float score_mae = 0, score_rmse = 0;
      evaluate(X_miss, score_mae, score_rmse, (size_t)N * D);

      impute::Logger::info("Done in " + std::to_string(elapsed_ms) +
                           " ms. MAE=" + std::to_string(score_mae));
      csv << algo->name() << "," << score_mae << "," << score_rmse << ","
          << elapsed_ms << "\n";
    }

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaHostUnregister(X_miss.data());
    cudaHostUnregister(Mask.data());
  }

private:
  int N = 0;
  int D = 0;
  std::vector<float> X_miss;
  std::vector<uint8_t> Mask;

  bool has_eval_data = false;
  std::vector<int> holdout_idx;
  std::vector<float> holdout_y;
  std::vector<std::shared_ptr<impute::IImputer>> algorithms;
  int limit_rows = 0;

  void evaluate(const std::vector<float> &X_pred, float &mae, float &rmse,
                size_t limit_idx) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    size_t count = 0;

    for (size_t k = 0; k < holdout_idx.size(); ++k) {
      int idx = holdout_idx[k];
      if ((size_t)idx >= limit_idx)
        continue;

      float pred = X_pred[idx];
      float truth = holdout_y[k];
      float diff = pred - truth;
      sum_abs += std::abs(diff);
      sum_sq += diff * diff;
      count++;
    }

    if (count > 0) {
      mae = static_cast<float>(sum_abs / count);
      rmse = static_cast<float>(std::sqrt(sum_sq / count));
    }
  }
};
