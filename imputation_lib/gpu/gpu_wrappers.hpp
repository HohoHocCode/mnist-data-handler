#pragma once

#include "../i_imputer.hpp"
#include "bgs_impute_gpu.cuh"
#include "bpca_impute_gpu.cuh"
#include "ills_impute_gpu.cuh"
#include "lls_impute_gpu.cuh"
#include "rlsp_impute_gpu.cuh"
#include "svd_impute_gpu.cuh"
#include <cuda_runtime.h>
#include <string>
#include <vector>

// Include implementations for single-compilation unit (CLI runner)
#include "amvi_impute_gpu.cu"
#include "arls_impute_gpu.cu"
#include "bgs_impute_gpu.cu"
#include "bpca_impute_gpu.cu"
#include "cmve_impute_gpu.cu"
#include "gmc_impute_gpu.cu"
#include "iknn_impute_gpu.cu"
#include "ills_impute_gpu.cu"
#include "knn_impute_gpu.cu"
#include "lincmb_impute_gpu.cu"
#include "lls_impute_gpu.cu"
#include "ls_impute_gpu.cu"
#include "ls_workspace.cu"
#include "rlsp_impute_gpu.cu"
#include "sknn_impute_gpu.cu"
#include "slls_impute_gpu.cu"
#include "svd_impute_gpu.cu"
#include "svd_workspace.cu"

namespace impute {

// Helper for CUDA check
inline void check_gpu(cudaError_t result, const char *func) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", func, cudaGetErrorString(result));
  }
}

class LlsImputerGpu : public IImputer {
public:
  LlsImputerGpu(int k) : K(k) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~LlsImputerGpu() { cudaStreamDestroy(stream_); }

  std::string name() const override {
    return "LlsImputerGpu (K=" + std::to_string(K) + ")";
  }

  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;

    check_gpu(cudaMalloc(&d_X, N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, N * D * sizeof(uint8_t)), "MallocMask");

    check_gpu(cudaMemcpyAsync(d_X, X, N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D X");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D Mask");

    impute_cuda(d_X, d_Mask, N, D, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H X");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");

    cudaFree(d_X);
    cudaFree(d_Mask);
  }

  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    lls_impute_cuda(d_X, d_Mask, N, D, K, stream);
  }

private:
  int K;
  cudaStream_t stream_;
};

class BpcaImputerGpu : public IImputer {
public:
  BpcaImputerGpu(int k_req = -1, int max_iter = 50)
      : K_req(k_req), max_iter_(max_iter) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~BpcaImputerGpu() { cudaStreamDestroy(stream_); }

  std::string name() const override { return "BpcaImputerGpu (Valid VB-EM)"; }

  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    int eff_K = (K_req == -1) ? (D - 1) : K_req;

    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, N * D * sizeof(uint8_t)), "MallocMask");

    check_gpu(cudaMemcpyAsync(d_X, X, N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D X");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D Mask");

    impute_cuda(d_X, d_Mask, N, D, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H X");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");

    cudaFree(d_X);
    cudaFree(d_Mask);
  }

  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    int eff_K = (K_req == -1) ? (D - 1) : K_req;
    bpca_impute_cuda(d_X, d_Mask, N, D, eff_K, max_iter_, stream);
  }

private:
  int K_req;
  int max_iter_;
  cudaStream_t stream_;
};

class SvdImputerGpu : public IImputer {
public:
  SvdImputerGpu(int rank, int max_iter = 50)
      : rank_(rank), max_iter_(max_iter) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~SvdImputerGpu() { cudaStreamDestroy(stream_); }

  std::string name() const override {
    return "SvdImputerGpu (Rank=" + std::to_string(rank_) + ")";
  }

  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, N * D * sizeof(uint8_t)), "MallocMask");

    check_gpu(cudaMemcpyAsync(d_X, X, N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D X");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D Mask");

    impute_cuda(d_X, d_Mask, N, D, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H X");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");

    cudaFree(d_X);
    cudaFree(d_Mask);
  }

  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    svd_impute(d_X, d_Mask, N, D, rank_, max_iter_, 1e-5f, stream);
  }

private:
  int rank_;
  int max_iter_;
  cudaStream_t stream_;
};

class IllsImputerGpu : public IImputer {
public:
  IllsImputerGpu(int k, int max_iter = 5) : K(k), max_iter_(max_iter) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~IllsImputerGpu() { cudaStreamDestroy(stream_); }

  std::string name() const override {
    return "IllsImputerGpu (K=" + std::to_string(K) + ")";
  }

  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");

    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D X");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D Mask");

    impute_cuda(d_X, d_Mask, N, D, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H X");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");

    cudaFree(d_X);
    cudaFree(d_Mask);
  }

  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    ills_impute_cuda(d_X, d_Mask, N, D, K, max_iter_, stream);
  }

private:
  int K;
  int max_iter_;
  cudaStream_t stream_;
};

class RlspImputerGpu : public IImputer {
public:
  RlspImputerGpu(int k, int n_pc) : K(k), npc(n_pc) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~RlspImputerGpu() { cudaStreamDestroy(stream_); }

  std::string name() const override {
    return "RlspImputerGpu (K=" + std::to_string(K) +
           ", PC=" + std::to_string(npc) + ")";
  }

  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");

    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D X");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D Mask");

    impute_cuda(d_X, d_Mask, N, D, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H X");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");

    cudaFree(d_X);
    cudaFree(d_Mask);
  }

  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    rlsp_impute_cuda(d_X, d_Mask, N, D, K, npc, stream);
  }

private:
  int K;
  int npc;
  cudaStream_t stream_;
};

class AmviImputerGpu : public IImputer {
public:
  AmviImputerGpu(int MaxK) : MaxK_(MaxK) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~AmviImputerGpu() { cudaStreamDestroy(stream_); }
  std::string name() const override {
    return "AMVI (MaxK=" + std::to_string(MaxK_) + ")";
  }
  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");
    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");

    impute::amvi_impute_cuda(d_X, d_Mask, N, D, MaxK_, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");
    cudaFree(d_X);
    cudaFree(d_Mask);
  }
  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    impute::amvi_impute_cuda(d_X, d_Mask, N, D, MaxK_, stream);
  }

private:
  int MaxK_;
  cudaStream_t stream_;
};

class ArlsImputerGpu : public IImputer {
public:
  ArlsImputerGpu(int P, float ridge = 0.1f) : P_(P), Ridge_(ridge) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~ArlsImputerGpu() { cudaStreamDestroy(stream_); }
  std::string name() const override {
    return "ARLS (P=" + std::to_string(P_) + ")";
  }
  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");
    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");

    impute::arls_impute_cuda(d_X, d_Mask, N, D, P_, Ridge_, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");
    cudaFree(d_X);
    cudaFree(d_Mask);
  }
  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    impute::arls_impute_cuda(d_X, d_Mask, N, D, P_, Ridge_, stream);
  }

private:
  int P_;
  float Ridge_;
  cudaStream_t stream_;
};

class LinCmbImputerGpu : public IImputer {
public:
  LinCmbImputerGpu(int k_knn, int rank_svd) : k_(k_knn), rank_(rank_svd) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~LinCmbImputerGpu() { cudaStreamDestroy(stream_); }
  std::string name() const override {
    return "LinCmb (k=" + std::to_string(k_) + ", r=" + std::to_string(rank_) +
           ")";
  }

  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");
    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");

    impute::lincmb_impute_cuda(d_X, d_Mask, N, D, k_, rank_, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");
    cudaFree(d_X);
    cudaFree(d_Mask);
  }

  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    impute::lincmb_impute_cuda(d_X, d_Mask, N, D, k_, rank_, stream);
  }

private:
  int k_, rank_;
  cudaStream_t stream_;
};

class BgsImputerGpu : public IImputer {
public:
  BgsImputerGpu(int max_genes, float ridge = 1e-1f)
      : max_genes_(max_genes), ridge_(ridge) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~BgsImputerGpu() { cudaStreamDestroy(stream_); }

  std::string name() const override {
    return "BgsImputerGpu (MaxGenes=" + std::to_string(max_genes_) +
           ", Ridge=" + std::to_string(ridge_) + ")";
  }

  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");

    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D X");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D Mask");

    impute_cuda(d_X, d_Mask, N, D, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H X");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");

    cudaFree(d_X);
    cudaFree(d_Mask);
  }

  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    bgs_impute_cuda(d_X, d_Mask, N, D, max_genes_, ridge_, stream);
  }

private:
  int max_genes_;
  float ridge_;
  cudaStream_t stream_;
};

class KnnImputerGpu : public IImputer {
public:
  KnnImputerGpu(int K) : K_(K) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~KnnImputerGpu() { cudaStreamDestroy(stream_); }
  std::string name() const override {
    return "KNN (K=" + std::to_string(K_) + ")";
  }
  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");
    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");

    impute::knn_impute(d_X, d_Mask, N, D, K_, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");
    cudaFree(d_X);
    cudaFree(d_Mask);
  }
  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    impute::knn_impute(d_X, d_Mask, N, D, K_, stream);
  }

private:
  int K_;
  cudaStream_t stream_;
};

class SknnImputerGpu : public IImputer {
public:
  SknnImputerGpu(int K) : K_(K) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~SknnImputerGpu() { cudaStreamDestroy(stream_); }
  std::string name() const override {
    return "SKNN (K=" + std::to_string(K_) + ")";
  }
  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");
    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");

    impute::sknn_impute_cuda(d_X, d_Mask, N, D, K_, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");
    cudaFree(d_X);
    cudaFree(d_Mask);
  }
  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    impute::sknn_impute_cuda(d_X, d_Mask, N, D, K_, stream);
  }

private:
  int K_;
  cudaStream_t stream_;
};

class IknnImputerGpu : public IImputer {
public:
  IknnImputerGpu(int K, int max_iter = 5) : K_(K), max_iter_(max_iter) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~IknnImputerGpu() { cudaStreamDestroy(stream_); }
  std::string name() const override {
    return "IKNN (K=" + std::to_string(K_) + ")";
  }
  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");
    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");

    impute::iknn_impute(d_X, d_Mask, N, D, K_, max_iter_, 1e-4f, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");
    cudaFree(d_X);
    cudaFree(d_Mask);
  }
  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    impute::iknn_impute(d_X, d_Mask, N, D, K_, max_iter_, 1e-4f, stream);
  }

private:
  int K_;
  int max_iter_;
  cudaStream_t stream_;
};

class LsImputerGpu : public IImputer {
public:
  LsImputerGpu(int K) : K_(K) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~LsImputerGpu() { cudaStreamDestroy(stream_); }
  std::string name() const override {
    return "LS (K=" + std::to_string(K_) + ")";
  }
  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");
    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");

    impute_ls::ls_impute(d_X, d_Mask, N, D, K_, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");
    cudaFree(d_X);
    cudaFree(d_Mask);
  }
  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    impute_ls::ls_impute(d_X, d_Mask, N, D, K_, stream);
  }

private:
  int K_;
  cudaStream_t stream_;
};

class GmcImputerGpu : public IImputer {
public:
  GmcImputerGpu(int K, int iter) : K_(K), iter_(iter) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~GmcImputerGpu() { cudaStreamDestroy(stream_); }
  std::string name() const override {
    return "GMC (K=" + std::to_string(K_) + ")";
  }
  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");
    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");

    // Default 100 iter if not specified? Wrapper passed generic iter.
    impute::gmc_impute_cuda(d_X, d_Mask, N, D, K_, iter_, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");
    cudaFree(d_X);
    cudaFree(d_Mask);
  }
  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    impute::gmc_impute_cuda(d_X, d_Mask, N, D, K_, iter_, stream);
  }

private:
  int K_;
  int iter_;
  cudaStream_t stream_;
};

class CmveImputerGpu : public IImputer {
public:
  CmveImputerGpu(int K) : K_(K) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~CmveImputerGpu() { cudaStreamDestroy(stream_); }
  std::string name() const override {
    return "CMVE (K=" + std::to_string(K_) + ")";
  }
  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");
    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");

    impute::cmve_impute_cuda(d_X, d_Mask, N, D, K_, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");
    cudaFree(d_X);
    cudaFree(d_Mask);
  }
  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    impute::cmve_impute_cuda(d_X, d_Mask, N, D, K_, stream);
  }

private:
  int K_;
  cudaStream_t stream_;
};

class SllsImputerGpu : public IImputer {
public:
  SllsImputerGpu(int K) : K_(K) {
    check_gpu(cudaStreamCreate(&stream_), "StreamCreate");
  }
  ~SllsImputerGpu() { cudaStreamDestroy(stream_); }
  std::string name() const override {
    return "SLLS (K=" + std::to_string(K_) + ")";
  }
  void impute(float *X, const uint8_t *Mask, int N, int D) override {
    float *d_X;
    uint8_t *d_Mask;
    check_gpu(cudaMalloc(&d_X, (size_t)N * D * sizeof(float)), "MallocX");
    check_gpu(cudaMalloc(&d_Mask, (size_t)N * D * sizeof(uint8_t)),
              "MallocMask");
    check_gpu(cudaMemcpyAsync(d_X, X, (size_t)N * D * sizeof(float),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");
    check_gpu(cudaMemcpyAsync(d_Mask, Mask, (size_t)N * D * sizeof(uint8_t),
                              cudaMemcpyHostToDevice, stream_),
              "H2D");

    impute::slls_impute_cuda(d_X, d_Mask, N, D, K_, stream_);

    check_gpu(cudaMemcpyAsync(X, d_X, (size_t)N * D * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_),
              "D2H");
    check_gpu(cudaStreamSynchronize(stream_), "Sync");
    cudaFree(d_X);
    cudaFree(d_Mask);
  }
  void impute_cuda(float *d_X, uint8_t *d_Mask, int N, int D,
                   cudaStream_t stream) override {
    impute::slls_impute_cuda(d_X, d_Mask, N, D, K_, stream);
  }

private:
  int K_;
  cudaStream_t stream_;
};

} // namespace impute
