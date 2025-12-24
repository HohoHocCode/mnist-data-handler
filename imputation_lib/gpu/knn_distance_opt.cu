#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK 16

__global__ void compute_distances_tiled(
    const float* __restrict__ X,
    const uint8_t* __restrict__ mask,
    float* __restrict__ dist,
    int N, int D
) {
    __shared__ float tile_i[BLOCK][BLOCK];
    __shared__ float tile_j[BLOCK][BLOCK];
    __shared__ uint8_t mask_i[BLOCK][BLOCK];
    __shared__ uint8_t mask_j[BLOCK][BLOCK];

    int i = blockIdx.y * BLOCK + threadIdx.y;
    int j = blockIdx.x * BLOCK + threadIdx.x;

    float s = 0.0f;

    for (int c0 = 0; c0 < D; c0 += BLOCK) {
        int c = c0 + threadIdx.x;

        if (i < N && c < D) {
            tile_i[threadIdx.y][threadIdx.x] = X[i * D + c];
            mask_i[threadIdx.y][threadIdx.x] = mask[i * D + c];
        }
        else {
            tile_i[threadIdx.y][threadIdx.x] = 0.0f;
            mask_i[threadIdx.y][threadIdx.x] = 1;
        }

        if (j < N && c < D) {
            tile_j[threadIdx.y][threadIdx.x] = X[j * D + c];
            mask_j[threadIdx.y][threadIdx.x] = mask[j * D + c];
        }
        else {
            tile_j[threadIdx.y][threadIdx.x] = 0.0f;
            mask_j[threadIdx.y][threadIdx.x] = 1;
        }

        __syncthreads();

        if (i < N && j < N) {
#pragma unroll
            for (int k = 0; k < BLOCK; k++) {
                if (!(mask_i[threadIdx.y][k] || mask_j[k][threadIdx.x])) {
                    float diff = tile_i[threadIdx.y][k] - tile_j[k][threadIdx.x];
                    s += diff * diff;
                }
            }
        }

        __syncthreads();
    }

    if (i < N && j < N) {
        if (i == j) s = 1e30f;
        dist[i * N + j] = s;
    }
}
