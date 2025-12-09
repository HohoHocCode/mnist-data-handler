#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Kernel chuyển FP16 → FP32
__global__ void convert_half_to_float_kernel(float* out, const __half* in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = __half2float(in[i]);
    }
}

// Hàm wrapper để gọi kernel từ C++ (main.cpp)
extern "C" void convert_half_to_float(float* out, const __half* in, int n)
{
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    convert_half_to_float_kernel << <grid, block >> > (out, in, n);
}
