#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using BFloat16 = hip_bfloat16;

__global__ void mix_variants(const BFloat16* normalized, const float* gates, BFloat16* output, int rows) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int variant = blockIdx.y;
    if (index >= rows * 2560) return;
    int row = index / 2560;
    int column = index % 2560;
    float products[4];
    for (int branch = 0; branch < 4; ++branch) {
        int offset = row * 10240 + branch * 2560 + column;
        float exponent = variant / 3 == 0 ? expf(-gates[offset]) : __builtin_amdgcn_exp2f(-gates[offset] * 1.4426950408889634f);
        float probability = variant / 3 == 2 ? __builtin_amdgcn_rcpf(1.0f + exponent) : 1.0f / (1.0f + exponent);
        products[branch] = probability * float(normalized[offset]);
    }
    float total;
    if (variant % 3 == 0) total = ((products[0] + products[1]) + products[2]) + products[3];
    else if (variant % 3 == 1) total = (products[0] + products[2]) + (products[1] + products[3]);
    else total = (products[0] + products[1]) + (products[2] + products[3]);
    output[variant * rows * 2560 + index] = BFloat16(total * 0.25f);
}

extern "C" int launch_mix_variants(const void* normalized, const float* gates, void* output, int rows, void* stream) {
    hipLaunchKernelGGL(mix_variants, dim3((rows * 2560 + 255) / 256, 9), dim3(256), 0, static_cast<hipStream_t>(stream), static_cast<const BFloat16*>(normalized), gates, static_cast<BFloat16*>(output), rows);
    return hipGetLastError();
}
