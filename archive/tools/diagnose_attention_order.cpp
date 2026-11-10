#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <cstdint>

using Short4 = short __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));

__global__ void attention_mmac_layout(float* output) {
    Short4 left = {0x3f80, 0x3f80, 0x3f80, 0x3f80};
    Short4 right = threadIdx.x % 16 == blockIdx.x ? left : Short4{0, 0, 0, 0};
    Float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    accumulator = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulator);
    for (int element = 0; element < 4; ++element) output[(blockIdx.x * 64 + threadIdx.x) * 4 + element] = accumulator[element];
}

extern "C" int probe_attention_mmac_layout(float* output, void* stream) {
    hipLaunchKernelGGL(attention_mmac_layout, dim3(6), dim3(64), 0, reinterpret_cast<hipStream_t>(stream), output);
    return hipGetLastError();
}

__global__ void attention_score_orders(const hip_bfloat16* query, const unsigned char* key,
                                      const int* table, const int* orders, float* output,
                                      int length) {
    int lane = threadIdx.x;
    int token = blockIdx.x * 16 + lane % 16;
    int head = blockIdx.y;
    int order = blockIdx.z;
    Float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    Short4 left_fragments[16], right_fragments[16];
    #pragma unroll
    for (int fragment = 0; fragment < 16; ++fragment) {
        int column = orders[order * 64 + fragment * 4 + lane / 16] * 4;
        Short4 left = {0, 0, 0, 0}, right = {0, 0, 0, 0};
        if (token < length) {
            int page = table[token / 64];
            for (int element = 0; element < 4; ++element) {
                int64_t index = ((int64_t(page) * 64 + token % 64) * 4 + head / 6) * 256 + column + element;
                float value = __half2float(__ushort_as_half(static_cast<unsigned short>(key[index]) << 8));
                left[element] = static_cast<short>(__float_as_uint(value) >> 16);
            }
        }
        if (lane % 16 < 6) right = *reinterpret_cast<const Short4*>(query + (head / 6 * 6 + lane % 16) * 256 + column);
        left_fragments[fragment] = left;
        right_fragments[fragment] = right;
    }
    #pragma unroll
    for (int fragment = 0; fragment < 16; ++fragment)
        accumulator = __builtin_hcu_mmac_f32_16x16x16_bf16(left_fragments[fragment], right_fragments[fragment], accumulator);
    if (lane / 16 == (head % 6) % 4 && token < length) {
        float scaled;
        asm volatile("v_mul_f32 %0, 0x3c38aa3b, %1" : "=v"(scaled) : "v"(accumulator[(head % 6) / 4]));
        output[(order * 24 + head) * length + token] = __half2float(__float2half_rn(scaled)) * 8.0f;
    }
}

extern "C" int probe_attention_scores(const void* query, const void* key, const int* table,
                                     const int* orders, float* output, int length,
                                     int count, void* stream) {
    hipLaunchKernelGGL(attention_score_orders, dim3((length + 15) / 16, 24, count), dim3(64), 0,
                       reinterpret_cast<hipStream_t>(stream), static_cast<const hip_bfloat16*>(query),
                       static_cast<const unsigned char*>(key), table, orders, output, length);
    return hipGetLastError();
}

__global__ void attention_probability_orders(float* scores, int length, float* statistics) {
    int begin = blockIdx.x * 512;
    int base = (blockIdx.z * 24 + blockIdx.y) * length;
    float values[8];
    float maximum = -INFINITY;
    for (int element = 0; element < 8; ++element) {
        int token = begin + threadIdx.x * 8 + element;
        values[element] = token < length ? scores[base + token] : -INFINITY;
        maximum = fmaxf(maximum, values[element]);
    }
    for (int offset = 32; offset > 0; offset /= 2) maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 64));
    float denominator = 0.0f;
    for (int element = 0; element < 8; ++element) {
        values[element] = __builtin_amdgcn_exp2f(values[element] - maximum);
        denominator += values[element];
    }
    for (int offset = 32; offset > 0; offset /= 2) denominator += __shfl_xor(denominator, offset, 64);
    if (statistics && threadIdx.x == 0) {
        int index = (blockIdx.z * 24 + blockIdx.y) * ((length + 511) / 512) + blockIdx.x;
        statistics[index * 2] = maximum;
        statistics[index * 2 + 1] = denominator;
    }
    float inverse = __builtin_amdgcn_rcpf(denominator + 1.0e-6f);
    for (int element = 0; element < 8; ++element) {
        int token = begin + threadIdx.x * 8 + element;
        float probability = values[element] * inverse;
        if (token < length) scores[base + token] = __uint_as_float((__float_as_uint(probability) + 0x8000u) & 0xffff0000u);
    }
}

extern "C" int probe_attention_probabilities(float* scores, int length, int count, void* stream) {
    hipLaunchKernelGGL(attention_probability_orders, dim3((length + 511) / 512, 24, count), dim3(64), 0,
                       reinterpret_cast<hipStream_t>(stream), scores, length, nullptr);
    return hipGetLastError();
}

extern "C" int probe_attention_statistics(float* scores, float* statistics, int length, int count, void* stream) {
    hipLaunchKernelGGL(attention_probability_orders, dim3((length + 511) / 512, 24, count), dim3(64), 0,
                       reinterpret_cast<hipStream_t>(stream), scores, length, statistics);
    return hipGetLastError();
}
