#include "../sgl_amd_mono/qwen38_27b/gfx936/kernel.cpp"

__global__ void probe_decode_head(const BFloat16* input, const BFloat16* weight, BFloat16* output, int rows, int channels, int order) {
    int lane = threadIdx.x & 63;
    int work = blockIdx.x * 4 + threadIdx.x / 64;
    if (work >= rows * channels) return;
    int row = work / channels;
    int channel = work % channels;
    float partials[2];
    for (int group = 0; group < 2; ++group) {
        Float4 accumulated = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int start = group * 512; start < Hidden; start += 1024) {
            int column = start + lane * 8;
            Short4 left = *reinterpret_cast<const Short4*>(input + row * Hidden + column);
            Short4 right = *reinterpret_cast<const Short4*>(weight + int64_t(channel) * Hidden + column);
            accumulated = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulated);
            left = *reinterpret_cast<const Short4*>(input + row * Hidden + column + 4);
            right = *reinterpret_cast<const Short4*>(weight + int64_t(channel) * Hidden + column + 4);
            accumulated = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulated);
        }
        float diagonal[16];
        for (int index = 0; index < 16; ++index)
            diagonal[index] = __shfl(accumulated[index / 4], (index % 4) * 17 + (index / 4) * 4, 64);
        float pieces[4];
        for (int index = 0; index < 4; ++index)
            pieces[index] = (diagonal[index * 4] + diagonal[index * 4 + 1]) + (diagonal[index * 4 + 2] + diagonal[index * 4 + 3]);
        float total = pieces[order];
        for (int index = 1; index < 4; ++index) total += pieces[(order + index) % 4];
        partials[group] = total;
    }
    if (lane == 0) output[row * channels + channel] = BFloat16(partials[0] + partials[1]);
}

extern "C" int probe_head(const void* input, const void* weight, void* output, int rows, int channels, int order, void* stream) {
    hipLaunchKernelGGL(probe_decode_head, dim3((rows * channels + 3) / 4), dim3(256), 0, static_cast<hipStream_t>(stream), static_cast<const BFloat16*>(input), static_cast<const BFloat16*>(weight), static_cast<BFloat16*>(output), rows, channels, order);
    return hipGetLastError();
}
