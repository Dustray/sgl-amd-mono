#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>

using BFloat16 = hip_bfloat16;
using Short4 = short __attribute__((ext_vector_type(4)));
using Float2 = float __attribute__((ext_vector_type(2)));
using Float4 = float __attribute__((ext_vector_type(4)));
constexpr int Hidden = 5120;
constexpr int Intermediate = 17408;
constexpr int Core = 6144;
constexpr int Key = 2048;
constexpr int Conv = 10240;
constexpr int Projection = 16384;
constexpr int ValueHeads = 48;
constexpr int Head = 128;
constexpr int Threads = 256;
constexpr int Blocks = 80;
enum class MatvecMode { Staggered, Split, Gating };

struct K1Arguments {
    const BFloat16* hidden;
    const BFloat16* residual;
    const BFloat16* norm_weight;
    const BFloat16* qkvz_weight;
    const BFloat16* ba_weight;
    const BFloat16* conv_weight;
    const BFloat16* gated_weight;
    const float* a_log;
    const BFloat16* dt_bias;
    BFloat16* conv_state;
    float* ssm_state;
    const void* cache_indices;
    BFloat16* normalized;
    BFloat16* projected;
    BFloat16* projected_ba;
    BFloat16* convolved;
    BFloat16* recurrent;
    BFloat16* output;
    BFloat16* residual_output;
    int* executions;
    unsigned int* synchronization;
    int64_t rows;
    int64_t conv_slot_stride;
    int64_t conv_channel_stride;
    int64_t conv_history_stride;
    int64_t ssm_slot_stride;
    int64_t index_bytes;
    int64_t slot_count;
    int64_t barrier_timeout_ticks;
};

struct K2Arguments {
    const BFloat16* core;
    const BFloat16* residual;
    const BFloat16* output_weight;
    const BFloat16* norm_weight;
    const BFloat16* gate_up_weight;
    const BFloat16* down_weight;
    BFloat16* attention_output;
    BFloat16* normalized;
    BFloat16* projected;
    BFloat16* activated;
    BFloat16* output;
    BFloat16* residual_output;
    int* executions;
    unsigned int* synchronization;
    int64_t rows;
    int64_t barrier_timeout_ticks;
};

__device__ __forceinline__ bool synchronize_grid(unsigned int* state, int64_t timeout_ticks, unsigned int stage) {
    __shared__ int completed;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        completed = 1;
        unsigned int epoch = atomicAdd(state + 1, 0u);
        unsigned int arrived = atomicAdd(state, 1u);
        if (arrived == Blocks - 1) {
            atomicExch(state, 0u);
            atomicExch(state + 1, epoch + 1);
        } else {
            int64_t begin = wall_clock64();
            while (atomicAdd(state + 1, 0u) == epoch) {
                if (atomicAdd(state + 2, 0u) || wall_clock64() - begin > timeout_ticks) {
                    atomicCAS(state + 2, 0u, stage);
                    completed = 0;
                    break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
        if (atomicAdd(state + 2, 0u)) completed = 0;
    }
    __syncthreads();
    __threadfence();
    return completed;
}

__device__ __forceinline__ float wave_sum(float value) {
    for (int offset = 32; offset > 0; offset /= 2) value += __shfl_down(value, offset, 64);
    return __shfl(value, 0, 64);
}

__device__ __forceinline__ float block_sum(float value, float* shared) {
    value = wave_sum(value);
    if ((threadIdx.x & 63) == 0) shared[threadIdx.x / 64] = value;
    __syncthreads();
    float total = threadIdx.x < 4 ? shared[threadIdx.x] : 0.0f;
    if (threadIdx.x < 64) total = wave_sum(total);
    if (threadIdx.x == 0) shared[4] = total;
    __syncthreads();
    return shared[4];
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

__device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(BFloat16(value));
}

__device__ __forceinline__ float native_exp(float value) {
    return __builtin_amdgcn_exp2f(value * 1.4426950408889634f);
}

__device__ __forceinline__ float native_softplus(float value) {
    if (value > 20.0f) return value;
    float logarithm = __builtin_amdgcn_logf(1.0f + native_exp(value));
    float high = __uint_as_float(0x3f317217);
    float low = __uint_as_float(0x3377d1cf);
    float product = logarithm * high;
    float correction = fmaf(logarithm, high, -product);
    correction = fmaf(logarithm, low, correction);
    return fmaf(logarithm, high, correction);
}

__device__ __forceinline__ float prefix_sum64(float value) {
    value += __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(value), 280, 15, 15, true));
    value += __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(value), 276, 15, 15, true));
    value += __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(value), 274, 15, 15, true));
    value += __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(value), 273, 15, 15, true));
    value += __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(value), __float_as_uint(value), 322, 10, 15, true));
    value += __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(value), 323, 15, 15, true));
    return __shfl(value, 63, 64);
}

__device__ __forceinline__ float half_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) value += __shfl_xor(value, offset, 32);
    return value;
}

__device__ void normalize(const BFloat16* hidden, const BFloat16* residual,
                          const BFloat16* weight, BFloat16* normalized,
                          BFloat16* residual_output, int row_index, float* shared) {
    for (int virtual_thread = threadIdx.x; virtual_thread < 1024; virtual_thread += Threads) {
        float variance = 0.0f;
        for (int element = 0; element < 8; ++element) {
            int column = virtual_thread * 8 + element;
            float value = 0.0f;
            if (column < Hidden) {
                value = static_cast<float>(hidden[row_index * Hidden + column]);
                if (residual != nullptr) value += static_cast<float>(residual[row_index * Hidden + column]);
                residual_output[row_index * Hidden + column] = BFloat16(value);
            }
            variance = residual != nullptr && element == 7 ? fmaf(value, value, variance) : variance + value * value;
        }
        variance = wave_sum(variance);
        if ((threadIdx.x & 63) == 0) shared[virtual_thread / 64] = variance;
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        float variance = threadIdx.x < 16 ? shared[threadIdx.x] : 0.0f;
        for (int offset = 8; offset > 0; offset /= 2) variance += __shfl_down(variance, offset, 64);
        if (threadIdx.x == 0) shared[16] = __builtin_amdgcn_rsqf(static_cast<float>(static_cast<double>(variance) / Hidden) + 1.0e-6f);
    }
    __syncthreads();
    float inverse = shared[16];
    for (int column = threadIdx.x; column < Hidden; column += Threads) {
        float value = static_cast<float>(hidden[row_index * Hidden + column]);
        if (residual != nullptr) value += static_cast<float>(residual[row_index * Hidden + column]);
        float scale = residual != nullptr ? 1.0f + static_cast<float>(weight[column]) : rounded(1.0f + static_cast<float>(weight[column]));
        normalized[row_index * Hidden + column] = BFloat16(value * inverse * scale);
    }
}

__device__ float split_dot(const BFloat16* input, const BFloat16* weight, int width) {
    int lane = threadIdx.x & 63;
    float partials[8];
    for (int group = 0; group < 8; ++group) {
        Float4 accumulated = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int start = 0; start < width; start += 4096) {
            int column = start + group * 512 + lane * 8;
            Short4 left = {0, 0, 0, 0}, right = {0, 0, 0, 0};
            if (column < width) {
                left = *reinterpret_cast<const Short4*>(input + column);
                right = *reinterpret_cast<const Short4*>(weight + column);
            }
            accumulated = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulated);
            if (column < width) {
                left = *reinterpret_cast<const Short4*>(input + column + 4);
                right = *reinterpret_cast<const Short4*>(weight + column + 4);
            }
            accumulated = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulated);
        }
        float diagonal[16];
        for (int index = 0; index < 16; ++index)
            diagonal[index] = __shfl(accumulated[index / 4], (index % 4) * 17 + (index / 4) * 4, 64);
        int base = lane / 16 * 4;
        Float2 left = {diagonal[base], diagonal[base + 1]}, ones = {1.0f, 1.0f};
        Float4 reduced = {0.0f, 0.0f, 0.0f, 0.0f};
        reduced = __builtin_hcu_mmac_16x16x8_f32(left, ones, reduced);
        left = {diagonal[base + 2], diagonal[base + 3]};
        reduced = __builtin_hcu_mmac_16x16x8_f32(left, ones, reduced);
        partials[group] = reduced[0];
    }
    int base = lane / 16 * 2;
    Float2 left = {partials[base], partials[base + 1]}, ones = {1.0f, 1.0f};
    Float4 reduced = {0.0f, 0.0f, 0.0f, 0.0f};
    reduced = __builtin_hcu_mmac_16x16x8_f32(left, ones, reduced);
    return reduced[0];
}

__device__ __forceinline__ float matrix_tile(const BFloat16* input, const BFloat16* weight, int width, int rows, int output_base, MatvecMode mode) {
    int lane = threadIdx.x & 63;
    if (mode == MatvecMode::Split) {
        float value = 0.0f;
        for (int row_index = 0; row_index < rows; ++row_index) {
            for (int output_index = 0; output_index < 16; ++output_index) {
                float projected = split_dot(input + row_index * width, weight + int64_t(output_index) * width, width);
                if (lane == row_index * 16 + output_index) value = projected;
            }
        }
        return value;
    }
    int weight_row = lane % 16;
    int input_row = lane % 16;
    if (mode == MatvecMode::Gating) {
        float partials[8];
        int rotation = ((output_base - Projection) / 32) * 1024;
        for (int group = 0; group < 8; ++group) {
            Float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int start = group * 32; start < width; start += 256) {
                for (int half = 0; half < 2; ++half) {
                    int column = (start + rotation) % width + half * 16 + (lane / 16) * 4;
                    Short4 left = *reinterpret_cast<const Short4*>(weight + int64_t(weight_row) * width + column);
                    Short4 right = {0, 0, 0, 0};
                    if (input_row < rows) right = *reinterpret_cast<const Short4*>(input + input_row * width + column);
                    accumulator = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulator);
                }
            }
            partials[group] = accumulator[0];
        }
        float value = partials[0];
        for (int group = 1; group < 8; ++group) value += partials[group];
        return value;
    }
    int rotation = mode == MatvecMode::Staggered ? ((output_base / 64) & 31) * 128 : 0;
    Float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int start = 0; start < width; start += 16) {
        int column = (start + rotation) % width + (lane / 16) * 4;
        Short4 left = *reinterpret_cast<const Short4*>(weight + int64_t(weight_row) * width + column);
        Short4 right = {0, 0, 0, 0};
        if (input_row < rows) right = *reinterpret_cast<const Short4*>(input + input_row * width + column);
        accumulator = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulator);
    }
    return accumulator[0];
}

__device__ __forceinline__ void run_dense_k1(K1Arguments args) {
    __shared__ float reduction[17];
    __shared__ float query[Head];
    __shared__ float key[Head];
    __shared__ float head_output[Head];
    if (blockIdx.x < args.rows)
        normalize(args.hidden, args.residual, args.norm_weight, args.normalized, args.residual_output, blockIdx.x, reduction);
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 1)) return;
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    int lane = threadIdx.x & 63;
    for (int work = wave; work < (Projection + 96) / 16; work += Blocks * 4) {
        int output_base = work * 16;
        int row_index = lane / 16;
        int output_index = output_base + lane % 16;
        const BFloat16* weight = output_base < Projection ? args.qkvz_weight + int64_t(output_base) * Hidden : args.ba_weight + int64_t(output_base - Projection) * Hidden;
        MatvecMode mode = args.rows == 2 ? MatvecMode::Split : (output_base >= Projection ? MatvecMode::Gating : MatvecMode::Staggered);
        float value = matrix_tile(args.normalized, weight, Hidden, args.rows, output_base, mode);
        if (row_index < args.rows) {
            if (output_index < Projection) args.projected[row_index * Projection + output_index] = BFloat16(value);
            else args.projected_ba[row_index * 96 + output_index - Projection] = BFloat16(value);
        }
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 2)) return;
    for (int work = blockIdx.x * Threads + threadIdx.x; work < args.rows * Conv; work += Blocks * Threads) {
        int row_index = work / Conv;
        int channel = work % Conv;
        int64_t slot = args.index_bytes == 8 ? static_cast<const int64_t*>(args.cache_indices)[row_index] : static_cast<const int32_t*>(args.cache_indices)[row_index];
        float current = static_cast<float>(args.projected[row_index * Projection + channel]);
        if (slot < 0 || slot >= args.slot_count) {
            args.convolved[work] = BFloat16(0.0f);
            continue;
        }
        BFloat16* history = args.conv_state + slot * args.conv_slot_stride + channel * args.conv_channel_stride;
        float oldest = static_cast<float>(history[0]);
        float middle = static_cast<float>(history[args.conv_history_stride]);
        float newest = static_cast<float>(history[2 * args.conv_history_stride]);
        float value = oldest * static_cast<float>(args.conv_weight[channel * 4]);
        value = fmaf(middle, static_cast<float>(args.conv_weight[channel * 4 + 1]), value);
        value = fmaf(newest, static_cast<float>(args.conv_weight[channel * 4 + 2]), value);
        value = fmaf(current, static_cast<float>(args.conv_weight[channel * 4 + 3]), value);
        history[0] = BFloat16(middle);
        history[args.conv_history_stride] = BFloat16(newest);
        history[2 * args.conv_history_stride] = BFloat16(current);
        args.convolved[work] = BFloat16(value * __builtin_amdgcn_rcpf(1.0f + native_exp(-value)));
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 3)) return;
    for (int work = blockIdx.x; work < args.rows * ValueHeads; work += Blocks) {
        int row_index = work / ValueHeads;
        int value_head = work % ValueHeads;
        int key_head = value_head / 3;
        int64_t slot = args.index_bytes == 8 ? static_cast<const int64_t*>(args.cache_indices)[row_index] : static_cast<const int32_t*>(args.cache_indices)[row_index];
        if (slot < 0 || slot >= args.slot_count) {
            if (threadIdx.x < Head) {
                args.recurrent[row_index * Core + value_head * Head + threadIdx.x] = BFloat16(0.0f);
                args.output[row_index * Core + value_head * Head + threadIdx.x] = BFloat16(0.0f);
            }
            __syncthreads();
            continue;
        }
        if (threadIdx.x < 64) {
            int channel = lane * 2;
            float query_first = static_cast<float>(args.convolved[row_index * Conv + key_head * Head + channel]);
            float query_second = static_cast<float>(args.convolved[row_index * Conv + key_head * Head + channel + 1]);
            float key_first = static_cast<float>(args.convolved[row_index * Conv + Key + key_head * Head + channel]);
            float key_second = static_cast<float>(args.convolved[row_index * Conv + Key + key_head * Head + channel + 1]);
            float query_inverse = __builtin_amdgcn_rsqf(prefix_sum64(query_first * query_first + query_second * query_second) + 1.0e-6f);
            float key_inverse = __builtin_amdgcn_rsqf(prefix_sum64(key_first * key_first + key_second * key_second) + 1.0e-6f);
            query[channel] = query_first * query_inverse;
            query[channel + 1] = query_second * query_inverse;
            key[channel] = key_first * key_inverse;
            key[channel + 1] = key_second * key_inverse;
        }
        __syncthreads();
        float gate_input = static_cast<float>(args.projected_ba[row_index * 96 + ValueHeads + value_head]) + static_cast<float>(args.dt_bias[value_head]);
        float softplus = native_softplus(gate_input);
        float decay = native_exp(-native_exp(args.a_log[value_head]) * softplus);
        float beta = 1.0f / (1.0f + native_exp(-static_cast<float>(args.projected_ba[row_index * 96 + value_head])));
        int key_start = (threadIdx.x & 31) * 4;
        for (int value_index = threadIdx.x / 32; value_index < Head; value_index += Threads / 32) {
            int64_t offset = slot * args.ssm_slot_stride + value_head * Head * Head + value_index * Head + key_start;
            float state[4], products[4];
            for (int element = 0; element < 4; ++element) {
                state[element] = args.ssm_state[offset + element] * decay;
                products[element] = state[element] * key[key_start + element];
            }
            float prediction = half_sum(((products[0] + products[1]) + products[2]) + products[3]);
            float value = static_cast<float>(args.convolved[row_index * Conv + 2 * Key + value_head * Head + value_index]);
            float delta = (value - prediction) * beta;
            for (int element = 0; element < 4; ++element) {
                state[element] = fmaf(delta, key[key_start + element], state[element]);
                args.ssm_state[offset + element] = state[element];
                products[element] = state[element] * query[key_start + element];
            }
            float result = half_sum(((products[0] + products[1]) + products[2]) + products[3]) * 0.08838834764831845f;
            if ((threadIdx.x & 31) == 0) {
                BFloat16 recurrent = BFloat16(result);
                args.recurrent[row_index * Core + value_head * Head + value_index] = recurrent;
                head_output[value_index] = static_cast<float>(recurrent);
            }
        }
        __syncthreads();
        if (threadIdx.x < 64) {
            int start = threadIdx.x * 2;
            float variance = prefix_sum64(head_output[start] * head_output[start] + head_output[start + 1] * head_output[start + 1]);
            float inverse = __builtin_amdgcn_rsqf(variance / Head + 1.0e-6f);
            for (int element = 0; element < 2; ++element) {
                int channel = start + element;
                float z_value = static_cast<float>(args.projected[row_index * Projection + Conv + value_head * Head + channel]);
                float gate = z_value * (1.0f / (1.0f + native_exp(-z_value)));
                args.output[row_index * Core + value_head * Head + channel] = BFloat16(head_output[channel] * inverse * static_cast<float>(args.gated_weight[channel]) * gate);
            }
        }
        __syncthreads();
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 4)) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(args.executions, 1);
}

__global__ __launch_bounds__(Threads, 1) void dense_k1(K1Arguments args) {
    run_dense_k1(args);
}

__global__ __launch_bounds__(Threads, 1) void dense_k2(K2Arguments args) {
    __shared__ float reduction[17];
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    int lane = threadIdx.x & 63;
    int row_index = lane / 16;
    for (int work = wave; work < Hidden / 16; work += Blocks * 4) {
        int output_index = work * 16 + lane % 16;
        float value = matrix_tile(args.core, args.output_weight + int64_t(work * 16) * Core, Core, args.rows, work * 16, MatvecMode::Split);
        if (row_index < args.rows) args.attention_output[row_index * Hidden + output_index] = BFloat16(value);
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 1)) return;
    if (blockIdx.x < args.rows)
        normalize(args.attention_output, args.residual, args.norm_weight, args.normalized, args.residual_output, blockIdx.x, reduction);
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 2)) return;
    for (int work = wave; work < Intermediate / 16; work += Blocks * 4) {
        int output_index = work * 16 + lane % 16;
        const BFloat16* gate_weight = args.gate_up_weight + int64_t(work * 16) * Hidden;
        const BFloat16* up_weight = gate_weight + int64_t(Intermediate) * Hidden;
        MatvecMode mode = args.rows == 2 ? MatvecMode::Split : MatvecMode::Staggered;
        float gate = matrix_tile(args.normalized, gate_weight, Hidden, args.rows, work * 16, mode);
        float up = matrix_tile(args.normalized, up_weight, Hidden, args.rows, work * 16 + Intermediate, mode);
        if (row_index < args.rows) {
            args.projected[row_index * 2 * Intermediate + output_index] = BFloat16(gate);
            args.projected[row_index * 2 * Intermediate + Intermediate + output_index] = BFloat16(up);
            args.activated[row_index * Intermediate + output_index] = BFloat16(rounded(silu(rounded(gate))) * rounded(up));
        }
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 3)) return;
    for (int work = wave; work < Hidden / 16; work += Blocks * 4) {
        int output_index = work * 16 + lane % 16;
        float value = matrix_tile(args.activated, args.down_weight + int64_t(work * 16) * Intermediate, Intermediate, args.rows, work * 16, MatvecMode::Split);
        if (row_index < args.rows) args.output[row_index * Hidden + output_index] = BFloat16(value);
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 4)) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(args.executions, 1);
}

extern "C" int mono_info(int64_t* information) {
    int device_index;
    hipError_t status = hipGetDevice(&device_index);
    if (status != hipSuccess) return status;
    int cooperative, multiprocessors, k1_occupancy, k2_occupancy;
    status = hipDeviceGetAttribute(&cooperative, hipDeviceAttributeCooperativeLaunch, device_index);
    if (status != hipSuccess) return status;
    status = hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, device_index);
    if (status != hipSuccess) return status;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&k1_occupancy, dense_k1, Threads, 0);
    if (status != hipSuccess) return status;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&k2_occupancy, dense_k2, Threads, 0);
    if (status != hipSuccess) return status;
    information[0] = sizeof(K1Arguments);
    information[1] = sizeof(K2Arguments);
    information[2] = cooperative;
    information[3] = multiprocessors;
    information[4] = k1_occupancy;
    information[5] = k2_occupancy;
    int wall_clock_rate;
    status = hipDeviceGetAttribute(&wall_clock_rate, hipDeviceAttributeWallClockRate, device_index);
    if (status != hipSuccess) return status;
    information[6] = wall_clock_rate;
    return hipSuccess;
}

hipError_t launch_cooperative(const void* kernel, void** arguments, void* stream, int blocks = Blocks) {
    hipStream_t launch_stream = reinterpret_cast<hipStream_t>(stream);
    hipStreamCaptureStatus capture_status;
    hipGraph_t graph;
    const hipGraphNode_t* dependencies;
    size_t dependency_count;
    hipError_t status = hipStreamGetCaptureInfo_v2(launch_stream, &capture_status, nullptr, &graph, &dependencies, &dependency_count);
    if (status != hipSuccess) return status;
    if (capture_status != hipStreamCaptureStatusNone) {
        if (capture_status != hipStreamCaptureStatusActive) return hipErrorStreamCaptureUnsupported;
        hipKernelNodeParams parameters{};
        parameters.func = const_cast<void*>(kernel);
        parameters.gridDim = dim3(blocks);
        parameters.blockDim = dim3(Threads);
        parameters.kernelParams = arguments;
        hipGraphNode_t node;
        status = hipGraphAddKernelNode(&node, graph, dependencies, dependency_count, &parameters);
        if (status != hipSuccess) return status;
        hipKernelNodeAttrValue graph_attribute{};
        graph_attribute.cooperative = 1;
        status = hipGraphKernelNodeSetAttribute(node, hipKernelNodeAttributeCooperative, &graph_attribute);
        if (status != hipSuccess) return status;
        graph_attribute.cooperative = 0;
        status = hipGraphKernelNodeGetAttribute(node, hipKernelNodeAttributeCooperative, &graph_attribute);
        if (status != hipSuccess) return status;
        if (!graph_attribute.cooperative) return hipErrorStreamCaptureUnsupported;
        return hipStreamUpdateCaptureDependencies(launch_stream, &node, 1, hipStreamSetCaptureDependencies);
    }
    hipLaunchAttribute attribute{};
    attribute.id = hipLaunchAttributeCooperative;
    attribute.val.cooperative = 1;
    hipLaunchConfig_t config{};
    config.gridDim = dim3(blocks);
    config.blockDim = dim3(Threads);
    config.stream = launch_stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return hipLaunchKernelExC(&config, kernel, arguments);
}

__global__ __launch_bounds__(Threads, 1) void probe_synchronization(unsigned int* state, int64_t timeout_ticks) {
    synchronize_grid(state, timeout_ticks, 1);
}

extern "C" int mono_probe_synchronization(unsigned int* state, int64_t timeout_ticks, int blocks, void* stream) {
    if (blocks < 1 || blocks > Blocks || timeout_ticks < 1) return hipErrorInvalidValue;
    void* arguments[] = {&state, &timeout_ticks};
    return launch_cooperative(reinterpret_cast<const void*>(probe_synchronization), arguments, stream, blocks);
}

extern "C" int mono_k1(K1Arguments* args, void* stream) {
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(dense_k1), arguments, stream);
}

extern "C" int mono_k2(K2Arguments* args, void* stream) {
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(dense_k2), arguments, stream);
}
