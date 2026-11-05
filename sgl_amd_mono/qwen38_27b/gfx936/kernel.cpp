#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdint>
#include <cmath>

using BFloat16 = hip_bfloat16;
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
    BFloat16* output;
    BFloat16* residual_output;
    int* executions;
    int64_t rows;
    int64_t conv_slot_stride;
    int64_t conv_channel_stride;
    int64_t conv_history_stride;
    int64_t ssm_slot_stride;
    int64_t index_bytes;
    int64_t slot_count;
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
    BFloat16* activated;
    BFloat16* output;
    BFloat16* residual_output;
    int* executions;
    int64_t rows;
};

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

__device__ void normalize(const BFloat16* hidden, const BFloat16* residual,
                          const BFloat16* weight, BFloat16* normalized,
                          BFloat16* residual_output, int row_index, float* shared) {
    float variance = 0.0f;
    for (int column = threadIdx.x; column < Hidden; column += Threads) {
        float value = static_cast<float>(hidden[row_index * Hidden + column]);
        if (residual != nullptr) value += static_cast<float>(residual[row_index * Hidden + column]);
        residual_output[row_index * Hidden + column] = BFloat16(value);
        variance += value * value;
    }
    float inverse = rsqrtf(block_sum(variance, shared) / Hidden + 1.0e-6f);
    for (int column = threadIdx.x; column < Hidden; column += Threads) {
        float value = static_cast<float>(hidden[row_index * Hidden + column]);
        if (residual != nullptr) value += static_cast<float>(residual[row_index * Hidden + column]);
        float scale = residual != nullptr ? 1.0f + static_cast<float>(weight[column]) : rounded(1.0f + static_cast<float>(weight[column]));
        normalized[row_index * Hidden + column] = BFloat16(value * inverse * scale);
    }
}

__device__ __forceinline__ float dot(const BFloat16* input, const BFloat16* weight, int width) {
    float total = 0.0f;
    for (int column = threadIdx.x & 63; column < width; column += 64)
        total = fmaf(static_cast<float>(input[column]), static_cast<float>(weight[column]), total);
    return wave_sum(total);
}

__global__ __launch_bounds__(Threads, 1) void dense_k1(K1Arguments args) {
    auto grid = cooperative_groups::this_grid();
    __shared__ float reduction[5];
    __shared__ float query[Head];
    __shared__ float key[Head];
    __shared__ float head_output[Head];
    if (blockIdx.x < args.rows)
        normalize(args.hidden, args.residual, args.norm_weight, args.normalized, args.residual_output, blockIdx.x, reduction);
    grid.sync();
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    for (int work = wave; work < args.rows * (Projection + 96); work += Blocks * 4) {
        int row_index = work / (Projection + 96);
        int output_index = work % (Projection + 96);
        const BFloat16* weight = output_index < Projection ? args.qkvz_weight + int64_t(output_index) * Hidden : args.ba_weight + int64_t(output_index - Projection) * Hidden;
        float value = dot(args.normalized + row_index * Hidden, weight, Hidden);
        if ((threadIdx.x & 63) == 0) {
            if (output_index < Projection) args.projected[row_index * Projection + output_index] = BFloat16(value);
            else args.projected_ba[row_index * 96 + output_index - Projection] = BFloat16(value);
        }
    }
    grid.sync();
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
        args.convolved[work] = BFloat16(silu(value));
    }
    grid.sync();
    for (int work = blockIdx.x; work < args.rows * ValueHeads; work += Blocks) {
        int row_index = work / ValueHeads;
        int value_head = work % ValueHeads;
        int key_head = value_head / 3;
        int64_t slot = args.index_bytes == 8 ? static_cast<const int64_t*>(args.cache_indices)[row_index] : static_cast<const int32_t*>(args.cache_indices)[row_index];
        if (slot < 0 || slot >= args.slot_count) {
            if (threadIdx.x < Head) args.output[row_index * Core + value_head * Head + threadIdx.x] = BFloat16(0.0f);
            __syncthreads();
            continue;
        }
        float query_value = 0.0f;
        float key_value = 0.0f;
        if (threadIdx.x < Head) {
            query_value = static_cast<float>(args.convolved[row_index * Conv + key_head * Head + threadIdx.x]);
            key_value = static_cast<float>(args.convolved[row_index * Conv + Key + key_head * Head + threadIdx.x]);
        }
        float query_inverse = rsqrtf(block_sum(query_value * query_value, reduction) + 1.0e-6f);
        float key_inverse = rsqrtf(block_sum(key_value * key_value, reduction) + 1.0e-6f);
        if (threadIdx.x < Head) {
            query[threadIdx.x] = query_value * query_inverse;
            key[threadIdx.x] = key_value * key_inverse;
        }
        __syncthreads();
        float gate_input = static_cast<float>(args.projected_ba[row_index * 96 + ValueHeads + value_head]) + static_cast<float>(args.dt_bias[value_head]);
        float softplus = gate_input <= 20.0f ? log1pf(expf(gate_input)) : gate_input;
        float decay = expf(-expf(args.a_log[value_head]) * softplus);
        float beta = 1.0f / (1.0f + expf(-static_cast<float>(args.projected_ba[row_index * 96 + value_head])));
        int lane = threadIdx.x & 63;
        for (int value_index = threadIdx.x / 64; value_index < Head; value_index += 4) {
            int64_t offset = slot * args.ssm_slot_stride + value_head * Head * Head + value_index * Head + lane;
            float state_first = args.ssm_state[offset] * decay;
            float state_second = args.ssm_state[offset + 64] * decay;
            float prediction = wave_sum(state_first * key[lane] + state_second * key[lane + 64]);
            float value = static_cast<float>(args.convolved[row_index * Conv + 2 * Key + value_head * Head + value_index]);
            float delta = (value - prediction) * beta;
            state_first = fmaf(delta, key[lane], state_first);
            state_second = fmaf(delta, key[lane + 64], state_second);
            args.ssm_state[offset] = state_first;
            args.ssm_state[offset + 64] = state_second;
            float result = wave_sum(state_first * query[lane] + state_second * query[lane + 64]) * 0.08838834764831845f;
            if (lane == 0) head_output[value_index] = rounded(result);
        }
        __syncthreads();
        float variance = threadIdx.x < Head ? head_output[threadIdx.x] * head_output[threadIdx.x] : 0.0f;
        float inverse = rsqrtf(block_sum(variance, reduction) / Head + 1.0e-6f);
        if (threadIdx.x < Head) {
            float z_value = static_cast<float>(args.projected[row_index * Projection + Conv + value_head * Head + threadIdx.x]);
            args.output[row_index * Core + value_head * Head + threadIdx.x] = BFloat16(head_output[threadIdx.x] * inverse * static_cast<float>(args.gated_weight[threadIdx.x]) * silu(z_value));
        }
        __syncthreads();
    }
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(args.executions, 1);
}

__global__ __launch_bounds__(Threads, 1) void dense_k2(K2Arguments args) {
    auto grid = cooperative_groups::this_grid();
    __shared__ float reduction[5];
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    for (int work = wave; work < args.rows * Hidden; work += Blocks * 4) {
        int row_index = work / Hidden;
        int output_index = work % Hidden;
        float value = dot(args.core + row_index * Core, args.output_weight + int64_t(output_index) * Core, Core);
        if ((threadIdx.x & 63) == 0) args.attention_output[work] = BFloat16(value);
    }
    grid.sync();
    if (blockIdx.x < args.rows)
        normalize(args.attention_output, args.residual, args.norm_weight, args.normalized, args.residual_output, blockIdx.x, reduction);
    grid.sync();
    for (int work = wave; work < args.rows * Intermediate; work += Blocks * 4) {
        int row_index = work / Intermediate;
        int output_index = work % Intermediate;
        const BFloat16* input = args.normalized + row_index * Hidden;
        const BFloat16* gate_weight = args.gate_up_weight + int64_t(output_index) * Hidden;
        const BFloat16* up_weight = gate_weight + int64_t(Intermediate) * Hidden;
        float gate = 0.0f;
        float up = 0.0f;
        for (int column = threadIdx.x & 63; column < Hidden; column += 64) {
            float value = static_cast<float>(input[column]);
            gate = fmaf(value, static_cast<float>(gate_weight[column]), gate);
            up = fmaf(value, static_cast<float>(up_weight[column]), up);
        }
        gate = wave_sum(gate);
        up = wave_sum(up);
        if ((threadIdx.x & 63) == 0) args.activated[work] = BFloat16(rounded(silu(rounded(gate))) * rounded(up));
    }
    grid.sync();
    for (int work = wave; work < args.rows * Hidden; work += Blocks * 4) {
        int row_index = work / Hidden;
        int output_index = work % Hidden;
        float value = dot(args.activated + row_index * Intermediate, args.down_weight + int64_t(output_index) * Intermediate, Intermediate);
        if ((threadIdx.x & 63) == 0) args.output[work] = BFloat16(value);
    }
    grid.sync();
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
    return hipSuccess;
}

hipError_t launch_cooperative(const void* kernel, void** arguments, void* stream) {
    hipLaunchAttribute attribute{};
    attribute.id = hipLaunchAttributeCooperative;
    attribute.val.cooperative = 1;
    hipLaunchConfig_t config{};
    config.gridDim = dim3(Blocks);
    config.blockDim = dim3(Threads);
    config.stream = reinterpret_cast<hipStream_t>(stream);
    config.attrs = &attribute;
    config.numAttrs = 1;
    return hipLaunchKernelExC(&config, kernel, arguments);
}

extern "C" int mono_k1(K1Arguments* args, void* stream) {
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(dense_k1), arguments, stream);
}

extern "C" int mono_k2(K2Arguments* args, void* stream) {
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(dense_k2), arguments, stream);
}
