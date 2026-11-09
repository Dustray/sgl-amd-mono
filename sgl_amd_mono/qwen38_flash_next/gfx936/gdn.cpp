#include "kernel.cpp"

constexpr int LocalKey = 256;
constexpr int LocalCore = 768;
constexpr int LocalConv = 1280;
constexpr int LocalProjection = 2048;
constexpr int LocalHeads = 6;
constexpr int Head = 128;

struct GDNArguments {
    MixArguments mix;
    const BFloat16* qkvz_weight;
    const BFloat16* ba_weight;
    const BFloat16* conv_weight;
    const BFloat16* gated_weight;
    const float* a_log;
    const BFloat16* dt_bias;
    const BFloat16* output_weight;
    BFloat16* conv_state;
    float* ssm_state;
    const void* indices;
    BFloat16* projected;
    BFloat16* projected_ba;
    BFloat16* convolved;
    BFloat16* recurrent;
    BFloat16* core;
    BFloat16* output;
    int* executions;
    int64_t conv_slot_stride;
    int64_t conv_channel_stride;
    int64_t conv_history_stride;
    int64_t ssm_slot_stride;
    int64_t index_bytes;
    int64_t slot_count;
};

__device__ __forceinline__ float gdn_prefix_sum(float value) {
    value += __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(value), 280, 15, 15, true));
    value += __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(value), 276, 15, 15, true));
    value += __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(value), 274, 15, 15, true));
    value += __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(value), 273, 15, 15, true));
    value += __uint_as_float(__builtin_amdgcn_update_dpp(__float_as_uint(value), __float_as_uint(value), 322, 10, 15, true));
    value += __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(value), 323, 15, 15, true));
    return __shfl(value, 63, 64);
}

__device__ __forceinline__ float gdn_half_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) value += __shfl_xor(value, offset, 32);
    return value;
}

__device__ __forceinline__ float gdn_exp(float value) {
    return __builtin_amdgcn_exp2f(value * 1.4426950408889634f);
}

__device__ __forceinline__ float gdn_softplus(float value) {
    if (value > 20.0f) return value;
    float logarithm = __builtin_amdgcn_logf(1.0f + gdn_exp(value));
    float high = __uint_as_float(0x3f317217);
    float low = __uint_as_float(0x3377d1cf);
    float product = logarithm * high;
    float correction = fmaf(logarithm, high, -product);
    correction = fmaf(logarithm, low, correction);
    return fmaf(logarithm, high, correction);
}

__global__ __launch_bounds__(Threads, 1) void gdn_attention(GDNArguments args) {
    __shared__ float query[Head];
    __shared__ float key[Head];
    __shared__ float head_output[Head];
    if (!run_mix(args.mix, 300)) return;
    int rows = args.mix.rows;
    gemv(args.mix.output, args.qkvz_weight, args.projected, nullptr, rows, Hidden, LocalProjection, 0);
    gemv(args.mix.output, args.ba_weight, args.projected_ba, nullptr, rows, Hidden, LocalHeads * 2, 0);
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 310)) return;
    for (int work = blockIdx.x * Threads + threadIdx.x; work < rows * LocalConv; work += Blocks * Threads) {
        int row = work / LocalConv;
        int channel = work % LocalConv;
        int64_t slot = args.index_bytes == 8 ? static_cast<const int64_t*>(args.indices)[row] : static_cast<const int32_t*>(args.indices)[row];
        if (slot < 0 || slot >= args.slot_count) {
            args.convolved[work] = BFloat16(0.0f);
            continue;
        }
        BFloat16* history = args.conv_state + slot * args.conv_slot_stride + channel * args.conv_channel_stride;
        float current = float(args.projected[row * LocalProjection + channel]);
        float oldest = float(history[0]);
        float middle = float(history[args.conv_history_stride]);
        float newest = float(history[2 * args.conv_history_stride]);
        float value = oldest * float(args.conv_weight[channel * 4]);
        value = fmaf(middle, float(args.conv_weight[channel * 4 + 1]), value);
        value = fmaf(newest, float(args.conv_weight[channel * 4 + 2]), value);
        value = fmaf(current, float(args.conv_weight[channel * 4 + 3]), value);
        history[0] = BFloat16(middle);
        history[args.conv_history_stride] = BFloat16(newest);
        history[2 * args.conv_history_stride] = BFloat16(current);
        args.convolved[work] = BFloat16(value * __builtin_amdgcn_rcpf(1.0f + gdn_exp(-value)));
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 311)) return;
    for (int work = blockIdx.x; work < rows * LocalHeads; work += Blocks) {
        int row = work / LocalHeads;
        int value_head = work % LocalHeads;
        int key_head = value_head / 3;
        int lane = threadIdx.x & 63;
        int64_t slot = args.index_bytes == 8 ? static_cast<const int64_t*>(args.indices)[row] : static_cast<const int32_t*>(args.indices)[row];
        if (slot < 0 || slot >= args.slot_count) {
            if (threadIdx.x < Head) {
                args.recurrent[row * LocalCore + value_head * Head + threadIdx.x] = BFloat16(0.0f);
                args.core[row * LocalCore + value_head * Head + threadIdx.x] = BFloat16(0.0f);
            }
            __syncthreads();
            continue;
        }
        if (threadIdx.x < 64) {
            int channel = lane * 2;
            float query_first = float(args.convolved[row * LocalConv + key_head * Head + channel]);
            float query_second = float(args.convolved[row * LocalConv + key_head * Head + channel + 1]);
            float key_first = float(args.convolved[row * LocalConv + LocalKey + key_head * Head + channel]);
            float key_second = float(args.convolved[row * LocalConv + LocalKey + key_head * Head + channel + 1]);
            float query_inverse = __builtin_amdgcn_rsqf(gdn_prefix_sum(query_first * query_first + query_second * query_second) + 1.0e-6f);
            float key_inverse = __builtin_amdgcn_rsqf(gdn_prefix_sum(key_first * key_first + key_second * key_second) + 1.0e-6f);
            query[channel] = query_first * query_inverse;
            query[channel + 1] = query_second * query_inverse;
            key[channel] = key_first * key_inverse;
            key[channel + 1] = key_second * key_inverse;
        }
        __syncthreads();
        float gate_input = float(args.projected_ba[row * 12 + LocalHeads + value_head]) + float(args.dt_bias[value_head]);
        float decay = gdn_exp(-gdn_exp(args.a_log[value_head]) * gdn_softplus(gate_input));
        float beta = 1.0f / (1.0f + gdn_exp(-float(args.projected_ba[row * 12 + value_head])));
        int key_start = (threadIdx.x & 31) * 4;
        for (int value_index = threadIdx.x / 32; value_index < Head; value_index += Threads / 32) {
            int64_t offset = slot * args.ssm_slot_stride + value_head * Head * Head + value_index * Head + key_start;
            float state[4], products[4];
            for (int element = 0; element < 4; ++element) {
                state[element] = args.ssm_state[offset + element] * decay;
                products[element] = state[element] * key[key_start + element];
            }
            float prediction = gdn_half_sum(((products[0] + products[1]) + products[2]) + products[3]);
            float value = float(args.convolved[row * LocalConv + 2 * LocalKey + value_head * Head + value_index]);
            float delta = (value - prediction) * beta;
            for (int element = 0; element < 4; ++element) {
                state[element] = fmaf(delta, key[key_start + element], state[element]);
                args.ssm_state[offset + element] = state[element];
                products[element] = state[element] * query[key_start + element];
            }
            float result = gdn_half_sum(((products[0] + products[1]) + products[2]) + products[3]) * 0.08838834764831845f;
            if ((threadIdx.x & 31) == 0) {
                BFloat16 recurrent = BFloat16(result);
                args.recurrent[row * LocalCore + value_head * Head + value_index] = recurrent;
                head_output[value_index] = float(recurrent);
            }
        }
        __syncthreads();
        if (threadIdx.x < 64) {
            int start = threadIdx.x * 2;
            float variance = gdn_prefix_sum(head_output[start] * head_output[start] + head_output[start + 1] * head_output[start + 1]);
            float inverse = __builtin_amdgcn_rsqf(variance / Head + 1.0e-6f);
            for (int element = 0; element < 2; ++element) {
                int channel = start + element;
                float z = float(args.projected[row * LocalProjection + LocalConv + value_head * Head + channel]);
                float gate = 1.0f / (1.0f + gdn_exp(-z));
                args.core[row * LocalCore + value_head * Head + channel] = BFloat16(head_output[channel] * inverse * float(args.gated_weight[channel]) * gate);
            }
        }
        __syncthreads();
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 312)) return;
    gemv(args.core, args.output_weight, args.output, nullptr, rows, LocalCore, Hidden, 0);
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 313)) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(args.executions, 1);
        atomicAdd(args.mix.executions, 1);
    }
    return;
}

extern "C" int flash_gdn_info(int64_t* result) {
    int occupancy;
    hipError_t status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, gdn_attention, Threads, 0);
    if (status != hipSuccess) return status;
    result[0] = sizeof(GDNArguments);
    result[1] = occupancy;
    return hipSuccess;
}

extern "C" int flash_gdn(GDNArguments* args, void* stream) {
    if (args->mix.rows < 1 || args->mix.rows > MaxRows || args->slot_count < 1 || (args->index_bytes != 4 && args->index_bytes != 8)) return hipErrorInvalidValue;
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(gdn_attention), arguments, stream);
}
