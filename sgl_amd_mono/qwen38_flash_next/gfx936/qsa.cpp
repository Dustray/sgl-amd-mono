#include "kernel.cpp"

constexpr int QSADim = 256;
constexpr int QSAHeads = 3;
constexpr int IndexDim = 128;
constexpr int IndexHeads = 4;
constexpr int QSAContext = 4096;
constexpr int QSABlocks = 1024;
constexpr int QSABudget = 2051;

struct QSAArguments {
    MixArguments mix;
    const BFloat16* qkv_weight;
    const BFloat16* q_norm;
    const BFloat16* k_norm;
    const BFloat16* output_weight;
    const BFloat16* index_weight;
    const BFloat16* index_q_norm;
    const BFloat16* index_k_norm;
    const float* rotary_cache;
    const int64_t* positions;
    const int32_t* sequence_lengths;
    const int64_t* requests;
    const int64_t* locations;
    const int32_t* request_tokens;
    BFloat16* key_cache;
    BFloat16* value_cache;
    BFloat16* index_ring;
    int64_t* ring_positions;
    BFloat16* compressed_keys;
    BFloat16* projected;
    BFloat16* index_projected;
    BFloat16* queries;
    BFloat16* index_queries;
    float* index_scores;
    int32_t* selected;
    float* attention_scores;
    BFloat16* core;
    BFloat16* output;
    int32_t* executions;
    int64_t request_stride;
    int64_t request_count;
    int64_t cache_slots;
    int64_t ring_slots;
    int64_t rotary_rows;
    int64_t position_axes;
    int64_t position_axis_stride;
    int64_t key_page_stride;
    int64_t key_token_stride;
    int64_t key_dim_stride;
    int64_t value_page_stride;
    int64_t value_token_stride;
    int64_t value_dim_stride;
};

__device__ __forceinline__ float qsa_round(float value) { return float(BFloat16(value)); }

__device__ __forceinline__ float qsa_sum(float value, float* shared) {
    value = wave_sum(value);
    if ((threadIdx.x & 63) == 0) shared[threadIdx.x / 64] = value;
    __syncthreads();
    float total = shared[0] + shared[1] + shared[2] + shared[3];
    __syncthreads();
    return total;
}

__device__ __forceinline__ int64_t qsa_position(QSAArguments args, int row, int pair) {
    int axis = 0;
    if (args.position_axes == 3) {
        if (pair % 3 == 1 && pair < 33) axis = 1;
        if (pair % 3 == 2 && pair < 30) axis = 2;
    }
    return args.positions[axis * args.position_axis_stride + row];
}

__device__ __forceinline__ float qsa_rotary(QSAArguments args, float* normalized, int column, int64_t position) {
    if (column >= 64) return normalized[column];
    int pair = column % 32;
    float cosine = qsa_round(args.rotary_cache[position * 64 + pair]);
    float sine = qsa_round(args.rotary_cache[position * 64 + 32 + pair]);
    float first = qsa_round(normalized[column] * cosine);
    float second = qsa_round(normalized[column < 32 ? column + 32 : column - 32] * sine);
    return qsa_round(column < 32 ? first - second : first + second);
}

__device__ __forceinline__ int64_t qsa_cache_offset(int64_t slot, int column, int64_t page_stride, int64_t token_stride, int64_t dim_stride) {
    return slot / 64 * page_stride + slot % 64 * token_stride + column * dim_stride;
}

__global__ __launch_bounds__(Threads, 1) void qsa_attention(QSAArguments args) {
    __shared__ float reduction[4];
    __shared__ float normalized[QSADim];
    __shared__ float sort_scores[QSABlocks];
    __shared__ int sort_indices[QSABlocks];
    __shared__ float probabilities[64];
    if (blockIdx.x < args.mix.rows) {
        int row = blockIdx.x;
        int length = args.sequence_lengths[row];
        int64_t request = args.requests[row];
        int64_t location = args.locations[row];
        bool valid = length > 0 && length <= QSAContext && request >= 0 && request < args.request_count;
        valid = valid && location >= 0 && location < args.cache_slots && request * 4 + 3 < args.ring_slots;
        for (int axis = 0; axis < args.position_axes; ++axis) {
            int64_t position = args.positions[axis * args.position_axis_stride + row];
            valid = valid && position >= 0 && position < args.rotary_rows;
        }
        if (valid) {
            valid = args.request_tokens[request * args.request_stride + length - 1] == location;
            for (int token = threadIdx.x; token < length; token += Threads) {
                int slot = args.request_tokens[request * args.request_stride + token];
                if (slot < 0 || slot >= args.cache_slots) valid = false;
            }
        }
        bool reserved = request == 0 && location == 0;
        if (!reserved && args.mix.rows == 2 && row == 0 && (args.requests[0] == args.requests[1] || args.locations[0] == args.locations[1])) valid = false;
        if (!valid) atomicCAS(args.mix.synchronization + 2, 0u, 400u);
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 400)) return;
    if (!run_mix(args.mix, 401)) return;
    gemv(args.mix.output, args.qkv_weight, args.projected, nullptr, args.mix.rows, Hidden, 2048, 0);
    gemv(args.mix.output, args.index_weight, args.index_projected, nullptr, args.mix.rows, Hidden, 640, 0);
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 405)) return;
    for (int work = blockIdx.x; work < args.mix.rows * 4; work += Blocks) {
        int row = work / 4;
        int head = work % 4;
        int column = threadIdx.x;
        bool is_query = head < 3;
        int offset = row * 2048 + (is_query ? head * 512 : 1536);
        float value = float(args.projected[offset + column]);
        float variance = qsa_sum(value * value, reduction) / QSADim;
        const BFloat16* weight = is_query ? args.q_norm : args.k_norm;
        normalized[column] = qsa_round(value * rsqrtf(variance + 1.0e-6f) * (1.0f + float(weight[column])));
        __syncthreads();
        BFloat16 rotated(qsa_rotary(args, normalized, column, qsa_position(args, row, column % 32)));
        if (is_query) args.queries[(row * 3 + head) * QSADim + column] = rotated;
        else {
            int64_t slot = args.locations[row];
            args.key_cache[qsa_cache_offset(slot, column, args.key_page_stride, args.key_token_stride, args.key_dim_stride)] = rotated;
            args.value_cache[qsa_cache_offset(slot, column, args.value_page_stride, args.value_token_stride, args.value_dim_stride)] = args.projected[row * 2048 + 1792 + column];
        }
        __syncthreads();
    }
    for (int work = blockIdx.x; work < args.mix.rows * IndexHeads; work += Blocks) {
        int row = work / IndexHeads;
        int head = work % IndexHeads;
        int column = threadIdx.x;
        float value = column < IndexDim ? float(args.index_projected[row * 640 + head * IndexDim + column]) : 0.0f;
        float inverse = rsqrtf(qsa_sum(value * value, reduction) / IndexDim + 1.0e-6f);
        if (column < IndexDim) normalized[column] = qsa_round(value * inverse * (1.0f + float(args.index_q_norm[column])));
        __syncthreads();
        if (column < IndexDim) args.index_queries[(row * IndexHeads + head) * IndexDim + column] = BFloat16(qsa_rotary(args, normalized, column, qsa_position(args, row, column % 32)));
        if (head == 0) {
            int64_t ring = args.requests[row] * 4 + (args.sequence_lengths[row] - 1) % 4;
            if (column < IndexDim) args.index_ring[ring * IndexDim + column] = args.index_projected[row * 640 + 512 + column];
            if (column < 3) args.ring_positions[ring * 3 + column] = args.positions[(args.position_axes == 1 ? 0 : column) * args.position_axis_stride + row];
        }
        __syncthreads();
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 406)) return;
    for (int row = blockIdx.x; row < args.mix.rows; row += Blocks) {
        if (args.sequence_lengths[row] % 4 == 0) {
            int column = threadIdx.x;
            int64_t ring_start = args.requests[row] * 4;
            float value = 0.0f;
            if (column < IndexDim) {
                for (int token = 0; token < 4; ++token) value += float(args.index_ring[(ring_start + token) * IndexDim + column]);
                value = qsa_round(value * 0.25f);
            }
            float inverse = rsqrtf(qsa_sum(value * value, reduction) / IndexDim + 1.0e-6f);
            if (column < IndexDim) normalized[column] = qsa_round(value * inverse * (1.0f + float(args.index_k_norm[column])));
            __syncthreads();
            if (column < IndexDim) {
                int pair = column % 32;
                int axis = args.position_axes == 1 ? 0 : (pair % 3 == 1 && pair < 33 ? 1 : (pair % 3 == 2 && pair < 30 ? 2 : 0));
                int64_t position = args.ring_positions[ring_start * 3 + axis];
                if (position < 0 || position >= args.rotary_rows) atomicCAS(args.mix.synchronization + 2, 0u, 407u);
                else args.compressed_keys[args.locations[row] / 4 * IndexDim + column] = BFloat16(qsa_rotary(args, normalized, column, position));
            }
            __syncthreads();
        }
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 408)) return;
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    int lane = threadIdx.x & 63;
    for (int work = wave; work < args.mix.rows * QSABlocks; work += Blocks * 4) {
        int row = work / QSABlocks;
        int group = work % QSABlocks;
        float score = -INFINITY;
        if (group < args.sequence_lengths[row] / 4) {
            int slot = args.request_tokens[args.requests[row] * args.request_stride + group * 4] / 4;
            score = 0.0f;
            for (int head = 0; head < IndexHeads; ++head) {
                float partial = 0.0f;
                for (int column = lane; column < IndexDim; column += 64) partial += float(args.index_queries[(row * IndexHeads + head) * IndexDim + column]) * float(args.compressed_keys[slot * IndexDim + column]);
                score += fmaxf(wave_sum(partial), 0.0f);
            }
        }
        if (lane == 0) args.index_scores[row * QSABlocks + group] = score;
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 409)) return;
    for (int row = blockIdx.x; row < args.mix.rows; row += Blocks) {
        for (int column = threadIdx.x; column < QSABlocks; column += Threads) {
            sort_scores[column] = args.index_scores[row * QSABlocks + column];
            sort_indices[column] = column;
        }
        __syncthreads();
        for (int width = 2; width <= QSABlocks; width *= 2) {
            for (int distance = width / 2; distance > 0; distance /= 2) {
                for (int column = threadIdx.x; column < QSABlocks; column += Threads) {
                    int partner = column ^ distance;
                    if (partner > column) {
                        bool descending = (column & width) == 0;
                        bool greater = sort_scores[partner] > sort_scores[column] || (sort_scores[partner] == sort_scores[column] && sort_indices[partner] < sort_indices[column]);
                        if (greater == descending) {
                            float score = sort_scores[column]; sort_scores[column] = sort_scores[partner]; sort_scores[partner] = score;
                            int index = sort_indices[column]; sort_indices[column] = sort_indices[partner]; sort_indices[partner] = index;
                        }
                    }
                }
                __syncthreads();
            }
        }
        int groups = min(args.sequence_lengths[row] / 4, 512);
        for (int column = threadIdx.x; column < QSABudget; column += Threads) {
            int token = -1;
            if (column < groups * 4) token = args.sequence_lengths[row] / 4 <= 512 ? column : sort_indices[column / 4] * 4 + column % 4;
            else if (column < groups * 4 + args.sequence_lengths[row] % 4) token = args.sequence_lengths[row] / 4 * 4 + column - groups * 4;
            args.selected[row * QSABudget + column] = token;
        }
        __syncthreads();
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 410)) return;
    for (int work = wave; work < args.mix.rows * QSAHeads * QSABudget; work += Blocks * 4) {
        int row_head = work / QSABudget;
        int row = row_head / QSAHeads;
        int selected = args.selected[row * QSABudget + work % QSABudget];
        float score = -INFINITY;
        if (selected >= 0) {
            int slot = args.request_tokens[args.requests[row] * args.request_stride + selected];
            float partial = 0.0f;
            for (int column = lane; column < QSADim; column += 64) {
                float query = qsa_round(float(args.queries[row_head * QSADim + column]) * 0.0625f * 1.4426950408f);
                partial += query * float(args.key_cache[qsa_cache_offset(slot, column, args.key_page_stride, args.key_token_stride, args.key_dim_stride)]);
            }
            score = wave_sum(partial);
        }
        if (lane == 0) args.attention_scores[work] = score;
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 411)) return;
    for (int work = blockIdx.x; work < args.mix.rows * QSAHeads; work += Blocks) {
        int row = work / QSAHeads;
        int head = work % QSAHeads;
        int count = min(args.sequence_lengths[row] / 4, 512) * 4 + args.sequence_lengths[row] % 4;
        float maximum = -INFINITY;
        float normalizer = 0.0f;
        float accumulator = 0.0f;
        for (int start = 0; start < count; start += 64) {
            if (threadIdx.x < 64) probabilities[threadIdx.x] = start + threadIdx.x < count ? args.attention_scores[work * QSABudget + start + threadIdx.x] : -INFINITY;
            __syncthreads();
            float next_maximum = maximum;
            for (int token = 0; token < 64; ++token) next_maximum = fmaxf(next_maximum, probabilities[token]);
            float alpha = exp2f(maximum - next_maximum);
            __syncthreads();
            if (threadIdx.x < 64) probabilities[threadIdx.x] = exp2f(probabilities[threadIdx.x] - next_maximum);
            __syncthreads();
            float total = 0.0f;
            for (int token = 0; token < 64; ++token) total += probabilities[token];
            normalizer = normalizer * alpha + total;
            accumulator *= alpha;
            for (int token = 0; token < min(64, count - start); ++token) {
                int logical = args.selected[row * QSABudget + start + token];
                int slot = args.request_tokens[args.requests[row] * args.request_stride + logical];
                float value = float(args.value_cache[qsa_cache_offset(slot, threadIdx.x, args.value_page_stride, args.value_token_stride, args.value_dim_stride)]);
                accumulator = fmaf(qsa_round(probabilities[token]), value, accumulator);
            }
            maximum = next_maximum;
            __syncthreads();
        }
        float attention = qsa_round(accumulator / normalizer);
        float gate = float(args.projected[row * 2048 + head * 512 + QSADim + threadIdx.x]);
        args.core[work * QSADim + threadIdx.x] = BFloat16(attention * sigmoid(gate));
        __syncthreads();
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 412)) return;
    gemv(args.core, args.output_weight, args.output, nullptr, args.mix.rows, 768, Hidden, 0);
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 413)) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(args.executions, 1);
        atomicAdd(args.mix.executions, 1);
    }
    return;
}

extern "C" int flash_qsa_info(int64_t* result) {
    int occupancy;
    hipError_t status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, qsa_attention, Threads, 0);
    if (status != hipSuccess) return status;
    result[0] = sizeof(QSAArguments);
    result[1] = occupancy;
    return hipSuccess;
}

extern "C" int flash_qsa(QSAArguments* args, void* stream) {
    if (args->mix.rows < 1 || args->mix.rows > MaxRows || (args->position_axes != 1 && args->position_axes != 3) || args->request_stride < QSAContext) return hipErrorInvalidValue;
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(qsa_attention), arguments, stream);
}
