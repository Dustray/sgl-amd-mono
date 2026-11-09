#include "kernel.cpp"
#include <hip/hip_fp16.h>

constexpr int AttentionHeads = 24;
constexpr int KVHeads = 4;
constexpr int AttentionDim = 256;
constexpr int AttentionProjection = 14336;
constexpr int MaximumContext = 4096;
constexpr int AttentionPartition = 512;

__device__ __forceinline__ float attention_rounded(float value) {
    return __uint_as_float((__float_as_uint(value) + 0x8000u) & 0xffff0000u);
}

__device__ __forceinline__ float attention_score(float value) {
    float scaled;
    asm volatile("v_mul_f32 %0, 0x3c38aa3b, %1" : "=v"(scaled) : "v"(value));
    return __half2float(__float2half_rn(scaled)) * 8.0f;
}

struct AttentionArguments {
    const BFloat16* hidden;
    const BFloat16* residual;
    const BFloat16* norm_weight;
    const BFloat16* qkv_weight;
    const BFloat16* q_norm_weight;
    const BFloat16* k_norm_weight;
    const BFloat16* rotary_cache;
    const int* rotary_axes;
    const void* positions;
    void* key_cache;
    void* value_cache;
    const void* cache_locations;
    const int* page_table;
    const int* cache_lengths;
    const void* request_indices;
    BFloat16* normalized;
    BFloat16* projected;
    BFloat16* query;
    BFloat16* key;
    BFloat16* value;
    BFloat16* gate;
    float* scores;
    BFloat16* output;
    BFloat16* residual_output;
    int* executions;
    unsigned int* synchronization;
    int64_t rows;
    int64_t rotary_dim;
    int64_t rotary_cache_rows;
    int64_t position_axes;
    int64_t position_axis_stride;
    int64_t position_row_stride;
    int64_t key_page_stride;
    int64_t key_token_stride;
    int64_t key_head_stride;
    int64_t key_dim_stride;
    int64_t value_page_stride;
    int64_t value_token_stride;
    int64_t value_head_stride;
    int64_t value_dim_stride;
    int64_t page_table_stride;
    int64_t page_size;
    int64_t page_count;
    int64_t cache_dtype;
    int64_t location_bytes;
    int64_t position_bytes;
    int64_t request_index_bytes;
    int64_t context_limit;
    int64_t barrier_timeout_ticks;
};

__device__ __forceinline__ int64_t attention_index(const void* pointer, int64_t index, int64_t bytes) {
    return bytes == 8 ? static_cast<const int64_t*>(pointer)[index] : static_cast<const int*>(pointer)[index];
}

__device__ __forceinline__ int attention_length(const AttentionArguments& args, int row) {
    if (args.request_indices != nullptr && attention_index(args.request_indices, row, args.request_index_bytes) == 0) return 0;
    return args.cache_lengths[row];
}

__device__ __forceinline__ float attention_load(const void* pointer, int64_t index, int64_t dtype) {
    if (dtype == 0) return static_cast<float>(static_cast<const BFloat16*>(pointer)[index]);
    return __half2float(__ushort_as_half(static_cast<unsigned short>(static_cast<const unsigned char*>(pointer)[index]) << 8));
}

__device__ __forceinline__ void attention_store(void* pointer, int64_t index, float value, int64_t dtype) {
    if (dtype == 0) {
        static_cast<BFloat16*>(pointer)[index] = BFloat16(value);
    } else {
        unsigned short bits = __half_as_ushort(__float2half_rn(value));
        unsigned short magnitude = bits & 0x7fffu;
        unsigned int rounded_bits = magnitude >= 0x7c00u ? magnitude : magnitude + 0x7fu + ((magnitude >> 8) & 1u);
        static_cast<unsigned char*>(pointer)[index] = static_cast<unsigned char>((bits >> 8 & 0x80u) | (rounded_bits >> 8));
    }
}

__device__ float attention_block_max(float value, float* shared) {
    for (int offset = 32; offset > 0; offset /= 2) value = fmaxf(value, __shfl_xor(value, offset, 64));
    if ((threadIdx.x & 63) == 0) shared[threadIdx.x / 64] = value;
    __syncthreads();
    value = threadIdx.x < 4 ? shared[threadIdx.x] : -INFINITY;
    if (threadIdx.x < 64) {
        for (int offset = 2; offset > 0; offset /= 2) value = fmaxf(value, __shfl_xor(value, offset, 64));
        if (threadIdx.x == 0) shared[4] = value;
    }
    __syncthreads();
    return shared[4];
}

__device__ float attention_head_variance(float input, float* shared) {
    float total = input * input;
    for (int offset = 8; offset > 0; offset /= 2) total += __shfl_xor(total, offset, 64);
    total += __shfl_xor(total, 16, 64);
    total += __shfl_xor(total, 32, 64);
    if ((threadIdx.x & 63) == 0) shared[threadIdx.x / 64] = total;
    __syncthreads();
    total = threadIdx.x < 4 ? shared[threadIdx.x] : 0.0f;
    if (threadIdx.x < 64) {
        total += __shfl_xor(total, 2, 64);
        total += __shfl_xor(total, 1, 64);
    }
    if (threadIdx.x == 0) shared[4] = total;
    __syncthreads();
    return shared[4] / AttentionDim;
}

__device__ float attention_projection(const BFloat16* input, const BFloat16* weight, int rows, int output_base) {
    if (rows == 2) return matrix_tile(input, weight, Hidden, rows, output_base, MatvecMode::Split);
    int lane = threadIdx.x & 63;
    int weight_row = lane % 16;
    int rotation = ((output_base / 128) & 7) * 512;
    float partials[4];
    for (int group = 0; group < 4; ++group) {
        Float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int start = group * 32; start < Hidden; start += 128) {
            for (int half = 0; half < 2; ++half) {
                int column = (start + rotation) % Hidden + half * 16 + (lane / 16) * 4;
                Short4 left = *reinterpret_cast<const Short4*>(weight + int64_t(weight_row) * Hidden + column);
                Short4 right = {0, 0, 0, 0};
                if (lane % 16 < rows) right = *reinterpret_cast<const Short4*>(input + (lane % 16) * Hidden + column);
                accumulator = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulator);
            }
        }
        partials[group] = accumulator[0];
    }
    float result = partials[0];
    for (int group = 1; group < 4; ++group) result += partials[group];
    return result;
}

__global__ __launch_bounds__(Threads, 1) void dense_attention_k1(AttentionArguments args) {
    __shared__ float reduction[17];
    __shared__ float normalized_head[AttentionDim];
    __shared__ float probabilities[AttentionPartition];
    __shared__ float partition_weights[MaximumContext / AttentionPartition];
    if (args.synchronization[2]) return;
    if (blockIdx.x < args.rows) {
        int row = blockIdx.x;
        int length = attention_length(args, row);
        int64_t location = attention_index(args.cache_locations, row, args.location_bytes);
        bool valid = length >= 0 && length <= args.context_limit;
        if (args.request_indices != nullptr) {
            int64_t request = attention_index(args.request_indices, row, args.request_index_bytes);
            valid = valid && request >= 0;
            if (request == 0) valid = valid && location == 0 && args.cache_lengths[row] >= 0 && args.cache_lengths[row] <= 1;
        }
        if (length > 0 && valid) {
            valid = location > 0 && location < args.page_count * args.page_size;
            int logical_page = (length - 1) / args.page_size;
            valid = valid && location == int64_t(args.page_table[row * args.page_table_stride + logical_page]) * args.page_size + (length - 1) % args.page_size;
            for (int page = threadIdx.x; page < (length + args.page_size - 1) / args.page_size; page += Threads) {
                int physical_page = args.page_table[row * args.page_table_stride + page];
                if (physical_page < 0 || physical_page >= args.page_count) valid = false;
            }
        }
        for (int axis = 0; axis < args.position_axes; ++axis) {
            int64_t position = attention_index(args.positions, axis * args.position_axis_stride + row * args.position_row_stride, args.position_bytes);
            if (position < 0 || position >= args.rotary_cache_rows) valid = false;
        }
        if (!valid) atomicCAS(args.synchronization + 2, 0u, 101u);
        if (row == 0 && threadIdx.x == 0 && args.rows == 2 && length > 0 && attention_length(args, 1) > 0 && location == attention_index(args.cache_locations, 1, args.location_bytes))
            atomicCAS(args.synchronization + 2, 0u, 102u);
        normalize(args.hidden, args.residual, args.norm_weight, args.normalized, args.residual_output, row, reduction);
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 1) || args.synchronization[2]) return;
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    int lane = threadIdx.x & 63;
    for (int work = wave; work < AttentionProjection / 16; work += Blocks * 4) {
        int output_base = work * 16;
        float projected = attention_projection(args.normalized, args.qkv_weight + int64_t(output_base) * Hidden, args.rows, output_base);
        int row = lane / 16;
        if (row < args.rows) args.projected[row * AttentionProjection + output_base + lane % 16] = BFloat16(projected);
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 2)) return;
    for (int work = blockIdx.x; work < args.rows * (AttentionHeads + KVHeads); work += Blocks) {
        int row = work / (AttentionHeads + KVHeads);
        int head = work % (AttentionHeads + KVHeads);
        int column = threadIdx.x;
        bool is_query = head < AttentionHeads;
        int local_head = is_query ? head : head - AttentionHeads;
        int projection_base = is_query ? local_head * 2 * AttentionDim : Core * 2 + local_head * AttentionDim;
        float input = static_cast<float>(args.projected[row * AttentionProjection + projection_base + column]);
        float variance = attention_head_variance(input, reduction);
        const BFloat16* weight = is_query ? args.q_norm_weight : args.k_norm_weight;
        normalized_head[column] = rounded(input * __builtin_amdgcn_rsqf(variance + 1.0e-6f) * (1.0f + static_cast<float>(weight[column])));
        __syncthreads();
        float rotated = normalized_head[column];
        if (column < args.rotary_dim) {
            int half_dim = args.rotary_dim / 2;
            int pair = column % half_dim;
            int axis = args.position_axes == 1 ? 0 : args.rotary_axes[pair];
            int64_t position = attention_index(args.positions, axis * args.position_axis_stride + row * args.position_row_stride, args.position_bytes);
            float cosine = static_cast<float>(args.rotary_cache[position * args.rotary_dim + pair]);
            float sine = static_cast<float>(args.rotary_cache[position * args.rotary_dim + half_dim + pair]);
            float other = normalized_head[column < half_dim ? column + half_dim : column - half_dim];
            rotated = rounded(rotated * cosine) + (column < half_dim ? -rounded(other * sine) : rounded(other * sine));
        }
        if (is_query) {
            args.query[row * Core + local_head * AttentionDim + column] = BFloat16(rotated);
            args.gate[row * Core + local_head * AttentionDim + column] = args.projected[row * AttentionProjection + projection_base + AttentionDim + column];
        } else {
            int key_index = row * KVHeads * AttentionDim + local_head * AttentionDim + column;
            BFloat16 key = BFloat16(rotated);
            BFloat16 value = args.projected[row * AttentionProjection + Core * 2 + KVHeads * AttentionDim + local_head * AttentionDim + column];
            args.key[key_index] = key;
            args.value[key_index] = value;
            if (attention_length(args, row) > 0) {
                int64_t location = attention_index(args.cache_locations, row, args.location_bytes);
                int64_t page = location / args.page_size;
                int64_t token = location % args.page_size;
                attention_store(args.key_cache, page * args.key_page_stride + token * args.key_token_stride + local_head * args.key_head_stride + column * args.key_dim_stride, static_cast<float>(key), args.cache_dtype);
                attention_store(args.value_cache, page * args.value_page_stride + token * args.value_token_stride + local_head * args.value_head_stride + column * args.value_dim_stride, static_cast<float>(value), args.cache_dtype);
            }
        }
        __syncthreads();
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 3)) return;
    int token_blocks = (args.context_limit + 63) / 64;
    for (int work = blockIdx.x; work < args.rows * AttentionHeads * token_blocks; work += Blocks) {
        int token_block = work % token_blocks;
        int head_index = work / token_blocks;
        int row = head_index / AttentionHeads;
        int query_head = head_index % AttentionHeads;
        int key_head = query_head / (AttentionHeads / KVHeads);
        int length = attention_length(args, row);
        int token = token_block * 64 + (threadIdx.x / 64) * 16 + lane % 16;
        Float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int fragment = 0; fragment < AttentionDim / 16; ++fragment) {
            Short4 left = {0, 0, 0, 0}, right = {0, 0, 0, 0};
            int column = (fragment / 2) * 32 + (lane / 16) * 8 + (fragment % 2) * 4;
            if (token < length) {
                int page = args.page_table[row * args.page_table_stride + token / args.page_size];
                int offset = token % args.page_size;
                for (int element = 0; element < 4; ++element) {
                    float key = attention_load(args.key_cache, page * args.key_page_stride + offset * args.key_token_stride + key_head * args.key_head_stride + (column + element) * args.key_dim_stride, args.cache_dtype);
                    left[element] = static_cast<short>(__float_as_uint(key) >> 16);
                }
            }
            if (lane % 16 == query_head % 6) right = *reinterpret_cast<const Short4*>(args.query + row * Core + query_head * AttentionDim + column);
            accumulator = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulator);
        }
        if (lane / 16 == (query_head % 6) % 4 && token < length) args.scores[head_index * MaximumContext + token] = attention_score(accumulator[(query_head % 6) / 4]);
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 4)) return;
    for (int work = blockIdx.x; work < args.rows * AttentionHeads; work += Blocks) {
        int row = work / AttentionHeads;
        int head = work % AttentionHeads;
        int length = attention_length(args, row);
        float maximum = -INFINITY;
        for (int token = threadIdx.x; token < length; token += Threads) maximum = fmaxf(maximum, args.scores[work * MaximumContext + token]);
        maximum = attention_block_max(maximum, reduction);
        float partials[MaximumContext / AttentionPartition];
        int partitions = (length + AttentionPartition - 1) / AttentionPartition;
        for (int partition = 0; partition < partitions; ++partition) {
            int begin = partition * AttentionPartition;
            int end = min(begin + AttentionPartition, length);
            float partition_maximum = -INFINITY;
            for (int token = begin + threadIdx.x; token < end; token += Threads) partition_maximum = fmaxf(partition_maximum, args.scores[work * MaximumContext + token]);
            partition_maximum = attention_block_max(partition_maximum, reduction);
            if (threadIdx.x < 64) {
                float denominator = 0.0f;
                for (int element = 0; element < 8; ++element) {
                    int token = begin + threadIdx.x * 8 + element;
                    float probability = token < end ? __builtin_amdgcn_exp2f(args.scores[work * MaximumContext + token] - partition_maximum) : 0.0f;
                    if (token < end) probabilities[token - begin] = probability;
                    denominator += probability;
                }
                for (int offset = 32; offset > 0; offset /= 2) denominator += __shfl_xor(denominator, offset, 64);
                if (threadIdx.x == 0) reduction[5] = denominator;
            }
            __syncthreads();
            float denominator = reduction[5];
            if (threadIdx.x == 0) partition_weights[partition] = denominator * __builtin_amdgcn_exp2f(partition_maximum - maximum);
            float inverse_denominator = __builtin_amdgcn_rcpf(denominator + 1.0e-6f);
            for (int token = threadIdx.x; token < end - begin; token += Threads) probabilities[token] = attention_rounded(probabilities[token] * inverse_denominator);
            __syncthreads();
            for (int tile = 0; tile < 4; ++tile) {
                int column = ((threadIdx.x / 64) + tile * 4) * 16 + lane % 16;
                Float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int fragment = 0; fragment < ((end - begin + 63) / 64) * 4; ++fragment) {
                    Short4 left = {0, 0, 0, 0}, right = {0, 0, 0, 0};
                    for (int element = 0; element < 4; ++element) {
                        int token = begin + (fragment / 4) * 64 + (lane / 16) * 16 + (fragment % 4) * 4 + element;
                        if (token < end) {
                            int page = args.page_table[row * args.page_table_stride + token / args.page_size];
                            int key_head = head / (AttentionHeads / KVHeads);
                            float value = attention_load(args.value_cache, page * args.value_page_stride + (token % args.page_size) * args.value_token_stride + key_head * args.value_head_stride + column * args.value_dim_stride, args.cache_dtype);
                            left[element] = static_cast<short>(__float_as_uint(value) >> 16);
                            if (lane % 16 == head % 6) right[element] = static_cast<short>(__float_as_uint(probabilities[token - begin]) >> 16);
                        }
                    }
                    accumulator = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulator);
                }
                if (lane / 16 == (head % 6) % 4) normalized_head[column] = attention_rounded(accumulator[(head % 6) / 4]);
            }
            __syncthreads();
            partials[partition] = normalized_head[threadIdx.x];
            __syncthreads();
        }
        float output = partitions == 1 ? partials[0] : 0.0f;
        if (partitions > 1) {
            float weight_sum = lane < partitions ? partition_weights[lane] : 0.0f;
            for (int offset = 32; offset > 0; offset /= 2) weight_sum += __shfl_xor(weight_sum, offset, 64);
            float inverse_weight_sum = __builtin_amdgcn_rcpf(__shfl(weight_sum, 0, 64) + 1.0e-6f);
            for (int partition = 0; partition < partitions; ++partition) {
                float weighted_partial = partials[partition] * partition_weights[partition];
                output = fmaf(weighted_partial, inverse_weight_sum, output);
            }
        }
        int index = row * Core + head * AttentionDim + threadIdx.x;
        float gate = static_cast<float>(args.gate[index]);
        float sigmoid = __builtin_amdgcn_rcpf(1.0f + native_exp(-gate));
        args.output[index] = BFloat16(attention_rounded(output) * sigmoid);
        __syncthreads();
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 5)) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(args.executions, 1);
}

extern "C" int mono_attention_info(int64_t* information) {
    int occupancy;
    hipError_t status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, dense_attention_k1, Threads, 0);
    information[0] = sizeof(AttentionArguments);
    information[1] = occupancy;
    return status;
}

extern "C" int mono_attention_k1(AttentionArguments* args, void* stream) {
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(dense_attention_k1), arguments, stream);
}

__global__ void probe_attention_score_rounding(const float* input, float* output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = attention_score(input[index]);
}

extern "C" int mono_attention_score_rounding(const float* input, float* output, int count, void* stream) {
    if (count < 1) return hipErrorInvalidValue;
    hipLaunchKernelGGL(probe_attention_score_rounding, dim3((count + 255) / 256), dim3(256), 0,
                       reinterpret_cast<hipStream_t>(stream), input, output, count);
    return hipGetLastError();
}
