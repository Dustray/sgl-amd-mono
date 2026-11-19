#include "kernel.cpp"

constexpr int QSADim = 256;
constexpr int QSAHeads = 3;
constexpr int IndexDim = 128;
constexpr int IndexHeads = 4;
// The index top-k ran as one 1024-wide bitonic sort, which pinned the kernel to 4096 context.
// Long contexts sort chunks of QSABlocks groups in one block each and merge the per-chunk
// best-of-QSATopK lists pairwise (see qsa_attention).
constexpr int QSABlocks = 1024;       // 4-token groups per chunk (context covered per sort block)
constexpr int QSAChunkLists = 64;     // max chunks per row (power of two) -> context cap 262144
constexpr int QSATopK = 512;          // blocks kept by the index top-k
constexpr int QSABudget = 2051;       // QSATopK * 4 + 3 partial tokens
static_assert(QSABlocks == 2 * QSATopK, "pairwise merge packs two top-k lists into one sort block");

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
    float* chunk_scores;
    int32_t* chunk_indices;
    float* merge_scores;
    int32_t* merge_indices;
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
    int64_t qsa_lists;
    int64_t context_limit;
    // Fused MTP verify: request rows repeat (DRAFT_TOKENS rows per request), so the
    // decode-side "no duplicate request rows" check must not fire; locations still must
    // be unique.
    int64_t verify_requests;
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
    stage_mark(703);
    __shared__ float reduction[4];
    __shared__ float normalized[QSADim];
    __shared__ float sort_scores[QSABlocks];
    __shared__ int sort_indices[QSABlocks];
    // Row checks run on block `row`; the slot table scan (length entries, up to the context cap)
    // is spread over the whole grid so a long context does not serialize on one CU.
    for (int row = 0; row < args.mix.rows; ++row) {
        int length = args.sequence_lengths[row];
        int64_t request = args.requests[row];
        int64_t location = args.locations[row];
        bool valid = length > 0 && length <= args.context_limit && request >= 0 && request < args.request_count;
        valid = valid && location >= 0 && location < args.cache_slots && request * 4 + 3 < args.ring_slots;
        if (blockIdx.x == row) {
            bool owned = valid;
            for (int axis = 0; axis < args.position_axes; ++axis) {
                int64_t position = args.positions[axis * args.position_axis_stride + row];
                owned = owned && position >= 0 && position < args.rotary_rows;
            }
            owned = owned && args.request_tokens[request * args.request_stride + length - 1] == location;
            bool reserved = request == 0 && location == 0;
            if (!reserved && args.verify_requests == 0 && args.mix.rows == 2 && row == 0 && (args.requests[0] == args.requests[1] || args.locations[0] == args.locations[1])) owned = false;
            if (!reserved && args.verify_requests > 0) {
                for (int other = 0; other < row; ++other) owned = owned && args.locations[row] != args.locations[other];
            }
            if (!owned) atomicCAS(args.mix.synchronization + 2, 0u, 400u);
        }
        if (valid) {
            for (int token = blockIdx.x * Threads + threadIdx.x; token < length; token += Blocks * Threads) {
                int slot = args.request_tokens[request * args.request_stride + token];
                if (slot < 0 || slot >= args.cache_slots) atomicCAS(args.mix.synchronization + 2, 0u, 400u);
            }
        }
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 400)) return;
    if (!run_mix(args.mix, 401)) return;
    // qkv (2560 -> 2048, 128 tiles per row) and index (2560 -> 640, 40 tiles per row) share one task
    // space with pipelined loads; two plain gemv calls both started at wave 0, so waves 0-39 ran
    // two tiles back to back (~148 us, 2026-11-16).
    {
        constexpr int QKVTiles = 2048 / 16;
        constexpr int IndexTiles = 640 / 16;
        int rows = args.mix.rows;
        int wave = blockIdx.x * 4 + threadIdx.x / 64;
        int lane = threadIdx.x & 63;
        for (int task = wave; task < rows * (QKVTiles + IndexTiles); task += Blocks * 4) {
            if (task < rows * QKVTiles) {
                int row = task / QKVTiles;
                int base = task % QKVTiles * 16;
                float value = matvec16_pipelined(args.mix.output + row * Hidden, args.qkv_weight + int64_t(base) * Hidden);
                if (lane < 16) args.projected[row * 2048 + base + lane] = BFloat16(value);
            } else {
                int index = task - rows * QKVTiles;
                int row = index / IndexTiles;
                int base = index % IndexTiles * 16;
                float value = matvec16_pipelined(args.mix.output + row * Hidden, args.index_weight + int64_t(base) * Hidden);
                if (lane < 16) args.index_projected[row * 640 + base + lane] = BFloat16(value);
            }
        }
    }
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
    // Chunks of QSABlocks groups this launch actually needs: the longest row's groups, rounded up to
    // a power of two. The buffers are sized for qsa_lists (the server context), but a short request
    // on a long-context server must not pay for the sorts and merge barriers of empty chunks, and
    // with one chunk the path below is the former single 1024-wide sort, bit for bit.
    int lists = 1;
    {
        int longest = 0;
        for (int row = 0; row < args.mix.rows; ++row) longest = max(longest, args.sequence_lengths[row] / 4);
        while (lists < args.qsa_lists && lists * QSABlocks < longest) lists *= 2;
    }
    int rounds = 0;
    while ((1 << rounds) < lists) ++rounds;
    // Index scores: one wave per 4-token group, -INF past the row's length up to lists * QSABlocks
    // (entries beyond that are left stale; captures and checks only read the row's groups).
    int64_t score_stride = args.qsa_lists * QSABlocks;
    int64_t score_width = int64_t(lists) * QSABlocks;
    for (int64_t work = wave; work < args.mix.rows * score_width; work += Blocks * 4) {
        int row = work / score_width;
        int64_t group = work % score_width;
        float score = -INFINITY;
        if (group < args.sequence_lengths[row] / 4) {
            int slot = args.request_tokens[args.requests[row] * args.request_stride + group * 4] / 4;
            score = 0.0f;
            for (int head = 0; head < IndexHeads; ++head) {
                float partial = 0.0f;
                for (int column = lane; column < IndexDim; column += 64) partial += float(args.index_queries[(row * IndexHeads + head) * IndexDim + column]) * float(args.compressed_keys[int64_t(slot) * IndexDim + column]);
                score += fmaxf(wave_sum(partial), 0.0f);
            }
        }
        if (lane == 0) args.index_scores[row * score_stride + group] = score;
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 409)) return;
    // Index top-k. Each block sorts one chunk of QSABlocks groups (score descending, index
    // ascending: a total order). With one chunk (every row <= 4096 tokens) the sorting block writes
    // `selected` itself, exactly as the former single 1024-wide sort, so short rows keep their
    // barriers and results bit for bit. With more chunks each chunk's best QSATopK go to
    // chunk_* and log2(lists) pairwise bitonic merges reduce them to one list; because the
    // order is total, the merged top QSATopK equal one full sort of every group.
    for (int64_t work = blockIdx.x; work < args.mix.rows * lists; work += Blocks) {
        int row = work / lists;
        int chunk = work % lists;
        int length = args.sequence_lengths[row];
        int groups = length / 4;
        // <= QSATopK groups are selected in order without sorting.
        if (groups <= QSATopK) {
            if (lists == 1) {
                for (int column = threadIdx.x; column < QSABudget; column += Threads)
                    args.selected[row * QSABudget + column] = column < length ? column : -1;
            }
            continue;
        }
        int remaining = groups - chunk * QSABlocks;
        int64_t list = (int64_t(row) * args.qsa_lists + chunk) * QSATopK;
        if (remaining <= 0) {
            // Chunk entirely past the length: filler list, ranks below every real group.
            for (int column = threadIdx.x; column < QSATopK; column += Threads) {
                args.chunk_scores[list + column] = -INFINITY;
                args.chunk_indices[list + column] = 0x7fffffff;
            }
            continue;
        }
        for (int column = threadIdx.x; column < QSABlocks; column += Threads) {
            sort_scores[column] = column < remaining ? args.index_scores[row * score_stride + int64_t(chunk) * QSABlocks + column] : -INFINITY;
            sort_indices[column] = column < remaining ? chunk * QSABlocks + column : 0x7fffffff;
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
        if (lists == 1) {
            int kept = min(groups, QSATopK);
            for (int column = threadIdx.x; column < QSABudget; column += Threads) {
                int token = -1;
                if (column < kept * 4) token = sort_indices[column / 4] * 4 + column % 4;
                else if (column < kept * 4 + length % 4) token = groups * 4 + column - kept * 4;
                args.selected[row * QSABudget + column] = token;
            }
        } else {
            for (int column = threadIdx.x; column < QSATopK; column += Threads) {
                args.chunk_scores[list + column] = sort_scores[column];
                args.chunk_indices[list + column] = sort_indices[column];
            }
        }
        __syncthreads();
    }
    if (lists > 1) {
        if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 416)) return;
        for (int round = 0; round < rounds; ++round) {
            const float* src_scores = round % 2 ? args.merge_scores : args.chunk_scores;
            const int32_t* src_indices = round % 2 ? args.merge_indices : args.chunk_indices;
            float* dst_scores = round % 2 ? args.chunk_scores : args.merge_scores;
            int32_t* dst_indices = round % 2 ? args.chunk_indices : args.merge_indices;
            int outputs = lists >> (round + 1);   // lists per row after this round
            for (int64_t item = blockIdx.x; item < args.mix.rows * outputs; item += Blocks) {
                int row = item / outputs;
                int pair = item % outputs;
                if (args.sequence_lengths[row] / 4 <= QSATopK) continue;
                // List p of round r holds chunks [p * 2^r, (p + 1) * 2^r); in round r the source
                // lists sit at stride lists >> r per row, compacted to the front.
                int64_t row_base = int64_t(row) * args.qsa_lists * QSATopK;
                int64_t left = row_base + int64_t(pair * 2) * QSATopK;
                int64_t right = left + QSATopK;
                int64_t target = row_base + int64_t(pair) * QSATopK;
                bool right_empty = int64_t(pair * 2 + 1) * (1 << round) * QSABlocks >= args.sequence_lengths[row] / 4;
                if (right_empty) {
                    // Right half is all fillers, which rank below every entry on the left.
                    for (int column = threadIdx.x; column < QSATopK; column += Threads) {
                        dst_scores[target + column] = src_scores[left + column];
                        dst_indices[target + column] = src_indices[left + column];
                    }
                    __syncthreads();
                    continue;
                }
                // Left descending + right reversed is bitonic; one merge stage sorts it descending.
                for (int column = threadIdx.x; column < QSATopK; column += Threads) {
                    sort_scores[column] = src_scores[left + column];
                    sort_indices[column] = src_indices[left + column];
                    sort_scores[QSABlocks - 1 - column] = src_scores[right + column];
                    sort_indices[QSABlocks - 1 - column] = src_indices[right + column];
                }
                __syncthreads();
                for (int distance = QSABlocks / 2; distance > 0; distance /= 2) {
                    for (int column = threadIdx.x; column < QSABlocks; column += Threads) {
                        int partner = column ^ distance;
                        if (partner > column) {
                            bool greater = sort_scores[partner] > sort_scores[column] || (sort_scores[partner] == sort_scores[column] && sort_indices[partner] < sort_indices[column]);
                            if (greater) {
                                float score = sort_scores[column]; sort_scores[column] = sort_scores[partner]; sort_scores[partner] = score;
                                int index = sort_indices[column]; sort_indices[column] = sort_indices[partner]; sort_indices[partner] = index;
                            }
                        }
                    }
                    __syncthreads();
                }
                for (int column = threadIdx.x; column < QSATopK; column += Threads) {
                    dst_scores[target + column] = sort_scores[column];
                    dst_indices[target + column] = sort_indices[column];
                }
                __syncthreads();
            }
            if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 417)) return;
        }
        const int32_t* merged = rounds % 2 ? args.merge_indices : args.chunk_indices;
        for (int row = blockIdx.x; row < args.mix.rows; row += Blocks) {
            int length = args.sequence_lengths[row];
            int groups = length / 4;
            int kept = min(groups, QSATopK);
            const int32_t* final_indices = merged + int64_t(row) * args.qsa_lists * QSATopK;
            for (int column = threadIdx.x; column < QSABudget; column += Threads) {
                int token = -1;
                if (column < kept * 4) token = groups <= QSATopK ? column : final_indices[column / 4] * 4 + column % 4;
                else if (column < kept * 4 + length % 4) token = groups * 4 + column - kept * 4;
                args.selected[row * QSABudget + column] = token;
            }
        }
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
    // One wave per (row, head, 64-column chunk), spread one task per CU first (12 waves at b1).
    // Each lane keeps the original per-column chain: max, normalizer and accumulator are updated
    // token by token in the same order as the former one-block-per-head loop, so results are
    // bit-identical. Per 64-token chunk every lane resolves one slot and all 64 value loads are
    // issued before the fmaf chain; the old loop did three dependent global loads per token
    // (~330 us on 3 CUs at b1, 2026-11-16).
    for (int task = (threadIdx.x / 64) * Blocks + blockIdx.x; task < args.mix.rows * QSAHeads * 4; task += Blocks * 4) {
        int work = task / 4;
        int column = task % 4 * 64 + lane;
        int row = work / QSAHeads;
        int head = work % QSAHeads;
        int count = min(args.sequence_lengths[row] / 4, 512) * 4 + args.sequence_lengths[row] % 4;
        const int32_t* tokens = args.request_tokens + args.requests[row] * args.request_stride;
        float maximum = -INFINITY;
        float normalizer = 0.0f;
        float accumulator = 0.0f;
        for (int start = 0; start < count; start += 64) {
            int valid = min(64, count - start);
            float score = lane < valid ? args.attention_scores[work * QSABudget + start + lane] : -INFINITY;
            int slot = lane < valid ? tokens[args.selected[row * QSABudget + start + lane]] : 0;
            float next_maximum = maximum;
#pragma unroll
            for (int token = 0; token < 64; ++token) next_maximum = fmaxf(next_maximum, __shfl(score, token, 64));
            float alpha = exp2f(maximum - next_maximum);
            float probability = exp2f(score - next_maximum);
            float total = 0.0f;
#pragma unroll
            for (int token = 0; token < 64; ++token) total += __shfl(probability, token, 64);
            normalizer = normalizer * alpha + total;
            accumulator *= alpha;
            // 16 value loads in flight per batch; 64 at once needed 256 VGPRs and spilled.
            for (int base = 0; base < valid; base += 16) {
                float values[16];
#pragma unroll
                for (int token = 0; token < 16; ++token) {
                    int token_slot = __shfl(slot, base + token, 64);
                    values[token] = base + token < valid ? float(args.value_cache[qsa_cache_offset(token_slot, column, args.value_page_stride, args.value_token_stride, args.value_dim_stride)]) : 0.0f;
                }
#pragma unroll
                for (int token = 0; token < 16; ++token) {
                    float token_probability = qsa_round(__shfl(probability, base + token, 64));
                    if (base + token < valid) accumulator = fmaf(token_probability, values[token], accumulator);
                }
            }
            maximum = next_maximum;
        }
        float attention = qsa_round(accumulator / normalizer);
        float gate = float(args.projected[row * 2048 + head * 512 + QSADim + column]);
        args.core[work * QSADim + column] = BFloat16(attention * sigmoid(gate));
    }
    if (!grid_sync(args.mix.synchronization, args.mix.timeout_ticks, 412)) return;
    pipelined_gemv<768>(args.core, args.output_weight, args.output, args.mix.rows, Hidden);
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
    if (args->mix.rows < 1 || args->mix.rows > MaxRows || (args->position_axes != 1 && args->position_axes != 3)) return hipErrorInvalidValue;
    int64_t lists = args->qsa_lists;
    if (lists < 1 || lists > QSAChunkLists || (lists & (lists - 1)) != 0) return hipErrorInvalidValue;
    if (args->context_limit < 1 || args->context_limit > lists * QSABlocks * 4 || args->request_stride < args->context_limit) return hipErrorInvalidValue;
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(qsa_attention), arguments, stream);
}
