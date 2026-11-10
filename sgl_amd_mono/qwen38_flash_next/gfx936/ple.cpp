#include "kernel.cpp"

struct PLEArguments {
    const BFloat16* input;
    const int64_t* contexts;
    const int64_t* multipliers;
    const int64_t* vocab_sizes;
    const int64_t* offsets;
    const BFloat16* embedding_weight;
    const BFloat16* key_weight;
    const BFloat16* value_weight;
    const BFloat16* key_norm_weight;
    const BFloat16* query_norm_weight;
    const BFloat16* conv_norm_weight;
    const BFloat16* conv_weight;
    const int64_t* state_indices;
    BFloat16* conv_state;
    BFloat16* local_embedding;
    BFloat16* embedding;
    BFloat16* key;
    BFloat16* value;
    BFloat16* normalized_key;
    BFloat16* normalized_query;
    BFloat16* gated;
    BFloat16* normalized_gated;
    BFloat16* output;
    int* executions;
    SyncArguments sync;
    int64_t rows;
    int64_t vocab_start;
    int64_t vocab_end;
    int64_t eos_token;
    int64_t state_slots;
};

__global__ __launch_bounds__(Threads, 1) void ple_decode(PLEArguments args) {
    if (args.sync.state[2]) return;
    if (blockIdx.x == 0 && threadIdx.x < args.rows) {
        int row = threadIdx.x;
        int64_t slot = args.state_indices[row];
        if (slot < 0 || slot >= args.state_slots || (slot != 0 && row == 0 && args.rows == 2 && slot == args.state_indices[1])) atomicCAS(args.sync.state + 2, 0u, 500u);
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 500)) return;
    for (int index = blockIdx.x * Threads + threadIdx.x; index < args.rows * Hidden; index += Blocks * Threads) {
        int row = index / Hidden;
        int head = index % Hidden / 160;
        int column = index % 160;
        int64_t token0 = args.contexts[row * 3];
        int64_t token1 = args.contexts[row * 3 + 1];
        int64_t token2 = args.contexts[row * 3 + 2];
        int64_t previous2 = token0 == args.eos_token || token1 == args.eos_token ? args.eos_token : token0;
        uint64_t mixed = uint64_t(token2) * uint64_t(args.multipliers[0]) ^ uint64_t(token1) * uint64_t(args.multipliers[1]);
        if (head >= 8) mixed ^= uint64_t(previous2) * uint64_t(args.multipliers[2]);
        int64_t vocabulary = args.vocab_sizes[head];
        int64_t hashed = int64_t(mixed) % vocabulary;
        if (hashed < 0) hashed += vocabulary;
        int64_t lookup = hashed + args.offsets[head];
        args.local_embedding[index] = lookup >= args.vocab_start && lookup < args.vocab_end ? args.embedding_weight[(lookup - args.vocab_start) * 160 + column] : BFloat16(0.0f);
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 501)) return;
    if (!all_reduce(args.local_embedding, args.embedding, args.rows * Hidden, args.sync, 510)) return;
    gemv(args.embedding, args.key_weight, args.key, nullptr, args.rows, Hidden, Hyper, 0);
    gemv(args.embedding, args.value_weight, args.value, nullptr, args.rows, Hidden, Hidden, 0);
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 520)) return;
    normalize(args.key, args.key_norm_weight, args.normalized_key, args.rows);
    normalize(args.input, args.query_norm_weight, args.normalized_query, args.rows);
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 521)) return;
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    int lane = threadIdx.x & 63;
    for (int work = wave; work < args.rows * Branches; work += Blocks * 4) {
        int row = work / Branches;
        float partial = 0.0f;
        for (int column = lane; column < Hidden; column += 64) {
            int offset = work * Hidden + column;
            partial += float(BFloat16(float(args.normalized_key[offset]) * float(args.normalized_query[offset])));
        }
        float gate = float(BFloat16(float(BFloat16(wave_sum(partial))) / 50.59644256269407f));
        float root = float(BFloat16(sqrtf(fmaxf(fabsf(gate), 1.0e-6f))));
        float transformed = float(BFloat16(gate > 0.0f ? root : (gate < 0.0f ? -root : 0.0f)));
        float activated = float(BFloat16(sigmoid(transformed)));
        for (int column = lane; column < Hidden; column += 64) args.gated[work * Hidden + column] = BFloat16(activated * float(args.value[row * Hidden + column]));
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 522)) return;
    normalize(args.gated, args.conv_norm_weight, args.normalized_gated, args.rows);
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 523)) return;
    for (int index = blockIdx.x * Threads + threadIdx.x; index < args.rows * Hyper; index += Blocks * Threads) {
        int row = index / Hyper;
        int channel = index % Hyper;
        int64_t slot = args.state_indices[row];
        if (slot == 0) {
            args.output[index] = args.input[index];
            continue;
        }
        BFloat16* history = args.conv_state + (slot * Hyper + channel) * 9;
        float current = float(args.normalized_gated[index]);
        float convolved = float(history[0]) * float(args.conv_weight[channel * 4]);
        convolved = fmaf(float(history[3]), float(args.conv_weight[channel * 4 + 1]), convolved);
        convolved = fmaf(float(history[6]), float(args.conv_weight[channel * 4 + 2]), convolved);
        convolved = fmaf(current, float(args.conv_weight[channel * 4 + 3]), convolved);
        for (int column = 0; column < 8; ++column) history[column] = history[column + 1];
        history[8] = BFloat16(current);
        float rounded = float(BFloat16(convolved));
        float activated = float(BFloat16(rounded * sigmoid(rounded)));
        float ple = float(BFloat16(float(args.gated[index]) + activated));
        args.output[index] = BFloat16(float(args.input[index]) + ple);
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 524)) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(args.executions, 1);
    return;
}

extern "C" int flash_ple_info(int64_t* result) {
    int occupancy;
    hipError_t status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, ple_decode, Threads, 0);
    if (status != hipSuccess) return status;
    result[0] = sizeof(PLEArguments);
    result[1] = occupancy;
    return hipSuccess;
}

extern "C" int flash_ple(PLEArguments* args, void* stream) {
    if (args->rows < 1 || args->rows > MaxRows || args->vocab_end <= args->vocab_start || args->state_slots <= 1) return hipErrorInvalidValue;
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(ple_decode), arguments, stream);
}
