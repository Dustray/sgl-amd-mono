#include "kernel.cpp"

constexpr int LocalVocab = 31040;
constexpr int Vocab = LocalVocab * Ranks;

struct EntryArguments {
    const void* input_ids;
    const BFloat16* weight;
    BFloat16* local;
    BFloat16* reduced;
    BFloat16* output;
    int* executions;
    SyncArguments sync;
    int64_t rows;
    int64_t index_bytes;
    int64_t vocab_start;
};

struct TailArguments {
    MixArguments mix;
    const BFloat16* weight;
    BFloat16* local;
    BFloat16* logits;
    int* executions;
    SyncArguments sync;
};

__global__ __launch_bounds__(Threads, 1) void decode_entry(EntryArguments args) {
    if (args.sync.state[2]) return;
    for (int index = blockIdx.x * Threads + threadIdx.x; index < args.rows * Hidden; index += Blocks * Threads) {
        int row = index / Hidden;
        int column = index % Hidden;
        int64_t token = args.index_bytes == 8 ? static_cast<const int64_t*>(args.input_ids)[row] : static_cast<const int32_t*>(args.input_ids)[row];
        if (token < 0 || token >= Vocab) atomicCAS(args.sync.state + 2, 0u, 600u);
        args.local[index] = token >= args.vocab_start && token < args.vocab_start + LocalVocab ? args.weight[(token - args.vocab_start) * Hidden + column] : BFloat16(0.0f);
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 600)) return;
    if (!all_reduce(args.local, args.reduced, args.rows * Hidden, args.sync, 610)) return;
    for (int index = blockIdx.x * Threads + threadIdx.x; index < args.rows * Hyper; index += Blocks * Threads) args.output[index] = args.reduced[index / Hyper * Hidden + index % Hidden];
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 620)) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(args.executions, 1);
    return;
}

__device__ __forceinline__ bool gather_logits(TailArguments args, int offset, int width) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int previous = atomicAdd(args.sync.state + 3, 1u);
        if (previous >= 0xffffffu) atomicCAS(args.sync.state + 2, 0u, 900u);
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 630)) return false;
    unsigned int epoch = atomicAdd(args.sync.state + 3, 0u);
    for (int index = blockIdx.x * Threads + threadIdx.x; index < args.mix.rows * width; index += Blocks * Threads) {
        unsigned int bits = reinterpret_cast<const unsigned short*>(args.local)[index / width * LocalVocab + offset + index % width];
        __scoped_atomic_store_n(&args.sync.peers[args.sync.rank]->values[index * 2], (epoch << 8) | (bits & 255u), __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        __scoped_atomic_store_n(&args.sync.peers[args.sync.rank]->values[index * 2 + 1], (epoch << 8) | (bits >> 8), __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
    __threadfence_system();
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 631)) return false;
    for (int index = blockIdx.x * Threads + threadIdx.x; index < args.mix.rows * Ranks * width; index += Blocks * Threads) {
        int row = index / (Ranks * width);
        int peer = index / width % Ranks;
        int column = index % width;
        int mailbox = (row * width + column) * 2;
        int64_t begin = wall_clock64();
        unsigned int low = __scoped_atomic_load_n(&args.sync.peers[peer]->values[mailbox], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
        unsigned int high = __scoped_atomic_load_n(&args.sync.peers[peer]->values[mailbox + 1], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
        while ((low >> 8) != epoch || (high >> 8) != epoch) {
            if ((low >> 8) > epoch || (high >> 8) > epoch || wall_clock64() - begin > args.sync.timeout_ticks) {
                atomicCAS(args.sync.state + 2, 0u, 632u);
                break;
            }
            __builtin_amdgcn_s_sleep(1);
            __builtin_amdgcn_buffer_wbinvl1_vol();
            __builtin_amdgcn_s_dcache_inv_vol();
            low = __scoped_atomic_load_n(&args.sync.peers[peer]->values[mailbox], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
            high = __scoped_atomic_load_n(&args.sync.peers[peer]->values[mailbox + 1], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
        }
        reinterpret_cast<unsigned short*>(args.logits)[row * Vocab + peer * LocalVocab + offset + column] = (low & 255u) | ((high & 255u) << 8);
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 633)) return false;
    if (blockIdx.x == 0 && threadIdx.x < Ranks) __scoped_atomic_store_n(&args.sync.peers[threadIdx.x]->consumed[args.sync.rank], epoch, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    return peer_wait(args.sync, true, epoch, 634);
}

__global__ __launch_bounds__(Threads, 1) void decode_tail(TailArguments args) {
    if (!run_mix(args.mix, 640)) return;
    gemv(args.mix.output, args.weight, args.local, nullptr, args.mix.rows, Hidden, LocalVocab, 0);
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 644)) return;
    for (int offset = 0; offset < LocalVocab; offset += Hyper) {
        if (!gather_logits(args, offset, min(Hyper, LocalVocab - offset))) return;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(args.executions, 1);
        atomicAdd(args.mix.executions, 1);
    }
    return;
}

extern "C" int flash_io_info(int64_t* result) {
    int entry_occupancy, tail_occupancy;
    hipError_t status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&entry_occupancy, decode_entry, Threads, 0);
    if (status != hipSuccess) return status;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&tail_occupancy, decode_tail, Threads, 0);
    if (status != hipSuccess) return status;
    result[0] = sizeof(EntryArguments);
    result[1] = sizeof(TailArguments);
    result[2] = entry_occupancy;
    result[3] = tail_occupancy;
    return hipSuccess;
}

extern "C" int flash_entry(EntryArguments* args, void* stream) {
    if (args->rows < 1 || args->rows > MaxRows || (args->index_bytes != 4 && args->index_bytes != 8) || args->vocab_start != args->sync.rank * LocalVocab) return hipErrorInvalidValue;
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(decode_entry), arguments, stream);
}

extern "C" int flash_tail(TailArguments* args, void* stream) {
    if (args->mix.rows < 1 || args->mix.rows > MaxRows) return hipErrorInvalidValue;
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(decode_tail), arguments, stream);
}
