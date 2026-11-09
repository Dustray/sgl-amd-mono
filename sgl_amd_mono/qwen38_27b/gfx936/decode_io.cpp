#include "kernel.cpp"

struct EntryArguments {
    K1Arguments k1;
    const BFloat16* embedding_weight;
    const void* input_ids;
    int64_t vocabulary;
    int64_t index_bytes;
};

struct TailArguments {
    const BFloat16* hidden;
    const BFloat16* residual;
    const BFloat16* norm_weight;
    const BFloat16* head_weight;
    BFloat16* normalized;
    BFloat16* residual_output;
    BFloat16* logits;
    int* executions;
    unsigned int* synchronization;
    int64_t rows;
    int64_t vocabulary;
    int64_t barrier_timeout_ticks;
};

__device__ float decode_head_dot(const BFloat16* input, const BFloat16* weight) {
    int lane = threadIdx.x & 63;
    float partials[2];
    for (int group = 0; group < 2; ++group) {
        Float4 accumulated = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int start = group * 512; start < Hidden; start += 1024) {
            int column = start + lane * 8;
            Short4 left = *reinterpret_cast<const Short4*>(input + column);
            Short4 right = *reinterpret_cast<const Short4*>(weight + column);
            accumulated = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulated);
            left = *reinterpret_cast<const Short4*>(input + column + 4);
            right = *reinterpret_cast<const Short4*>(weight + column + 4);
            accumulated = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulated);
        }
        float diagonal[16];
        for (int index = 0; index < 16; ++index)
            diagonal[index] = __shfl(accumulated[index / 4], (index % 4) * 17 + (index / 4) * 4, 64);
        float total = (diagonal[12] + diagonal[13]) + (diagonal[14] + diagonal[15]);
        for (int index = 0; index < 12; index += 4)
            total += (diagonal[index] + diagonal[index + 1]) + (diagonal[index + 2] + diagonal[index + 3]);
        partials[group] = total;
    }
    return partials[0] + partials[1];
}

__global__ __launch_bounds__(Threads, 1) void dense_entry_k1(EntryArguments args) {
    if (args.k1.synchronization[2]) return;
    if (blockIdx.x < args.k1.rows) {
        int row = blockIdx.x;
        int64_t token = args.index_bytes == 8 ? static_cast<const int64_t*>(args.input_ids)[row] : static_cast<const int*>(args.input_ids)[row];
        if (token < 0 || token >= args.vocabulary) {
            if (threadIdx.x == 0) atomicCAS(args.k1.synchronization + 2, 0u, 201u);
        } else {
            BFloat16* hidden = const_cast<BFloat16*>(args.k1.hidden);
            for (int column = threadIdx.x; column < Hidden; column += Threads)
                hidden[row * Hidden + column] = args.embedding_weight[token * Hidden + column];
        }
    }
    if (!synchronize_grid(args.k1.synchronization, args.k1.barrier_timeout_ticks, 200) || args.k1.synchronization[2]) return;
    run_dense_k1(args.k1);
}

__global__ __launch_bounds__(Threads, 1) void dense_tail(TailArguments args) {
    __shared__ float reduction[17];
    if (args.synchronization[2]) return;
    if (blockIdx.x < args.rows)
        normalize(args.hidden, args.residual, args.norm_weight, args.normalized, args.residual_output, blockIdx.x, reduction);
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 1)) return;
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    int lane = threadIdx.x & 63;
    for (int work = wave; work < args.rows * args.vocabulary; work += Blocks * 4) {
        int row = work / args.vocabulary;
        int channel = work % args.vocabulary;
        float value = decode_head_dot(args.normalized + row * Hidden, args.head_weight + int64_t(channel) * Hidden);
        if (lane == 0) args.logits[work] = BFloat16(value);
    }
    if (!synchronize_grid(args.synchronization, args.barrier_timeout_ticks, 2)) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(args.executions, 1);
}

extern "C" int mono_decode_io_info(int64_t* information) {
    int entry_occupancy, tail_occupancy;
    hipError_t status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&entry_occupancy, dense_entry_k1, Threads, 0);
    if (status != hipSuccess) return status;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&tail_occupancy, dense_tail, Threads, 0);
    information[0] = sizeof(EntryArguments);
    information[1] = sizeof(TailArguments);
    information[2] = entry_occupancy;
    information[3] = tail_occupancy;
    return status;
}

extern "C" int mono_entry_k1(EntryArguments* args, void* stream) {
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(dense_entry_k1), arguments, stream);
}

extern "C" int mono_tail(TailArguments* args, void* stream) {
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(dense_tail), arguments, stream);
}
