#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>
#include <cstring>

using BFloat16 = hip_bfloat16;
using Short4 = short __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));
constexpr int Hidden = 2560;
constexpr int Branches = 4;
constexpr int Hyper = Hidden * Branches;
constexpr int Lowrank = 320;
constexpr int Experts = 512;
constexpr int Topk = 10;
constexpr int Intermediate = 80;
constexpr int Ranks = 8;
constexpr int Blocks = 80;
constexpr int Threads = 256;
constexpr int MaxRows = 2;

struct PeerMemory {
    unsigned int published[Ranks];
    unsigned int consumed[Ranks];
    unsigned int failure;
    unsigned int reserved;
    unsigned int values[MaxRows * Hyper * 2];
};

struct SyncArguments {
    PeerMemory* peers[Ranks];
    unsigned int* state;
    int64_t rank;
    int64_t timeout_ticks;
};

struct MixArguments {
    const BFloat16* input;
    const BFloat16* norm_weight;
    const BFloat16* down_weight;
    const BFloat16* up_weight;
    BFloat16* normalized;
    BFloat16* lowrank;
    float* gates;
    BFloat16* output;
    unsigned int* synchronization;
    int* executions;
    int64_t rows;
    int64_t timeout_ticks;
};

struct PostArguments {
    const BFloat16* attention;
    const BFloat16* residual;
    const BFloat16* normalized_residual;
    const BFloat16* attention_inject;
    const BFloat16* norm_weight;
    const BFloat16* mix_down;
    const BFloat16* mix_up;
    const BFloat16* mlp_inject;
    const BFloat16* router_weight;
    const BFloat16* shared_gate_weight;
    const BFloat16* expert_gate_up;
    const BFloat16* expert_down;
    const BFloat16* shared_gate_up;
    const BFloat16* shared_down;
    BFloat16* reduced_attention;
    BFloat16* post_attention;
    BFloat16* normalized;
    BFloat16* lowrank;
    float* gates;
    BFloat16* mixed;
    BFloat16* router_logits;
    int* top_ids;
    float* top_weights;
    float* inject_gates;
    BFloat16* projected;
    BFloat16* activated;
    BFloat16* expert_output;
    BFloat16* local_moe;
    BFloat16* reduced_moe;
    BFloat16* output;
    int* executions;
    SyncArguments sync;
    int64_t rows;
};

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float wave_sum(float value) {
    for (int offset = 32; offset > 0; offset /= 2) value += __shfl_down(value, offset, 64);
    return __shfl(value, 0, 64);
}

// Device-side stage timing. The host attaches one int64 buffer per rank; grid_sync records
// the block-0 arrival and release timestamps for each barrier.
// Layout: [0] cursor, [1] dropped flag, [2 + 2n] = stage, [3 + 2n] = ticks.
__device__ int64_t* stage_buffer = nullptr;
__device__ int64_t stage_capacity = 0;

__device__ __forceinline__ void stage_mark(unsigned int stage) {
    int64_t capacity = stage_capacity;
    if (stage_buffer == nullptr || capacity < 4 || blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned long long index = atomicAdd(reinterpret_cast<unsigned long long*>(stage_buffer), 1ull);
    if (index >= static_cast<unsigned long long>(capacity / 2 - 1)) {
        stage_buffer[1] = 1;
        return;
    }
    volatile int64_t* record = stage_buffer + 2 + index * 2;
    record[0] = static_cast<int64_t>(stage);
    record[1] = static_cast<int64_t>(wall_clock64());
}

__device__ __forceinline__ bool grid_sync(unsigned int* state, int64_t timeout, unsigned int stage) {
    __shared__ int success;
    stage_mark(stage + 1000);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        success = 1;
        unsigned int epoch = atomicAdd(state + 1, 0u);
        unsigned int arrived = atomicAdd(state, 1u);
        if (arrived == Blocks - 1) {
            atomicExch(state, 0u);
            atomicExch(state + 1, epoch + 1);
        } else {
            int64_t begin = wall_clock64();
            while (atomicAdd(state + 1, 0u) == epoch) {
                if (atomicAdd(state + 2, 0u) || wall_clock64() - begin > timeout) {
                    atomicCAS(state + 2, 0u, stage);
                    success = 0;
                    break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
        if (atomicAdd(state + 2, 0u)) success = 0;
    }
    __syncthreads();
    __threadfence();
    if (success) stage_mark(stage);
    return success;
}

__device__ __forceinline__ float matvec16(const BFloat16* input, const BFloat16* weight, int width, int valid_outputs = 16, int split_k = 0) {
    int lane = threadIdx.x & 63;
    Float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    float partial_sum = 0.0f;
    for (int start = 0; start < width; start += 16) {
        int column = start + (lane / 16) * 4;
        Short4 left = {0, 0, 0, 0};
        Short4 right = {0, 0, 0, 0};
        if (lane % 16 < valid_outputs) left = *reinterpret_cast<const Short4*>(weight + int64_t(lane % 16) * width + column);
        if (lane % 16 == 0) right = *reinterpret_cast<const Short4*>(input + column);
        accumulator = __builtin_hcu_mmac_f32_16x16x16_bf16(left, right, accumulator);
        if (split_k && ((start + 16) % split_k == 0 || start + 16 == width)) {
            partial_sum += accumulator[0];
            accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    return split_k ? partial_sum : accumulator[0];
}

__device__ __forceinline__ void gemv(const BFloat16* input, const BFloat16* weight, BFloat16* output, float* raw, int rows, int width, int outputs, int activation) {
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    int lane = threadIdx.x & 63;
    int tiles = (outputs + 15) / 16;
    for (int task = wave; task < rows * tiles; task += Blocks * 4) {
        int row = task / tiles;
        int base = task % tiles * 16;
        int valid = outputs - base < 16 ? outputs - base : 16;
        float value = matvec16(input + row * width, weight + int64_t(base) * width, width, valid, activation == 1 ? 256 : 0);
        if (lane < valid) {
            if (activation == 1) value = value * 0.25f * sigmoid(value * 0.25f);
            if (raw != nullptr) raw[row * outputs + base + lane] = value;
            if (output != nullptr) output[row * outputs + base + lane] = BFloat16(value);
        }
    }
}

__device__ __forceinline__ void normalize(const BFloat16* input, const BFloat16* weight, BFloat16* output, int rows) {
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    int lane = threadIdx.x & 63;
    for (int task = wave; task < rows * Branches; task += Blocks * 4) {
        int offset = task * Hidden;
        float sum = 0.0f;
        for (int column = lane; column < Hidden; column += 64) {
            float value = float(input[offset + column]);
            sum += value * value;
        }
        float inverse = rsqrtf(wave_sum(sum) / Hidden + 1.0e-6f);
        for (int column = lane; column < Hidden; column += 64) {
            float value = float(input[offset + column]);
            output[offset + column] = BFloat16(value * inverse * (1.0f + float(weight[(task % Branches) * Hidden + column])));
        }
    }
}

__device__ __forceinline__ bool run_mix(MixArguments args, unsigned int stage) {
    normalize(args.input, args.norm_weight, args.normalized, args.rows);
    if (!grid_sync(args.synchronization, args.timeout_ticks, stage)) return false;
    gemv(args.normalized, args.down_weight, args.lowrank, nullptr, args.rows, Hyper, Lowrank, 1);
    if (!grid_sync(args.synchronization, args.timeout_ticks, stage + 1)) return false;
    gemv(args.lowrank, args.up_weight, nullptr, args.gates, args.rows, Lowrank, Hyper, 0);
    if (!grid_sync(args.synchronization, args.timeout_ticks, stage + 2)) return false;
    for (int index = blockIdx.x * Threads + threadIdx.x; index < args.rows * Hidden; index += Blocks * Threads) {
        int row = index / Hidden;
        int column = index % Hidden;
        float value = 0.0f;
        for (int branch = 0; branch < Branches; ++branch) {
            int offset = row * Hyper + branch * Hidden + column;
            value += sigmoid(args.gates[offset]) * float(args.normalized[offset]);
        }
        args.output[index] = BFloat16(value * 0.25f);
    }
    return grid_sync(args.synchronization, args.timeout_ticks, stage + 3);
}

__device__ __forceinline__ bool peer_wait(SyncArguments sync, bool consumed, unsigned int epoch, unsigned int stage) {
    if (blockIdx.x == 0 && threadIdx.x < Ranks) {
        int64_t begin = wall_clock64();
        int peer = threadIdx.x;
        {
            unsigned int* flag = consumed ? &sync.peers[sync.rank]->consumed[peer] : &sync.peers[sync.rank]->published[peer];
            while (__scoped_atomic_load_n(flag, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE) < epoch) {
                if (wall_clock64() - begin > sync.timeout_ticks) {
                    sync.state[4] = epoch;
                    sync.state[5] = peer;
                    sync.state[6] = __scoped_atomic_load_n(flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
                    sync.state[7] = sync.rank;
                    atomicCAS(sync.state + 2, 0u, stage);
                    __scoped_atomic_store_n(&sync.peers[sync.rank]->failure, stage, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
                    break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    return grid_sync(sync.state, sync.timeout_ticks, stage);
}

__device__ __forceinline__ bool all_reduce(const BFloat16* input, BFloat16* output, int count, SyncArguments sync, unsigned int stage) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int previous = atomicAdd(sync.state + 3, 1u);
        if (previous >= 0xffffffu) atomicCAS(sync.state + 2, 0u, 900u);
    }
    if (!grid_sync(sync.state, sync.timeout_ticks, stage)) return false;
    unsigned int epoch = atomicAdd(sync.state + 3, 0u);
    for (int index = blockIdx.x * Threads + threadIdx.x; index < count; index += Blocks * Threads) {
        unsigned int bits = reinterpret_cast<const unsigned short*>(input)[index];
        unsigned int low = (epoch << 8) | (bits & 255u);
        unsigned int high = (epoch << 8) | (bits >> 8);
        __scoped_atomic_store_n(&sync.peers[sync.rank]->values[index * 2], low, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        __scoped_atomic_store_n(&sync.peers[sync.rank]->values[index * 2 + 1], high, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
    __threadfence_system();
    if (!grid_sync(sync.state, sync.timeout_ticks, stage + 1)) return false;
    if (blockIdx.x == 0 && threadIdx.x < Ranks) __scoped_atomic_store_n(&sync.peers[threadIdx.x]->published[sync.rank], epoch, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    for (int index = blockIdx.x * Threads + threadIdx.x; index < count; index += Blocks * Threads) {
        float value = 0.0f;
        for (int peer = 0; peer < Ranks; ++peer) {
            int64_t begin = wall_clock64();
            unsigned int low = __scoped_atomic_load_n(&sync.peers[peer]->values[index * 2], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
            unsigned int high = __scoped_atomic_load_n(&sync.peers[peer]->values[index * 2 + 1], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
            while ((low >> 8) != epoch || (high >> 8) != epoch) {
                if ((low >> 8) > epoch || (high >> 8) > epoch || wall_clock64() - begin > sync.timeout_ticks) {
                    atomicCAS(sync.state + 2, 0u, stage + 2);
                    break;
                }
                __builtin_amdgcn_s_sleep(1);
                __builtin_amdgcn_buffer_wbinvl1_vol();
                __builtin_amdgcn_s_dcache_inv_vol();
                low = __scoped_atomic_load_n(&sync.peers[peer]->values[index * 2], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
                high = __scoped_atomic_load_n(&sync.peers[peer]->values[index * 2 + 1], __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
            }
            value += __uint_as_float(((low & 255u) | ((high & 255u) << 8)) << 16);
        }
        reinterpret_cast<unsigned short*>(output)[index] = __float_as_uint(value) >> 16;
    }
    __threadfence_system();
    if (!grid_sync(sync.state, sync.timeout_ticks, stage + 3)) return false;
    if (blockIdx.x == 0 && threadIdx.x < Ranks) __scoped_atomic_store_n(&sync.peers[threadIdx.x]->consumed[sync.rank], epoch, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    return peer_wait(sync, true, epoch, stage + 4);
}

__device__ __forceinline__ bool inject(const BFloat16* block, const BFloat16* residual, const BFloat16* normalized, const BFloat16* weight, float* gates, BFloat16* output, int rows, SyncArguments sync, unsigned int stage) {
    __shared__ float partials[Branches][8];
    for (int row = blockIdx.x; row < rows; row += Blocks) {
        for (int branch = 0; branch < Branches; ++branch) {
            float sum = 0.0f;
            for (int vector = 0; vector < 5; ++vector) {
                int offset = (threadIdx.x + vector * Threads) * 8;
                for (int pair = 0; pair < 4; ++pair) {
                    int column = offset + pair * 2;
                    float first = float(normalized[row * Hyper + column]) * float(weight[branch * Hyper + column]);
                    float second = float(normalized[row * Hyper + column + 1]) * float(weight[branch * Hyper + column + 1]);
                    sum += first + second;
                }
            }
            for (int distance = 16; distance > 0; distance /= 2) sum += __shfl_xor(sum, distance, 32);
            if (threadIdx.x % 32 == 0) partials[branch][threadIdx.x / 32] = sum;
        }
        __syncthreads();
        if (threadIdx.x < Branches) {
            float sum = 0.0f;
            for (int split = 0; split < 8; ++split) sum += partials[threadIdx.x][split];
            gates[row * Branches + threadIdx.x] = sum;
        }
        __syncthreads();
    }
    if (!grid_sync(sync.state, sync.timeout_ticks, stage)) return false;
    for (int index = blockIdx.x * Threads + threadIdx.x; index < rows * Hyper; index += Blocks * Threads) {
        int row = index / Hyper;
        int branch = index % Hyper / Hidden;
        int column = index % Hidden;
        float scale = 2.0f * sigmoid(gates[row * Branches + branch] * 0.25f);
        output[index] = BFloat16(fmaf(scale, float(block[row * Hidden + column]), float(residual[index])));
    }
    return grid_sync(sync.state, sync.timeout_ticks, stage + 1);
}

__global__ __launch_bounds__(Threads, 1) void hc_mix(MixArguments args) {
    stage_mark(700);
    if (run_mix(args, 1) && blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(args.executions, 1);
}

__global__ __launch_bounds__(Threads, 1) void post_attention(PostArguments args) {
    stage_mark(701);
    if (!all_reduce(args.attention, args.reduced_attention, args.rows * Hidden, args.sync, 10)) return;
    if (!inject(args.reduced_attention, args.residual, args.normalized_residual, args.attention_inject, args.inject_gates, args.post_attention, args.rows, args.sync, 20)) return;
    MixArguments mix{args.post_attention, args.norm_weight, args.mix_down, args.mix_up, args.normalized, args.lowrank, args.gates, args.mixed, args.sync.state, args.executions, args.rows, args.sync.timeout_ticks};
    if (!run_mix(mix, 30)) return;
    gemv(args.mixed, args.router_weight, args.router_logits, nullptr, args.rows, Hidden, Experts, 0);
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 40)) return;
    if (blockIdx.x == 0 && threadIdx.x < args.rows) {
        int row = threadIdx.x;
        float maximum = -INFINITY;
        for (int expert = 0; expert < Experts; ++expert) maximum = fmaxf(maximum, float(args.router_logits[row * Experts + expert]));
        float total = 0.0f;
        for (int selected = 0; selected < Topk; ++selected) {
            int best = -1;
            float value = -INFINITY;
            for (int expert = 0; expert < Experts; ++expert) {
                bool used = false;
                for (int previous = 0; previous < selected; ++previous) used |= args.top_ids[row * Topk + previous] == expert;
                float candidate = float(args.router_logits[row * Experts + expert]);
                if (!used && candidate > value) { value = candidate; best = expert; }
            }
            args.top_ids[row * Topk + selected] = best;
            float probability = expf(value - maximum);
            args.top_weights[row * Topk + selected] = probability;
            total += probability;
        }
        for (int selected = 0; selected < Topk; ++selected) args.top_weights[row * Topk + selected] /= total;
    }
    gemv(args.mixed, args.shared_gate_weight, nullptr, args.inject_gates + args.rows * Branches, args.rows, Hidden, 1, 0);
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 41)) return;
    int wave = blockIdx.x * 4 + threadIdx.x / 64;
    int lane = threadIdx.x & 63;
    constexpr int Slots = Topk + 1;
    constexpr int UpTiles = Intermediate * 2 / 16;
    for (int task = wave; task < args.rows * Slots * UpTiles; task += Blocks * 4) {
        int row = task / (Slots * UpTiles);
        int slot = task / UpTiles % Slots;
        int base = task % UpTiles * 16;
        const BFloat16* weight = slot == Topk ? args.shared_gate_up : args.expert_gate_up + int64_t(args.top_ids[row * Topk + slot]) * (2 * Intermediate) * Hidden;
        float value = matvec16(args.mixed + row * Hidden, weight + int64_t(base) * Hidden, Hidden);
        if (lane < 16) args.projected[(row * Slots + slot) * Intermediate * 2 + base + lane] = BFloat16(value);
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 42)) return;
    for (int index = blockIdx.x * Threads + threadIdx.x; index < args.rows * Slots * Intermediate; index += Blocks * Threads) {
        int pair = index / Intermediate;
        int column = index % Intermediate;
        float gate = float(args.projected[pair * Intermediate * 2 + column]);
        float up = float(args.projected[pair * Intermediate * 2 + Intermediate + column]);
        float activated = float(BFloat16(gate * sigmoid(gate)));
        args.activated[index] = BFloat16(activated * up);
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 43)) return;
    for (int task = wave; task < args.rows * Slots * (Hidden / 16); task += Blocks * 4) {
        int row = task / (Slots * (Hidden / 16));
        int slot = task / (Hidden / 16) % Slots;
        int base = task % (Hidden / 16) * 16;
        const BFloat16* weight = slot == Topk ? args.shared_down : args.expert_down + int64_t(args.top_ids[row * Topk + slot]) * Hidden * Intermediate;
        float value = matvec16(args.activated + (row * Slots + slot) * Intermediate, weight + int64_t(base) * Intermediate, Intermediate);
        if (slot != Topk) value *= args.top_weights[row * Topk + slot];
        if (lane < 16) args.expert_output[(row * Slots + slot) * Hidden + base + lane] = BFloat16(value);
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 44)) return;
    for (int index = blockIdx.x * Threads + threadIdx.x; index < args.rows * Hidden; index += Blocks * Threads) {
        int row = index / Hidden;
        int column = index % Hidden;
        float value = 0.0f;
        for (int slot = 0; slot < Topk; ++slot) value += float(args.expert_output[(row * Slots + slot) * Hidden + column]);
        float routed = float(BFloat16(value));
        float shared = float(args.expert_output[(row * Slots + Topk) * Hidden + column]);
        float gate = args.inject_gates[args.rows * Branches + row];
        args.local_moe[index] = BFloat16(fmaf(shared, sigmoid(gate), routed));
    }
    if (!grid_sync(args.sync.state, args.sync.timeout_ticks, 45)) return;
    if (!all_reduce(args.local_moe, args.reduced_moe, args.rows * Hidden, args.sync, 50)) return;
    if (!inject(args.reduced_moe, args.post_attention, args.normalized, args.mlp_inject, args.inject_gates, args.output, args.rows, args.sync, 60)) return;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(args.executions, 1);
}

__global__ __launch_bounds__(Threads, 1) void reduce_probe(const BFloat16* input, BFloat16* output, int count, SyncArguments sync) {
    all_reduce(input, output, count, sync, 100);
}

__global__ __launch_bounds__(Threads, 1) void reduce_pair_probe(const BFloat16* first, const BFloat16* second, BFloat16* first_output, BFloat16* second_output, int count, SyncArguments sync) {
    if (!all_reduce(first, first_output, count, sync, 100)) return;
    all_reduce(second, second_output, count, sync, 200);
}

hipError_t launch_cooperative(const void* kernel, void** arguments, void* stream) {
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
        parameters.gridDim = dim3(Blocks);
        parameters.blockDim = dim3(Threads);
        parameters.kernelParams = arguments;
        hipGraphNode_t node;
        status = hipGraphAddKernelNode(&node, graph, dependencies, dependency_count, &parameters);
        if (status != hipSuccess) return status;
        hipKernelNodeAttrValue attribute{};
        attribute.cooperative = 1;
        status = hipGraphKernelNodeSetAttribute(node, hipKernelNodeAttributeCooperative, &attribute);
        if (status != hipSuccess) return status;
        attribute.cooperative = 0;
        status = hipGraphKernelNodeGetAttribute(node, hipKernelNodeAttributeCooperative, &attribute);
        if (status != hipSuccess || !attribute.cooperative) return hipErrorStreamCaptureUnsupported;
        return hipStreamUpdateCaptureDependencies(launch_stream, &node, 1, hipStreamSetCaptureDependencies);
    }
    hipLaunchAttribute attribute{};
    attribute.id = hipLaunchAttributeCooperative;
    attribute.val.cooperative = 1;
    hipLaunchConfig_t config{};
    config.gridDim = dim3(Blocks);
    config.blockDim = dim3(Threads);
    config.stream = launch_stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return hipLaunchKernelExC(&config, kernel, arguments);
}

extern "C" int flash_info(int64_t* result) {
    int device, cooperative, multiprocessors, mix_occupancy, post_occupancy, clock;
    hipError_t status = hipGetDevice(&device);
    if (status != hipSuccess) return status;
    status = hipDeviceGetAttribute(&cooperative, hipDeviceAttributeCooperativeLaunch, device);
    if (status != hipSuccess) return status;
    status = hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, device);
    if (status != hipSuccess) return status;
    status = hipDeviceGetAttribute(&clock, hipDeviceAttributeWallClockRate, device);
    if (status != hipSuccess) return status;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&mix_occupancy, hc_mix, Threads, 0);
    if (status != hipSuccess) return status;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&post_occupancy, post_attention, Threads, 0);
    if (status != hipSuccess) return status;
    int64_t values[] = {sizeof(MixArguments), sizeof(PostArguments), sizeof(SyncArguments), sizeof(PeerMemory), cooperative, multiprocessors, mix_occupancy, post_occupancy, clock};
    std::memcpy(result, values, sizeof(values));
    return hipSuccess;
}

extern "C" int flash_stage_attach(void* buffer, int64_t capacity) {
    int64_t ticks = capacity;
    hipError_t status = hipMemcpyToSymbol(HIP_SYMBOL(stage_buffer), &buffer, sizeof(void*));
    if (status != hipSuccess) return status;
    return hipMemcpyToSymbol(HIP_SYMBOL(stage_capacity), &ticks, sizeof(ticks));
}

extern "C" int flash_allocate(void** pointer, void* handle) {
    constexpr size_t granule = 2 << 20;
    constexpr size_t allocation = (sizeof(PeerMemory) + granule - 1) / granule * granule;
    hipError_t status = hipExtMallocWithFlags(pointer, allocation, hipDeviceMallocUncached);
    if (status != hipSuccess) return status;
    status = hipMemset(*pointer, 0, allocation);
    if (status != hipSuccess) return status;
    return hipIpcGetMemHandle(reinterpret_cast<hipIpcMemHandle_t*>(handle), *pointer);
}

extern "C" int flash_open(void** pointer, const void* handle) {
    hipIpcMemHandle_t value;
    std::memcpy(&value, handle, sizeof(value));
    return hipIpcOpenMemHandle(pointer, value, hipIpcMemLazyEnablePeerAccess);
}

extern "C" int flash_close(void* pointer, int owner) {
    return owner ? hipFree(pointer) : hipIpcCloseMemHandle(pointer);
}

extern "C" int flash_snapshot(void* pointer, unsigned int* output) {
    return hipMemcpy(output, pointer, sizeof(unsigned int) * 18, hipMemcpyDeviceToHost);
}

extern "C" int flash_mix(MixArguments* args, void* stream) {
    if (args->rows < 1 || args->rows > MaxRows || args->timeout_ticks < 1) return hipErrorInvalidValue;
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(hc_mix), arguments, stream);
}

extern "C" int flash_post(PostArguments* args, void* stream) {
    if (args->rows < 1 || args->rows > MaxRows || args->sync.rank < 0 || args->sync.rank >= Ranks) return hipErrorInvalidValue;
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(post_attention), arguments, stream);
}

extern "C" int flash_reduce(const BFloat16* input, BFloat16* output, int count, SyncArguments* sync, void* stream) {
    if (count < 1 || count > MaxRows * Hyper) return hipErrorInvalidValue;
    void* arguments[] = {&input, &output, &count, sync};
    return launch_cooperative(reinterpret_cast<const void*>(reduce_probe), arguments, stream);
}

extern "C" int flash_reduce_pair(const BFloat16* first, const BFloat16* second, BFloat16* first_output, BFloat16* second_output, int count, SyncArguments* sync, void* stream) {
    if (count < 1 || count > MaxRows * Hyper) return hipErrorInvalidValue;
    void* arguments[] = {&first, &second, &first_output, &second_output, &count, sync};
    return launch_cooperative(reinterpret_cast<const void*>(reduce_pair_probe), arguments, stream);
}
