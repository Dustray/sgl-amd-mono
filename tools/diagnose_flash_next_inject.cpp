#include "kernel.cpp"

__global__ __launch_bounds__(Threads, 1) void combine_probe(PostArguments args) {
    inject(args.attention, args.residual, args.normalized_residual, args.attention_inject, args.inject_gates, args.post_attention, args.rows, args.sync, 700);
}

extern "C" int flash_probe_combine(PostArguments* args, void* stream) {
    void* arguments[] = {args};
    return launch_cooperative(reinterpret_cast<const void*>(combine_probe), arguments, stream);
}
