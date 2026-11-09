#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

static std::map<void*, size_t> allocations;

extern "C" hipError_t hipMalloc(void** pointer, size_t bytes) {
    static auto original = reinterpret_cast<hipError_t (*)(void**, size_t)>(dlsym(RTLD_NEXT, "hipMalloc"));
    auto status = original(pointer, bytes);
    if (status == hipSuccess) allocations[*pointer] = bytes;
    return status;
}

extern "C" hipError_t hipLaunchKernel(const void* function, dim3 grid, dim3 block,
                                      void** arguments, size_t shared, hipStream_t stream) {
    static auto original = reinterpret_cast<decltype(&hipLaunchKernel)>(dlsym(RTLD_NEXT, "hipLaunchKernel"));
    const char* directory = std::getenv("MONO_NATIVE_CAPTURE_DIR");
    const char* name = directory ? hipKernelNameRefByPtr(function, stream) : nullptr;
    auto status = original(function, grid, block, arguments, shared, stream);
    if (!directory || !name || !std::strstr(name, "paged_attention_kernel")) return status;
    if (status != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) return status;
    static int capture = 0;
    std::string prefix = std::string(directory) + "/launch-" + std::to_string(capture++);
    FILE* metadata = std::fopen((prefix + ".txt").c_str(), "w");
    if (!metadata) return status;
    std::fprintf(metadata, "%s\ngrid %u %u %u block %u %u %u shared %zu\n", name,
                 grid.x, grid.y, grid.z, block.x, block.y, block.z, shared);
    const int sizes[] = {8, 8, 8, 8, 8, 4, 4, 8, 8, 4, 8, 4, 4, 4, 4, 8, 4, 1, 4, 4, 4};
    for (int index = 0; index < 21; ++index) {
        uint64_t value = 0;
        std::memcpy(&value, arguments[index], sizes[index]);
        std::fprintf(metadata, "arg %d size %d hex %llx int %lld\n", index, sizes[index],
                     static_cast<unsigned long long>(value), static_cast<long long>(value));
        if (index < 2) {
            auto pointer = reinterpret_cast<void*>(value);
            size_t bytes = allocations.count(pointer) ? allocations[pointer] : 2048;
            std::vector<unsigned char> contents(bytes);
            auto copied = hipMemcpy(contents.data(), pointer, bytes, hipMemcpyDeviceToHost);
            std::fprintf(metadata, "buffer %d bytes %zu status %d\n", index, bytes, copied);
            if (copied == hipSuccess) {
                FILE* output = std::fopen((prefix + "-buffer" + std::to_string(index) + ".bin").c_str(), "wb");
                if (output) { std::fwrite(contents.data(), 1, bytes, output); std::fclose(output); }
            }
        }
    }
    std::fclose(metadata);
    return status;
}
