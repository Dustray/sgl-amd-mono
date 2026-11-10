#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

__global__ void cooperative_probe(const float* source, float* scratch, float* output) {
    auto grid = cooperative_groups::this_grid();
    if (threadIdx.x == 0) scratch[blockIdx.x] = source[blockIdx.x];
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (int block_index = 0; block_index < gridDim.x; ++block_index) total += scratch[block_index];
        *output = total;
    }
    grid.sync();
}

extern "C" int cooperative_info(int* supported, int* multiprocessors, int* occupancy) {
    int device_index;
    hipError_t status = hipGetDevice(&device_index);
    if (status != hipSuccess) return status;
    status = hipDeviceGetAttribute(supported, hipDeviceAttributeCooperativeLaunch, device_index);
    if (status != hipSuccess) return status;
    status = hipDeviceGetAttribute(multiprocessors, hipDeviceAttributeMultiprocessorCount, device_index);
    if (status != hipSuccess) return status;
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(occupancy, cooperative_probe, 256, 0);
}

extern "C" int launch_probe(void* source, void* scratch, void* output, void* stream) {
    void* arguments[] = {&source, &scratch, &output};
    hipLaunchAttribute attribute{};
    attribute.id = hipLaunchAttributeCooperative;
    attribute.val.cooperative = 1;
    hipLaunchConfig_t config{};
    config.gridDim = dim3(80);
    config.blockDim = dim3(256);
    config.stream = reinterpret_cast<hipStream_t>(stream);
    config.attrs = &attribute;
    config.numAttrs = 1;
    return hipLaunchKernelExC(&config, reinterpret_cast<void*>(cooperative_probe), arguments);
}
