#include "gpu_device.hpp"
#include <stdexcept>
#include <algorithm>

GPUDevice::GPUDevice(int device_id) {
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device properties for device " +
                                 std::to_string(device_id));
    }

    info_.id = device_id;
    info_.name = props.name;
    info_.total_memory = props.totalGlobalMem;
    info_.sm_count = props.multiProcessorCount;
    info_.max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    info_.compute_major = props.major;
    info_.compute_minor = props.minor;
    info_.warp_size = props.warpSize;
}

GPUDevice::~GPUDevice() = default;

void GPUDevice::set_current() const {
    cudaSetDevice(info_.id);
}

cudaStream_t GPUDevice::create_stream() const {
    set_current();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return stream;
}

void GPUDevice::destroy_stream(cudaStream_t stream) {
    cudaStreamDestroy(stream);
}

uint32_t GPUDevice::optimal_thread_count(size_t bytes_per_thread) const {
    // Use ~80% of available memory, capped by SM capacity
    size_t usable_memory = (info_.total_memory * 80) / 100;
    size_t memory_limited = usable_memory / bytes_per_thread;

    // SM-limited: total threads across all SMs
    size_t sm_limited = (size_t)info_.sm_count * info_.max_threads_per_sm;

    // Round down to multiple of 256 (block size)
    uint32_t count = static_cast<uint32_t>(std::min(memory_limited, sm_limited));
    count = (count / 256) * 256;

    // Minimum 256 threads
    return std::max(count, (uint32_t)256);
}

std::vector<GPUDeviceInfo> GPUDevice::enumerate() {
    int count = 0;
    cudaGetDeviceCount(&count);

    std::vector<GPUDeviceInfo> devices;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, i) == cudaSuccess) {
            GPUDeviceInfo info;
            info.id = i;
            info.name = props.name;
            info.total_memory = props.totalGlobalMem;
            info.sm_count = props.multiProcessorCount;
            info.max_threads_per_sm = props.maxThreadsPerMultiProcessor;
            info.compute_major = props.major;
            info.compute_minor = props.minor;
            info.warp_size = props.warpSize;
            devices.push_back(info);
        }
    }
    return devices;
}

std::vector<GPUDevice> GPUDevice::select(const std::vector<int>& device_ids) {
    std::vector<GPUDevice> devices;
    if (device_ids.empty()) {
        // Select all
        int count = 0;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) {
            devices.emplace_back(i);
        }
    } else {
        for (int id : device_ids) {
            devices.emplace_back(id);
        }
    }
    return devices;
}
