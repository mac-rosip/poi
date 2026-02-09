#pragma once

// =============================================================================
// gpu_device.hpp — GPU device abstraction
// =============================================================================
//
// Wraps CUDA device enumeration, selection, and properties.
// Each GPUDevice represents a single physical GPU.
//
// Dependencies: cuda_runtime.h
// =============================================================================

#include <string>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

struct GPUDeviceInfo {
    int id;
    std::string name;
    size_t total_memory;
    int sm_count;
    int max_threads_per_sm;
    int compute_major;
    int compute_minor;
    int warp_size;
};

class GPUDevice {
public:
    explicit GPUDevice(int device_id);
    ~GPUDevice();

    // Device info
    int id() const { return info_.id; }
    const std::string& name() const { return info_.name; }
    const GPUDeviceInfo& info() const { return info_; }

    // Set this device as current for CUDA calls
    void set_current() const;

    // Create a CUDA stream on this device
    cudaStream_t create_stream() const;
    static void destroy_stream(cudaStream_t stream);

    // Compute optimal thread count for this device
    uint32_t optimal_thread_count(size_t bytes_per_thread) const;

    // Static: enumerate all available CUDA devices
    static std::vector<GPUDeviceInfo> enumerate();

    // Static: select devices by ID list (empty = all)
    static std::vector<GPUDevice> select(const std::vector<int>& device_ids);

private:
    GPUDeviceInfo info_;
};
