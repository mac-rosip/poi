#pragma once

#include <cstdint>
#include <cuda_runtime.h>

class GPUMemory {
public:
    GPUMemory(size_t size);
    ~GPUMemory();

    void* get() const { return d_ptr_; }
    size_t size() const { return size_; }

    void copy_to_device(const void* host_ptr);
    void copy_to_host(void* host_ptr) const;

private:
    void* d_ptr_;
    size_t size_;
};

class PinnedMemory {
public:
    PinnedMemory(size_t size);
    ~PinnedMemory();

    void* get() const { return h_ptr_; }
    size_t size() const { return size_; }

private:
    void* h_ptr_;
    size_t size_;
};

// Utility functions
cudaError_t gpu_memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind);
cudaError_t gpu_malloc(void** dev_ptr, size_t size);
cudaError_t gpu_free(void* dev_ptr);