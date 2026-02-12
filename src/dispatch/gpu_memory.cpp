#include "gpu_memory.hpp"
#include <cstring>
#include <stdexcept>

// GPUMemory implementation
GPUMemory::GPUMemory(size_t size) : d_ptr_(nullptr), size_(size) {
    cudaError_t err = cudaMalloc(&d_ptr_, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
}

GPUMemory::~GPUMemory() {
    if (d_ptr_) {
        cudaFree(d_ptr_);
        d_ptr_ = nullptr;
    }
}

void GPUMemory::copy_to_device(const void* host_ptr) {
    cudaError_t err = cudaMemcpy(d_ptr_, host_ptr, size_, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy to device failed: " + std::string(cudaGetErrorString(err)));
    }
}

void GPUMemory::copy_to_host(void* host_ptr) const {
    cudaError_t err = cudaMemcpy(host_ptr, d_ptr_, size_, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy to host failed: " + std::string(cudaGetErrorString(err)));
    }
}

// PinnedMemory implementation
PinnedMemory::PinnedMemory(size_t size) : h_ptr_(nullptr), size_(size) {
    cudaError_t err = cudaMallocHost(&h_ptr_, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed: " + std::string(cudaGetErrorString(err)));
    }
}

PinnedMemory::~PinnedMemory() {
    if (h_ptr_) {
        cudaFreeHost(h_ptr_);
        h_ptr_ = nullptr;
    }
}

// Utility functions
cudaError_t gpu_memcpy(void* dst, const void* src, size_t size, cudaMemcpyKind kind) {
    return cudaMemcpy(dst, src, size, kind);
}

cudaError_t gpu_malloc(void** dev_ptr, size_t size) {
    return cudaMalloc(dev_ptr, size);
}

cudaError_t gpu_free(void* dev_ptr) {
    return cudaFree(dev_ptr);
}
