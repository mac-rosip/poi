#pragma once

// =============================================================================
// dispatcher.hpp — CUDA kernel orchestration and mining loop
// =============================================================================
//
// The Dispatcher manages the full vanity mining pipeline:
//   1. Initialize GPU devices and allocate memory
//   2. Generate random seeds and initialize key points
//   3. Run the iterate → score loop until a match is found
//   4. Report results (private key, address, score)
//
// Supports both secp256k1 (ETH/TRX) and Ed25519 (Solana) pipelines.
//
// Dependencies: gpu_device, gpu_memory, speed_sample, scorer, types
// =============================================================================

#include "gpu_device.hpp"
#include "gpu_memory.hpp"
#include "../speed_sample.hpp"
#include "../scoring/scorer.hpp"
#include "../types.hpp"
#include <vector>
#include <string>
#include <functional>
#include <cstdint>
#include <atomic>

// Result returned when a vanity match is found
struct VanityResult {
    bool found;
    uint32_t score;
    ChainType chain;
    std::string address;
    std::vector<uint8_t> private_key;
    std::vector<uint8_t> public_key_hash;
};

// Callback for progress reporting
using ProgressCallback = std::function<void(double speed, uint64_t total, uint32_t best_score)>;

// Configuration for the dispatcher
struct DispatcherConfig {
    ChainType chain;
    CurveType curve;
    Scorer scorer;
    std::vector<int> device_ids;        // Empty = all devices
    uint32_t iterations_per_launch;     // Iterations per kernel call
    bool benchmark_mode;

    DispatcherConfig()
        : chain(ChainType::ETHEREUM)
        , curve(CurveType::SECP256K1)
        , iterations_per_launch(256)
        , benchmark_mode(false)
    {}
};

class Dispatcher {
public:
    explicit Dispatcher(const DispatcherConfig& config);
    ~Dispatcher();

    // Initialize devices, allocate memory, generate precomp tables
    void init();

    // Run the mining loop. Returns when a result is found or stopped.
    VanityResult run(ProgressCallback progress_cb = nullptr);

    // Stop the mining loop (from another thread)
    void stop();

    // Get current speed
    double speed() const;
    uint64_t total_checked() const;

private:
    DispatcherConfig config_;
    std::vector<GPUDevice> devices_;
    SpeedSample speed_sample_;
    std::atomic<bool> running_;
    uint32_t best_score_;

    // Per-device state for secp256k1 pipeline
    struct Secp256k1State {
        cudaStream_t stream;
        GPUMemory* d_points_x;
        GPUMemory* d_points_y;
        GPUMemory* d_priv_keys;
        GPUMemory* d_hashes;
        GPUMemory* d_results;
        uint32_t num_points;
    };

    // Per-device state for Ed25519 pipeline
    struct Ed25519State {
        cudaStream_t stream;
        GPUMemory* d_seeds;
        GPUMemory* d_pubkeys;
        GPUMemory* d_priv_seeds;
        GPUMemory* d_results;
        // Pinned host buffers for DMA
        PinnedMemory* h_pubkeys;
        PinnedMemory* h_priv_seeds;
        uint32_t num_points;
    };

    std::vector<Secp256k1State> secp_states_;
    std::vector<Ed25519State> ed_states_;

    // Internal methods
    void init_secp256k1();
    void init_ed25519();
    VanityResult run_secp256k1(ProgressCallback progress_cb);
    VanityResult run_ed25519(ProgressCallback progress_cb);
    void generate_random_seeds(uint8_t* buffer, size_t size);
    void cleanup();
};
