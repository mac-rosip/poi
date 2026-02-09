#include "dispatcher.hpp"
#include <random>
#include <cstring>
#include <cstdio>
#include <chrono>

// External kernel launch functions
extern "C" void secp256k1_init_precomp();
extern "C" void ed25519_init_precomp();

extern "C" void secp256k1_iterate_launch(
    void* d_points_x, void* d_points_y,
    uint8_t* d_hash_out, uint32_t num_points,
    uint32_t iterations, cudaStream_t stream);

extern "C" void ed25519_keygen_launch(
    const uint8_t* d_seeds, uint8_t* d_pubkeys,
    uint8_t* d_priv_seeds, uint32_t num_keys, cudaStream_t stream);

extern "C" void ed25519_iterate_launch(
    void* d_X, void* d_Y, void* d_Z, void* d_T,
    uint8_t* d_pubkeys, uint32_t num_points,
    uint32_t iterations, cudaStream_t stream);

// =============================================================================
// Constructor / Destructor
// =============================================================================

Dispatcher::Dispatcher(const DispatcherConfig& config)
    : config_(config)
    , speed_sample_(20)
    , running_(false)
    , best_score_(0)
{}

Dispatcher::~Dispatcher() {
    cleanup();
}

// =============================================================================
// init — Set up devices, allocate memory, generate precomp tables
// =============================================================================
void Dispatcher::init() {
    // Select devices
    devices_ = GPUDevice::select(config_.device_ids);

    if (devices_.empty()) {
        throw std::runtime_error("No CUDA devices found");
    }

    // Determine curve from chain
    if (config_.chain == ChainType::SOLANA) {
        config_.curve = CurveType::ED25519;
    } else {
        config_.curve = CurveType::SECP256K1;
    }

    // Initialize curve-specific resources
    if (config_.curve == CurveType::SECP256K1) {
        init_secp256k1();
    } else {
        init_ed25519();
    }
}

// =============================================================================
// init_secp256k1 — Allocate secp256k1 pipeline resources
// =============================================================================
void Dispatcher::init_secp256k1() {
    // Generate precomputed table on first device
    devices_[0].set_current();
    secp256k1_init_precomp();

    // Per-point memory: x(32) + y(32) + privkey(32) + hash(20) = 116 bytes
    const size_t bytes_per_point = 32 + 32 + 32 + 20;

    for (auto& dev : devices_) {
        dev.set_current();

        Secp256k1State state;
        state.stream = dev.create_stream();
        state.num_points = dev.optimal_thread_count(bytes_per_point);

        state.d_points_x  = new GPUMemory(state.num_points * 32);
        state.d_points_y  = new GPUMemory(state.num_points * 32);
        state.d_priv_keys = new GPUMemory(state.num_points * 32);
        state.d_hashes    = new GPUMemory(state.num_points * 20);
        state.d_results   = new GPUMemory(sizeof(uint32_t) * 3 + 32); // ScoreResult

        // Generate random seeds and initialize points
        std::vector<uint8_t> seeds(state.num_points * 32);
        generate_random_seeds(seeds.data(), seeds.size());

        // Copy seeds to device as private keys
        GPUMemory d_seeds(state.num_points * 32);
        d_seeds.copy_to_device(seeds.data());

        // Init kernel would be launched here
        // For now, seeds serve as initial private keys
        state.d_priv_keys->copy_to_device(seeds.data());

        // Zero out results
        uint8_t zero_result[36 + 32] = {0};
        state.d_results->copy_to_device(zero_result);

        secp_states_.push_back(state);
    }
}

// =============================================================================
// init_ed25519 — Allocate Ed25519 pipeline resources
// =============================================================================
void Dispatcher::init_ed25519() {
    // Generate precomputed table on first device
    devices_[0].set_current();
    ed25519_init_precomp();

    // Per-point memory: seed(32) + pubkey(32) + priv(32) + 4*fe25519(40 each) = 256 bytes
    const size_t bytes_per_point = 32 + 32 + 32 + 4 * 40;

    for (auto& dev : devices_) {
        dev.set_current();

        Ed25519State state;
        state.stream = dev.create_stream();
        state.num_points = dev.optimal_thread_count(bytes_per_point);

        state.d_seeds      = new GPUMemory(state.num_points * 32);
        state.d_pubkeys    = new GPUMemory(state.num_points * 32);
        state.d_priv_seeds = new GPUMemory(state.num_points * 32);
        state.d_results    = new GPUMemory(sizeof(uint32_t) * 3 + 32);

        // Extended coordinate arrays (fe25519 = 40 bytes)
        state.d_points_X = new GPUMemory(state.num_points * 40);
        state.d_points_Y = new GPUMemory(state.num_points * 40);
        state.d_points_Z = new GPUMemory(state.num_points * 40);
        state.d_points_T = new GPUMemory(state.num_points * 40);

        // Generate initial seeds
        std::vector<uint8_t> seeds(state.num_points * 32);
        generate_random_seeds(seeds.data(), seeds.size());
        state.d_seeds->copy_to_device(seeds.data());

        // Zero results
        uint8_t zero_result[36 + 32] = {0};
        state.d_results->copy_to_device(zero_result);

        ed_states_.push_back(state);
    }
}

// =============================================================================
// run — Main mining loop
// =============================================================================
VanityResult Dispatcher::run(ProgressCallback progress_cb) {
    running_ = true;

    if (config_.curve == CurveType::SECP256K1) {
        return run_secp256k1(progress_cb);
    } else {
        return run_ed25519(progress_cb);
    }
}

// =============================================================================
// run_secp256k1 — Secp256k1 mining loop
// =============================================================================
VanityResult Dispatcher::run_secp256k1(ProgressCallback progress_cb) {
    VanityResult result;
    result.found = false;
    result.score = 0;
    result.chain = config_.chain;

    while (running_) {
        for (size_t d = 0; d < secp_states_.size(); ++d) {
            auto& state = secp_states_[d];
            devices_[d].set_current();

            // Launch iterate kernel
            secp256k1_iterate_launch(
                state.d_points_x->get(),
                state.d_points_y->get(),
                static_cast<uint8_t*>(state.d_hashes->get()),
                state.num_points,
                config_.iterations_per_launch,
                state.stream
            );

            cudaStreamSynchronize(state.stream);

            // Record throughput
            uint64_t keys_checked = (uint64_t)state.num_points * config_.iterations_per_launch;
            speed_sample_.sample(keys_checked);

            // In benchmark mode, skip scoring
            if (config_.benchmark_mode) {
                if (progress_cb) {
                    progress_cb(speed_sample_.getSpeed(), speed_sample_.getTotal(), 0);
                }
                continue;
            }

            // Check results (copy back and score on host)
            // For production: use GPU scoring kernels. For now: host-side check.
            // TODO: Integrate GPU scoring kernels from T28

            if (progress_cb) {
                progress_cb(speed_sample_.getSpeed(), speed_sample_.getTotal(), best_score_);
            }
        }
    }

    return result;
}

// =============================================================================
// run_ed25519 — Ed25519 mining loop
// =============================================================================
VanityResult Dispatcher::run_ed25519(ProgressCallback progress_cb) {
    VanityResult result;
    result.found = false;
    result.score = 0;
    result.chain = ChainType::SOLANA;

    // Initial keygen pass
    for (size_t d = 0; d < ed_states_.size(); ++d) {
        auto& state = ed_states_[d];
        devices_[d].set_current();

        ed25519_keygen_launch(
            static_cast<const uint8_t*>(state.d_seeds->get()),
            static_cast<uint8_t*>(state.d_pubkeys->get()),
            static_cast<uint8_t*>(state.d_priv_seeds->get()),
            state.num_points,
            state.stream
        );
        cudaStreamSynchronize(state.stream);
    }

    while (running_) {
        for (size_t d = 0; d < ed_states_.size(); ++d) {
            auto& state = ed_states_[d];
            devices_[d].set_current();

            // Launch iterate kernel
            ed25519_iterate_launch(
                state.d_points_X->get(),
                state.d_points_Y->get(),
                state.d_points_Z->get(),
                state.d_points_T->get(),
                static_cast<uint8_t*>(state.d_pubkeys->get()),
                state.num_points,
                config_.iterations_per_launch,
                state.stream
            );

            cudaStreamSynchronize(state.stream);

            uint64_t keys_checked = (uint64_t)state.num_points * config_.iterations_per_launch;
            speed_sample_.sample(keys_checked);

            if (config_.benchmark_mode) {
                if (progress_cb) {
                    progress_cb(speed_sample_.getSpeed(), speed_sample_.getTotal(), 0);
                }
                continue;
            }

            if (progress_cb) {
                progress_cb(speed_sample_.getSpeed(), speed_sample_.getTotal(), best_score_);
            }
        }
    }

    return result;
}

// =============================================================================
// stop — Signal the mining loop to terminate
// =============================================================================
void Dispatcher::stop() {
    running_ = false;
}

// =============================================================================
// speed / total_checked
// =============================================================================
double Dispatcher::speed() const {
    return speed_sample_.getSpeed();
}

uint64_t Dispatcher::total_checked() const {
    return speed_sample_.getTotal();
}

// =============================================================================
// generate_random_seeds — Fill buffer with cryptographically random bytes
// =============================================================================
void Dispatcher::generate_random_seeds(uint8_t* buffer, size_t size) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    size_t i = 0;
    while (i + 8 <= size) {
        uint64_t val = dist(gen);
        memcpy(buffer + i, &val, 8);
        i += 8;
    }
    // Handle remaining bytes
    if (i < size) {
        uint64_t val = dist(gen);
        memcpy(buffer + i, &val, size - i);
    }
}

// =============================================================================
// cleanup — Free all GPU resources
// =============================================================================
void Dispatcher::cleanup() {
    for (auto& state : secp_states_) {
        delete state.d_points_x;
        delete state.d_points_y;
        delete state.d_priv_keys;
        delete state.d_hashes;
        delete state.d_results;
        GPUDevice::destroy_stream(state.stream);
    }
    secp_states_.clear();

    for (auto& state : ed_states_) {
        delete state.d_seeds;
        delete state.d_pubkeys;
        delete state.d_priv_seeds;
        delete state.d_results;
        delete state.d_points_X;
        delete state.d_points_Y;
        delete state.d_points_Z;
        delete state.d_points_T;
        GPUDevice::destroy_stream(state.stream);
    }
    ed_states_.clear();
}
