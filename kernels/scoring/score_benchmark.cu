// =============================================================================
// score_benchmark.cu — Benchmark scoring kernel (no-op scoring)
// =============================================================================
//
// In benchmark mode, no pattern matching is performed. The kernel simply
// counts iterations for throughput measurement. This allows measuring
// raw key generation speed without scoring overhead.
//
// The kernel is a no-op that returns immediately — the actual benchmark
// throughput is measured by the dispatcher based on the iterate kernel's
// execution time.
// =============================================================================

#include <cstdint>

// =============================================================================
// Kernel: score_benchmark
//
// Does nothing — benchmark mode skips scoring entirely.
// The dispatcher measures hashrate from the iterate kernel alone.
// =============================================================================
__global__ void score_benchmark(
    const uint8_t* __restrict__ data,
    uint32_t num_entries,
    uint32_t entry_size
) {
    // Intentionally empty — benchmark mode doesn't score
    // This kernel exists so the dispatcher has a uniform interface
}

// =============================================================================
// Host-callable wrapper
// =============================================================================
extern "C" void score_benchmark_launch(
    const uint8_t* d_data,
    uint32_t num_entries,
    uint32_t entry_size,
    cudaStream_t stream
) {
    // In benchmark mode, we don't actually launch the scoring kernel.
    // The dispatcher skips scoring and just measures iterate throughput.
    // This function is provided for interface completeness.

    // No-op: don't launch kernel to save GPU cycles in benchmark mode
    (void)d_data;
    (void)num_entries;
    (void)entry_size;
    (void)stream;
}
