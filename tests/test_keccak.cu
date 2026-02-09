// =============================================================================
// test_keccak.cu — Unit tests for Keccak-256 (Ethereum variant)
// =============================================================================

#include <gtest/gtest.h>
#include "common/keccak.cuh"
#include <cstdint>
#include <cstring>

// ---- Kernel wrappers ----

__global__ void kernel_keccak256(ethash_hash* out, const uint8_t* input, size_t len) {
    keccak256(*out, input, len);
}

__global__ void kernel_keccak256_64(ethash_hash* out, const uint8_t* input) {
    keccak256_64(*out, input);
}

// ---- Test helper ----
struct KeccakTestHelper {
    ethash_hash* d_hash;
    uint8_t* d_input;

    KeccakTestHelper(size_t input_size) {
        cudaMalloc(&d_hash, sizeof(ethash_hash));
        cudaMalloc(&d_input, input_size);
    }

    ~KeccakTestHelper() {
        cudaFree(d_hash);
        cudaFree(d_input);
    }

    void upload_input(const uint8_t* data, size_t len) {
        cudaMemcpy(d_input, data, len, cudaMemcpyHostToDevice);
    }

    ethash_hash download_hash() {
        ethash_hash h;
        cudaMemcpy(&h, d_hash, sizeof(ethash_hash), cudaMemcpyDeviceToHost);
        return h;
    }
};

static std::string hash_to_hex(const ethash_hash& h) {
    static const char hex[] = "0123456789abcdef";
    std::string s;
    s.reserve(64);
    for (int i = 0; i < 32; ++i) {
        s.push_back(hex[(h.bytes[i] >> 4) & 0x0F]);
        s.push_back(hex[h.bytes[i] & 0x0F]);
    }
    return s;
}

// ---- Tests ----

// Test: Keccak-256 of empty string
// Known: Keccak-256("") = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
TEST(Keccak256, EmptyString) {
    KeccakTestHelper h(1);
    uint8_t empty = 0;
    h.upload_input(&empty, 0);
    kernel_keccak256<<<1,1>>>(h.d_hash, h.d_input, 0);
    cudaDeviceSynchronize();
    ethash_hash result = h.download_hash();
    std::string hex = hash_to_hex(result);
    EXPECT_EQ(hex, "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");
}

// Test: Keccak-256 of "abc"
// Known: Keccak-256("abc") = 4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45
TEST(Keccak256, Abc) {
    KeccakTestHelper h(3);
    uint8_t input[] = {'a', 'b', 'c'};
    h.upload_input(input, 3);
    kernel_keccak256<<<1,1>>>(h.d_hash, h.d_input, 3);
    cudaDeviceSynchronize();
    ethash_hash result = h.download_hash();
    std::string hex = hash_to_hex(result);
    EXPECT_EQ(hex, "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");
}

// Test: keccak256_64 with 64 zero bytes
TEST(Keccak256, SixtyFourZeros) {
    KeccakTestHelper h(64);
    uint8_t input[64] = {0};
    h.upload_input(input, 64);
    kernel_keccak256_64<<<1,1>>>(h.d_hash, h.d_input);
    cudaDeviceSynchronize();
    ethash_hash result64 = h.download_hash();

    // Compare with general keccak256
    kernel_keccak256<<<1,1>>>(h.d_hash, h.d_input, 64);
    cudaDeviceSynchronize();
    ethash_hash result_gen = h.download_hash();

    // Both should produce the same hash
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(result64.bytes[i], result_gen.bytes[i])
            << "Mismatch at byte " << i;
    }
}

// Test: keccak256_64 consistency with general keccak256 for random-ish data
TEST(Keccak256, SixtyFourBytesConsistency) {
    KeccakTestHelper h(64);
    uint8_t input[64];
    for (int i = 0; i < 64; ++i) {
        input[i] = (uint8_t)(i * 7 + 13);
    }
    h.upload_input(input, 64);

    kernel_keccak256_64<<<1,1>>>(h.d_hash, h.d_input);
    cudaDeviceSynchronize();
    ethash_hash result64 = h.download_hash();

    kernel_keccak256<<<1,1>>>(h.d_hash, h.d_input, 64);
    cudaDeviceSynchronize();
    ethash_hash result_gen = h.download_hash();

    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(result64.bytes[i], result_gen.bytes[i])
            << "Mismatch at byte " << i;
    }
}

// Test: Keccak-256 is NOT SHA-3 (different padding)
// SHA-3-256("") = a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a
// Keccak-256("") = c5d2460186f7233c... (different!)
TEST(Keccak256, NotSha3) {
    KeccakTestHelper h(1);
    uint8_t empty = 0;
    h.upload_input(&empty, 0);
    kernel_keccak256<<<1,1>>>(h.d_hash, h.d_input, 0);
    cudaDeviceSynchronize();
    ethash_hash result = h.download_hash();
    std::string hex = hash_to_hex(result);
    // Must NOT equal SHA-3-256 of empty string
    EXPECT_NE(hex, "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
}
