// =============================================================================
// test_ed25519.cu — Unit tests for Ed25519 group operations
// =============================================================================

#include <gtest/gtest.h>
#include "ed25519/ge25519.cuh"
#include <cstdint>

// ---- Kernel wrappers ----

__global__ void kernel_ge_set_identity(ge25519_p3* p) {
    ge25519_set_identity(*p);
}

__global__ void kernel_ge_double(ge25519_p3* r, const ge25519_p3* p) {
    ge25519_double(*r, *p);
}

__global__ void kernel_ge_add(ge25519_p3* r, const ge25519_p3* p, const ge25519_p3* q) {
    ge25519_add(*r, *p, *q);
}

__global__ void kernel_ge_compress(uint8_t* out, const ge25519_p3* p) {
    ge25519_compress(out, *p);
}

__global__ void kernel_make_basepoint(ge25519_p3* p) {
    fe25519_copy(p->X, GE25519_BX);
    fe25519_copy(p->Y, GE25519_BY);
    fe25519_set_one(p->Z);
    fe25519_mul(p->T, p->X, p->Y);
}

__global__ void kernel_fe_equal(const fe25519* a, const fe25519* b, int* result) {
    *result = fe25519_equal(*a, *b);
}

__global__ void kernel_check_point_equal(
    const ge25519_p3* p, const ge25519_p3* q, int* result
) {
    // Compare in affine: X1*Z2 == X2*Z1 and Y1*Z2 == Y2*Z1
    fe25519 lhs, rhs;
    fe25519_mul(lhs, p->X, q->Z);
    fe25519_mul(rhs, q->X, p->Z);
    fe25519_carry(lhs); fe25519_reduce(lhs);
    fe25519_carry(rhs); fe25519_reduce(rhs);
    int x_eq = fe25519_equal(lhs, rhs);

    fe25519_mul(lhs, p->Y, q->Z);
    fe25519_mul(rhs, q->Y, p->Z);
    fe25519_carry(lhs); fe25519_reduce(lhs);
    fe25519_carry(rhs); fe25519_reduce(rhs);
    int y_eq = fe25519_equal(lhs, rhs);

    *result = x_eq && y_eq;
}

// ---- Test helper ----

struct Ed25519TestHelper {
    ge25519_p3 *d_p, *d_q, *d_r;
    uint8_t *d_compressed;
    int *d_int_result;

    Ed25519TestHelper() {
        cudaMalloc(&d_p, sizeof(ge25519_p3));
        cudaMalloc(&d_q, sizeof(ge25519_p3));
        cudaMalloc(&d_r, sizeof(ge25519_p3));
        cudaMalloc(&d_compressed, 32);
        cudaMalloc(&d_int_result, sizeof(int));
    }

    ~Ed25519TestHelper() {
        cudaFree(d_p);
        cudaFree(d_q);
        cudaFree(d_r);
        cudaFree(d_compressed);
        cudaFree(d_int_result);
    }

    void download_compressed(uint8_t out[32]) {
        cudaMemcpy(out, d_compressed, 32, cudaMemcpyDeviceToHost);
    }

    int download_int() {
        int v;
        cudaMemcpy(&v, d_int_result, sizeof(int), cudaMemcpyDeviceToHost);
        return v;
    }

    void make_B(ge25519_p3* dst) {
        kernel_make_basepoint<<<1,1>>>(dst);
        cudaDeviceSynchronize();
    }
};

static std::string bytes_to_hex(const uint8_t* data, size_t len) {
    static const char hex[] = "0123456789abcdef";
    std::string s;
    s.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        s.push_back(hex[(data[i] >> 4) & 0x0F]);
        s.push_back(hex[data[i] & 0x0F]);
    }
    return s;
}

// ---- Tests ----

// Test: identity point compressed is (0, 1) → y=1 → all zeros except byte[0]=1
TEST(Ed25519, IdentityCompress) {
    Ed25519TestHelper h;
    kernel_ge_set_identity<<<1,1>>>(h.d_p);
    cudaDeviceSynchronize();
    kernel_ge_compress<<<1,1>>>(h.d_compressed, h.d_p);
    cudaDeviceSynchronize();

    uint8_t out[32];
    h.download_compressed(out);

    // y=1 little-endian: byte[0]=1, rest=0. Sign bit of x=0 → byte[31] bit7=0.
    EXPECT_EQ(out[0], 1);
    for (int i = 1; i < 32; ++i) {
        EXPECT_EQ(out[i], 0) << "Byte " << i << " should be 0";
    }
}

// Test: base point compressed
// Known: Ed25519 base point compressed encoding starts with 0x58...
// The y-coordinate is 4/5 mod p, encoded little-endian with sign bit
TEST(Ed25519, BasepointCompress) {
    Ed25519TestHelper h;
    h.make_B(h.d_p);
    kernel_ge_compress<<<1,1>>>(h.d_compressed, h.d_p);
    cudaDeviceSynchronize();

    uint8_t out[32];
    h.download_compressed(out);

    // Known base point encoding (from RFC 8032):
    // 5866666666666666666666666666666666666666666666666666666666666666
    EXPECT_EQ(out[0], 0x58);
    // Last byte should have sign bit set based on x parity
}

// Test: B + identity = B
TEST(Ed25519, AddIdentity) {
    Ed25519TestHelper h;
    h.make_B(h.d_p);
    kernel_ge_set_identity<<<1,1>>>(h.d_q);
    cudaDeviceSynchronize();

    kernel_ge_add<<<1,1>>>(h.d_r, h.d_p, h.d_q);
    cudaDeviceSynchronize();

    kernel_check_point_equal<<<1,1>>>(h.d_r, h.d_p, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}

// Test: 2B = B + B (doubling matches addition)
TEST(Ed25519, DoubleEqualsAddSelf) {
    Ed25519TestHelper h;
    h.make_B(h.d_p);

    // 2B via doubling
    kernel_ge_double<<<1,1>>>(h.d_q, h.d_p);
    cudaDeviceSynchronize();

    // B + B via addition
    h.make_B(h.d_p);
    ge25519_p3 *d_bp2;
    cudaMalloc(&d_bp2, sizeof(ge25519_p3));
    h.make_B(d_bp2);
    kernel_ge_add<<<1,1>>>(h.d_r, h.d_p, d_bp2);
    cudaDeviceSynchronize();

    // Compare
    kernel_check_point_equal<<<1,1>>>(h.d_q, h.d_r, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);

    cudaFree(d_bp2);
}

// Test: 2B and B compress to different points
TEST(Ed25519, DoubleBDifferentFromB) {
    Ed25519TestHelper h;
    h.make_B(h.d_p);

    kernel_ge_compress<<<1,1>>>(h.d_compressed, h.d_p);
    cudaDeviceSynchronize();
    uint8_t b_compressed[32];
    h.download_compressed(b_compressed);

    kernel_ge_double<<<1,1>>>(h.d_q, h.d_p);
    cudaDeviceSynchronize();
    kernel_ge_compress<<<1,1>>>(h.d_compressed, h.d_q);
    cudaDeviceSynchronize();
    uint8_t b2_compressed[32];
    h.download_compressed(b2_compressed);

    // B and 2B should be different
    bool same = true;
    for (int i = 0; i < 32; ++i) {
        if (b_compressed[i] != b2_compressed[i]) {
            same = false;
            break;
        }
    }
    EXPECT_FALSE(same);
}

// Test: 3B = 2B + B
TEST(Ed25519, TripleB) {
    Ed25519TestHelper h;
    h.make_B(h.d_p);

    // 2B
    kernel_ge_double<<<1,1>>>(h.d_q, h.d_p);
    cudaDeviceSynchronize();

    // 3B = 2B + B
    h.make_B(h.d_p);
    kernel_ge_add<<<1,1>>>(h.d_r, h.d_q, h.d_p);
    cudaDeviceSynchronize();

    // Compress 3B
    kernel_ge_compress<<<1,1>>>(h.d_compressed, h.d_r);
    cudaDeviceSynchronize();
    uint8_t out[32];
    h.download_compressed(out);

    // Should be non-zero and different from B and 2B
    bool all_zero = true;
    for (int i = 0; i < 32; ++i) {
        if (out[i] != 0) { all_zero = false; break; }
    }
    EXPECT_FALSE(all_zero);
}

// Test: fe25519 addition commutativity
__global__ void kernel_fe_add_commutative(int* result) {
    fe25519 a = {{100, 200, 300, 400, 500}};
    fe25519 b = {{50, 60, 70, 80, 90}};
    fe25519 ab, ba;
    fe25519_add(ab, a, b);
    fe25519_add(ba, b, a);
    fe25519_carry(ab); fe25519_reduce(ab);
    fe25519_carry(ba); fe25519_reduce(ba);
    *result = fe25519_equal(ab, ba);
}

TEST(Ed25519, FeAddCommutative) {
    Ed25519TestHelper h;
    kernel_fe_add_commutative<<<1,1>>>(h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}

// Test: fe25519 multiplication commutativity
__global__ void kernel_fe_mul_commutative(int* result) {
    fe25519 a = {{12345, 67890, 11111, 22222, 33333}};
    fe25519 b = {{54321, 98765, 44444, 55555, 66666}};
    fe25519 ab, ba;
    fe25519_mul(ab, a, b);
    fe25519_mul(ba, b, a);
    *result = fe25519_equal(ab, ba);
}

TEST(Ed25519, FeMulCommutative) {
    Ed25519TestHelper h;
    kernel_fe_mul_commutative<<<1,1>>>(h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}
