// =============================================================================
// test_mp_uint256.cu — Unit tests for 256-bit multiprecision arithmetic
// =============================================================================

#include <gtest/gtest.h>
#include "common/mp_uint256.cuh"
#include <cstdint>
#include <cstring>

// Helper: run a device function via a single-thread kernel and copy result back

// ---- Kernel wrappers ----

__global__ void kernel_mp_add(mp_number* r, const mp_number* a, const mp_number* b, uint32_t* carry) {
    *carry = mp_add(*r, *a, *b);
}

__global__ void kernel_mp_sub(mp_number* r, const mp_number* a, const mp_number* b, uint32_t* borrow) {
    *borrow = mp_sub(*r, *a, *b);
}

__global__ void kernel_mp_mod_add(mp_number* r, const mp_number* a, const mp_number* b) {
    mp_mod_add(*r, *a, *b);
}

__global__ void kernel_mp_mod_sub(mp_number* r, const mp_number* a, const mp_number* b) {
    mp_mod_sub(*r, *a, *b);
}

__global__ void kernel_mp_mod_mul(mp_number* r, const mp_number* a, const mp_number* b) {
    mp_mod_mul(*r, *a, *b);
}

__global__ void kernel_mp_mod_sqr(mp_number* r, const mp_number* a) {
    mp_mod_sqr(*r, *a);
}

__global__ void kernel_mp_is_zero(const mp_number* a, int* result) {
    *result = mp_is_zero(*a);
}

__global__ void kernel_mp_cmp(const mp_number* a, const mp_number* b, int* result) {
    *result = mp_cmp(*a, *b);
}

__global__ void kernel_mp_set_ui(mp_number* r, uint32_t val) {
    mp_set_ui(*r, val);
}

// ---- Helper to allocate, run kernel, and read back ----

struct MPTestHelper {
    mp_number *d_a, *d_b, *d_r;
    uint32_t *d_carry;
    int *d_int_result;

    MPTestHelper() {
        cudaMalloc(&d_a, sizeof(mp_number));
        cudaMalloc(&d_b, sizeof(mp_number));
        cudaMalloc(&d_r, sizeof(mp_number));
        cudaMalloc(&d_carry, sizeof(uint32_t));
        cudaMalloc(&d_int_result, sizeof(int));
    }

    ~MPTestHelper() {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_r);
        cudaFree(d_carry);
        cudaFree(d_int_result);
    }

    void upload(const mp_number& a, const mp_number& b) {
        cudaMemcpy(d_a, &a, sizeof(mp_number), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, &b, sizeof(mp_number), cudaMemcpyHostToDevice);
    }

    void upload_a(const mp_number& a) {
        cudaMemcpy(d_a, &a, sizeof(mp_number), cudaMemcpyHostToDevice);
    }

    mp_number download_r() {
        mp_number r;
        cudaMemcpy(&r, d_r, sizeof(mp_number), cudaMemcpyDeviceToHost);
        return r;
    }

    uint32_t download_carry() {
        uint32_t c;
        cudaMemcpy(&c, d_carry, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return c;
    }

    int download_int() {
        int v;
        cudaMemcpy(&v, d_int_result, sizeof(int), cudaMemcpyDeviceToHost);
        return v;
    }
};

// Helper: create mp_number from uint32_t value
static mp_number mp_from_u32(uint32_t val) {
    mp_number n = {};
    n.d[0] = val;
    return n;
}

// Helper: check if mp_number equals uint32_t
static bool mp_equals_u32(const mp_number& n, uint32_t val) {
    if (n.d[0] != val) return false;
    for (int i = 1; i < MP_WORDS; ++i) {
        if (n.d[i] != 0) return false;
    }
    return true;
}

// ---- Tests ----

TEST(MpUint256, SetUI) {
    MPTestHelper h;
    kernel_mp_set_ui<<<1,1>>>(h.d_r, 42);
    cudaDeviceSynchronize();
    mp_number r = h.download_r();
    EXPECT_TRUE(mp_equals_u32(r, 42));
}

TEST(MpUint256, IsZero) {
    MPTestHelper h;
    mp_number zero = {};
    h.upload_a(zero);
    kernel_mp_is_zero<<<1,1>>>(h.d_a, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);

    mp_number one = mp_from_u32(1);
    h.upload_a(one);
    kernel_mp_is_zero<<<1,1>>>(h.d_a, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 0);
}

TEST(MpUint256, AddSimple) {
    MPTestHelper h;
    mp_number a = mp_from_u32(100);
    mp_number b = mp_from_u32(200);
    h.upload(a, b);
    kernel_mp_add<<<1,1>>>(h.d_r, h.d_a, h.d_b, h.d_carry);
    cudaDeviceSynchronize();
    mp_number r = h.download_r();
    EXPECT_TRUE(mp_equals_u32(r, 300));
    EXPECT_EQ(h.download_carry(), 0u);
}

TEST(MpUint256, SubSimple) {
    MPTestHelper h;
    mp_number a = mp_from_u32(300);
    mp_number b = mp_from_u32(100);
    h.upload(a, b);
    kernel_mp_sub<<<1,1>>>(h.d_r, h.d_a, h.d_b, h.d_carry);
    cudaDeviceSynchronize();
    mp_number r = h.download_r();
    EXPECT_TRUE(mp_equals_u32(r, 200));
    EXPECT_EQ(h.download_carry(), 0u);
}

TEST(MpUint256, SubBorrow) {
    MPTestHelper h;
    mp_number a = mp_from_u32(100);
    mp_number b = mp_from_u32(200);
    h.upload(a, b);
    kernel_mp_sub<<<1,1>>>(h.d_r, h.d_a, h.d_b, h.d_carry);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_carry(), 1u); // borrow occurred
}

TEST(MpUint256, CmpEqual) {
    MPTestHelper h;
    mp_number a = mp_from_u32(42);
    h.upload(a, a);
    kernel_mp_cmp<<<1,1>>>(h.d_a, h.d_b, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 0);
}

TEST(MpUint256, CmpLess) {
    MPTestHelper h;
    mp_number a = mp_from_u32(10);
    mp_number b = mp_from_u32(20);
    h.upload(a, b);
    kernel_mp_cmp<<<1,1>>>(h.d_a, h.d_b, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), -1);
}

TEST(MpUint256, CmpGreater) {
    MPTestHelper h;
    mp_number a = mp_from_u32(20);
    mp_number b = mp_from_u32(10);
    h.upload(a, b);
    kernel_mp_cmp<<<1,1>>>(h.d_a, h.d_b, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}

TEST(MpUint256, ModAddSmall) {
    MPTestHelper h;
    mp_number a = mp_from_u32(7);
    mp_number b = mp_from_u32(11);
    h.upload(a, b);
    kernel_mp_mod_add<<<1,1>>>(h.d_r, h.d_a, h.d_b);
    cudaDeviceSynchronize();
    mp_number r = h.download_r();
    EXPECT_TRUE(mp_equals_u32(r, 18));
}

TEST(MpUint256, ModSubSmall) {
    MPTestHelper h;
    mp_number a = mp_from_u32(20);
    mp_number b = mp_from_u32(7);
    h.upload(a, b);
    kernel_mp_mod_sub<<<1,1>>>(h.d_r, h.d_a, h.d_b);
    cudaDeviceSynchronize();
    mp_number r = h.download_r();
    EXPECT_TRUE(mp_equals_u32(r, 13));
}

TEST(MpUint256, ModMulSmall) {
    MPTestHelper h;
    mp_number a = mp_from_u32(6);
    mp_number b = mp_from_u32(7);
    h.upload(a, b);
    kernel_mp_mod_mul<<<1,1>>>(h.d_r, h.d_a, h.d_b);
    cudaDeviceSynchronize();
    mp_number r = h.download_r();
    EXPECT_TRUE(mp_equals_u32(r, 42));
}

TEST(MpUint256, ModSqrSmall) {
    MPTestHelper h;
    mp_number a = mp_from_u32(5);
    h.upload_a(a);
    kernel_mp_mod_sqr<<<1,1>>>(h.d_r, h.d_a);
    cudaDeviceSynchronize();
    mp_number r = h.download_r();
    EXPECT_TRUE(mp_equals_u32(r, 25));
}

TEST(MpUint256, AddWithCarryPropagation) {
    MPTestHelper h;
    mp_number a = {};
    a.d[0] = 0xFFFFFFFF;
    mp_number b = mp_from_u32(1);
    h.upload(a, b);
    kernel_mp_add<<<1,1>>>(h.d_r, h.d_a, h.d_b, h.d_carry);
    cudaDeviceSynchronize();
    mp_number r = h.download_r();
    EXPECT_EQ(r.d[0], 0u);
    EXPECT_EQ(r.d[1], 1u);
    EXPECT_EQ(h.download_carry(), 0u);
}

TEST(MpUint256, MulIdentity) {
    MPTestHelper h;
    mp_number a = mp_from_u32(12345);
    mp_number one = mp_from_u32(1);
    h.upload(a, one);
    kernel_mp_mod_mul<<<1,1>>>(h.d_r, h.d_a, h.d_b);
    cudaDeviceSynchronize();
    mp_number r = h.download_r();
    EXPECT_TRUE(mp_equals_u32(r, 12345));
}

TEST(MpUint256, MulByZero) {
    MPTestHelper h;
    mp_number a = mp_from_u32(12345);
    mp_number zero = {};
    h.upload(a, zero);
    kernel_mp_mod_mul<<<1,1>>>(h.d_r, h.d_a, h.d_b);
    cudaDeviceSynchronize();
    mp_number r = h.download_r();
    EXPECT_TRUE(mp_equals_u32(r, 0));
}
