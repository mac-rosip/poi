// =============================================================================
// test_secp256k1.cu — Unit tests for secp256k1 point operations
// =============================================================================

#include <gtest/gtest.h>
#include "secp256k1/secp256k1_ops.cuh"
#include <cstdint>

// ---- Kernel wrappers ----

__global__ void kernel_point_double(secp256k1_point* r, const secp256k1_point* p) {
    point_double(*r, *p);
}

__global__ void kernel_point_add(secp256k1_point* r, const secp256k1_point* p, const secp256k1_point* q) {
    point_add(*r, *p, *q);
}

__global__ void kernel_point_negate(secp256k1_point* r, const secp256k1_point* p) {
    point_negate(*r, *p);
}

__global__ void kernel_point_equal(const secp256k1_point* p, const secp256k1_point* q, int* result) {
    *result = point_equal(*p, *q);
}

__global__ void kernel_point_is_infinity(const secp256k1_point* p, int* result) {
    *result = point_is_infinity(*p);
}

__global__ void kernel_make_G(secp256k1_point* p) {
    mp_copy(p->x, SECP256K1_GX);
    mp_copy(p->y, SECP256K1_GY);
}

__global__ void kernel_point_scalar_mul(secp256k1_point* r, const mp_number* k, const secp256k1_point* p) {
    point_scalar_mul(*r, *k, *p);
}

// ---- Test helper ----

struct PointTestHelper {
    secp256k1_point *d_p, *d_q, *d_r;
    int *d_int_result;
    mp_number *d_scalar;

    PointTestHelper() {
        cudaMalloc(&d_p, sizeof(secp256k1_point));
        cudaMalloc(&d_q, sizeof(secp256k1_point));
        cudaMalloc(&d_r, sizeof(secp256k1_point));
        cudaMalloc(&d_int_result, sizeof(int));
        cudaMalloc(&d_scalar, sizeof(mp_number));
    }

    ~PointTestHelper() {
        cudaFree(d_p);
        cudaFree(d_q);
        cudaFree(d_r);
        cudaFree(d_int_result);
        cudaFree(d_scalar);
    }

    secp256k1_point download_r() {
        secp256k1_point r;
        cudaMemcpy(&r, d_r, sizeof(secp256k1_point), cudaMemcpyDeviceToHost);
        return r;
    }

    int download_int() {
        int v;
        cudaMemcpy(&v, d_int_result, sizeof(int), cudaMemcpyDeviceToHost);
        return v;
    }

    void make_G_on_device(secp256k1_point* dst) {
        kernel_make_G<<<1,1>>>(dst);
        cudaDeviceSynchronize();
    }
};

// ---- Tests ----

// Test: infinity point is detected
TEST(Secp256k1, InfinityCheck) {
    PointTestHelper h;
    secp256k1_point inf = {};
    cudaMemcpy(h.d_p, &inf, sizeof(secp256k1_point), cudaMemcpyHostToDevice);
    kernel_point_is_infinity<<<1,1>>>(h.d_p, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}

// Test: G is not infinity
TEST(Secp256k1, GNotInfinity) {
    PointTestHelper h;
    h.make_G_on_device(h.d_p);
    kernel_point_is_infinity<<<1,1>>>(h.d_p, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 0);
}

// Test: G equals G
TEST(Secp256k1, GEqualsG) {
    PointTestHelper h;
    h.make_G_on_device(h.d_p);
    h.make_G_on_device(h.d_q);
    kernel_point_equal<<<1,1>>>(h.d_p, h.d_q, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}

// Test: 2G = G + G (doubling matches addition)
TEST(Secp256k1, DoubleEqualsAddSelf) {
    PointTestHelper h;
    h.make_G_on_device(h.d_p);

    // Compute 2G via doubling
    kernel_point_double<<<1,1>>>(h.d_r, h.d_p);
    cudaDeviceSynchronize();
    secp256k1_point doubled = h.download_r();

    // Compute G + G via addition
    h.make_G_on_device(h.d_q);
    kernel_point_add<<<1,1>>>(h.d_r, h.d_p, h.d_q);
    cudaDeviceSynchronize();
    secp256k1_point added = h.download_r();

    // They should be equal
    cudaMemcpy(h.d_p, &doubled, sizeof(secp256k1_point), cudaMemcpyHostToDevice);
    cudaMemcpy(h.d_q, &added, sizeof(secp256k1_point), cudaMemcpyHostToDevice);
    kernel_point_equal<<<1,1>>>(h.d_p, h.d_q, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}

// Test: G + (-G) = infinity
TEST(Secp256k1, AddNegateIsInfinity) {
    PointTestHelper h;
    h.make_G_on_device(h.d_p);

    // Negate G
    kernel_point_negate<<<1,1>>>(h.d_q, h.d_p);
    cudaDeviceSynchronize();

    // G + (-G) should be infinity
    kernel_point_add<<<1,1>>>(h.d_r, h.d_p, h.d_q);
    cudaDeviceSynchronize();
    kernel_point_is_infinity<<<1,1>>>(h.d_r, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}

// Test: G + infinity = G
TEST(Secp256k1, AddInfinityIsIdentity) {
    PointTestHelper h;
    h.make_G_on_device(h.d_p);

    // Set q = infinity
    secp256k1_point inf = {};
    cudaMemcpy(h.d_q, &inf, sizeof(secp256k1_point), cudaMemcpyHostToDevice);

    kernel_point_add<<<1,1>>>(h.d_r, h.d_p, h.d_q);
    cudaDeviceSynchronize();

    // R should equal G
    kernel_point_equal<<<1,1>>>(h.d_r, h.d_p, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}

// Test: 3G = 2G + G
TEST(Secp256k1, TripleG) {
    PointTestHelper h;
    h.make_G_on_device(h.d_p);

    // 2G
    kernel_point_double<<<1,1>>>(h.d_q, h.d_p);
    cudaDeviceSynchronize();

    // 3G = 2G + G
    kernel_point_add<<<1,1>>>(h.d_r, h.d_q, h.d_p);
    cudaDeviceSynchronize();
    secp256k1_point three_G = h.download_r();

    // Also compute via scalar mul: 3 * G
    mp_number three = {};
    three.d[0] = 3;
    cudaMemcpy(h.d_scalar, &three, sizeof(mp_number), cudaMemcpyHostToDevice);
    h.make_G_on_device(h.d_p);
    kernel_point_scalar_mul<<<1,1>>>(h.d_r, h.d_scalar, h.d_p);
    cudaDeviceSynchronize();
    secp256k1_point scalar_3G = h.download_r();

    // Compare
    cudaMemcpy(h.d_p, &three_G, sizeof(secp256k1_point), cudaMemcpyHostToDevice);
    cudaMemcpy(h.d_q, &scalar_3G, sizeof(secp256k1_point), cudaMemcpyHostToDevice);
    kernel_point_equal<<<1,1>>>(h.d_p, h.d_q, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}

// Test: scalar multiplication by 1 gives G
TEST(Secp256k1, ScalarMulByOne) {
    PointTestHelper h;
    h.make_G_on_device(h.d_p);

    mp_number one = {};
    one.d[0] = 1;
    cudaMemcpy(h.d_scalar, &one, sizeof(mp_number), cudaMemcpyHostToDevice);

    kernel_point_scalar_mul<<<1,1>>>(h.d_r, h.d_scalar, h.d_p);
    cudaDeviceSynchronize();

    kernel_point_equal<<<1,1>>>(h.d_r, h.d_p, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}

// Test: scalar multiplication by 2 gives 2G
TEST(Secp256k1, ScalarMulByTwo) {
    PointTestHelper h;
    h.make_G_on_device(h.d_p);

    // 2G via doubling
    kernel_point_double<<<1,1>>>(h.d_q, h.d_p);
    cudaDeviceSynchronize();

    // 2 * G via scalar mul
    mp_number two = {};
    two.d[0] = 2;
    cudaMemcpy(h.d_scalar, &two, sizeof(mp_number), cudaMemcpyHostToDevice);
    h.make_G_on_device(h.d_p);
    kernel_point_scalar_mul<<<1,1>>>(h.d_r, h.d_scalar, h.d_p);
    cudaDeviceSynchronize();

    kernel_point_equal<<<1,1>>>(h.d_r, h.d_q, h.d_int_result);
    cudaDeviceSynchronize();
    EXPECT_EQ(h.download_int(), 1);
}
