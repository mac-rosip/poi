cmake_minimum_required(VERSION 3.18)
project(hyperfanity LANGUAGES CXX CUDA)

# ---- Language standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- CUDA architectures ----
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")

# ---- Dependencies ----
find_package(CUDAToolkit REQUIRED)

# ===========================================================================
# Library 1 : hyperfanity_kernels  (STATIC, CUDA)
# ===========================================================================
add_library(hyperfanity_kernels STATIC
    kernels/secp256k1/secp256k1_init.cu
    kernels/secp256k1/secp256k1_inverse.cu
    kernels/secp256k1/secp256k1_iterate.cu
    kernels/secp256k1/secp256k1_iterate_bitcoin.cu
    kernels/ed25519/ed25519_keygen.cu
    kernels/ed25519/ed25519_inverse.cu
    kernels/ed25519/ed25519_iterate.cu
    kernels/scoring/score_prefix.cu
    kernels/scoring/score_matching.cu
    kernels/scoring/score_benchmark.cu
)

target_include_directories(hyperfanity_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

target_compile_options(hyperfanity_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        --ptxas-options=-v
        -maxrregcount=64
    >
)

# ===========================================================================
# Library 2 : hyperfanity_lib  (STATIC, CXX)
# ===========================================================================
add_library(hyperfanity_lib STATIC
    src/speed_sample.cpp
    src/chain/tron.cpp
    src/chain/solana.cpp
    src/chain/ethereum.cpp
    src/chain/bitcoin.cpp
    src/chain/base58.cpp
    src/scoring/mode.cpp
    src/dispatch/dispatcher.cpp
    src/dispatch/gpu_device.cpp
    src/crypto/sha256.cpp
    src/crypto/sha512.cpp
)

target_include_directories(hyperfanity_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(hyperfanity_lib PUBLIC
    hyperfanity_kernels
    CUDA::cudart
)

# ===========================================================================
# Main executable
# ===========================================================================
add_executable(hyperfanity src/main.cpp)

target_link_libraries(hyperfanity PRIVATE
    hyperfanity_lib
)

# ===========================================================================
# Tests  (optional, default ON)
# ===========================================================================
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    find_package(GTest QUIET)

    if(GTest_FOUND)
        enable_testing()

        # -- CUDA tests (link kernels + GTest) --
        set(CUDA_TESTS
            test_mp_uint256:tests/test_mp_uint256.cu
            test_secp256k1:tests/test_secp256k1.cu
            test_ed25519:tests/test_ed25519.cu
            test_keccak:tests/test_keccak.cu
        )

        foreach(entry IN LISTS CUDA_TESTS)
            string(REPLACE ":" ";" pair ${entry})
            list(GET pair 0 test_name)
            list(GET pair 1 test_src)

            add_executable(${test_name} ${test_src})
            target_link_libraries(${test_name} PRIVATE
                hyperfanity_kernels
                GTest::gtest_main
            )
            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()

        # -- C++ tests (link lib + GTest) --
        set(CXX_TESTS
            test_base58:tests/test_base58.cpp
            test_tron_address:tests/test_tron_address.cpp
            test_solana_address:tests/test_solana_address.cpp
            test_ethereum_address:tests/test_ethereum_address.cpp
            test_bitcoin_address:tests/test_bitcoin_address.cpp
        )

        foreach(entry IN LISTS CXX_TESTS)
            string(REPLACE ":" ";" pair ${entry})
            list(GET pair 0 test_name)
            list(GET pair 1 test_src)

            add_executable(${test_name} ${test_src})
            target_link_libraries(${test_name} PRIVATE
                hyperfanity_lib
                GTest::gtest_main
            )
            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()
    else()
        message(STATUS "GTest not found -- skipping test targets")
    endif()
endif()
