# HYPERFANITY — Task Tracker

> Each task is an independent unit of work assignable to an agent.
> Tasks are executed in waves: all tasks in a wave can run in parallel.
> A wave cannot start until all tasks in the previous wave are complete.

## Status Key
- [ ] Not started
- [~] In progress
- [x] Complete
- [!] Blocked / needs review

---

## WAVE 1 — Foundation (no dependencies)
> These tasks have zero inter-dependencies and can all run in parallel.

### T01: CMake Build System
- **Role**: Build Engineer
- **Output**: `CMakeLists.txt` with CUDA language support, sm_60+ targets, separate kernel/host libs, test targets
- **Deps**: none
- [x] Status

### T02: Core Types & Shared Headers
- **Role**: Systems Architect
- **Output**: `src/types.hpp` (mp_number, point, result, seed256), `src/hex_utils.hpp`
- **Deps**: none
- [x] Status

### T03: 256-bit Multiprecision Arithmetic (CUDA)
- **Role**: GPU Math Engineer
- **Output**: `kernels/common/mp_uint256.cuh` — add, sub, mul, mod_sub, mod_mul, mod_inverse for secp256k1 field (mod p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F)
- **Reference**: `profanity2/profanity.cl` lines 46–300
- **Deps**: none
- [x] Status

### T04: Keccak-256 GPU Implementation
- **Role**: Cryptography Engineer
- **Output**: `kernels/common/keccak.cuh` — SHA-3 Keccak-256 hash function for CUDA
- **Reference**: `profanity2/keccak.cl`
- **Deps**: none
- [x] Status

### T05: SHA-256 Host Implementation
- **Role**: Cryptography Engineer
- **Output**: `src/crypto/sha256.hpp`, `src/crypto/sha256.cpp` — for TRX Base58Check checksum (host-side)
- **Deps**: none
- [x] Status

### T06: SHA-512 Host + GPU Implementation
- **Role**: Cryptography Engineer
- **Output**: `src/crypto/sha512.hpp`, `src/crypto/sha512.cpp`, `kernels/ed25519/sha512.cuh` — for Ed25519 key derivation
- **Deps**: none
- [x] Status

### T07: Base58 / Base58Check Encoding
- **Role**: Encoding Specialist
- **Output**: `src/chain/base58.hpp`, `src/chain/base58.cpp` — encode/decode for both TRX (Base58Check) and Solana (plain Base58)
- **Deps**: none
- [x] Status

### T08: Ed25519 Field Arithmetic (fe25519)
- **Role**: GPU Math Engineer
- **Output**: `kernels/ed25519/fe25519.cuh` — field arithmetic mod 2^255-19, radix-2^51 (5 limbs), add/sub/mul/sq/invert
- **Deps**: none
- [x] Status

### T09: Secp256k1 Precomputed Table Generator
- **Role**: Tooling Engineer
- **Output**: `tools/gen_precomp_secp256k1.py` → generates `kernels/secp256k1/secp256k1_precomp.cuh` (8160 generator multiples)
- **Deps**: none
- [x] Status

### T10: Ed25519 Precomputed Table Generator
- **Role**: Tooling Engineer
- **Output**: `tools/gen_precomp_ed25519.py` → generates `kernels/ed25519/ed25519_precomp.cuh` (base point multiples in Niels form)
- **Deps**: none
- [x] Status

### T11: Speed Sample / Throughput Measurement
- **Role**: Systems Engineer
- **Output**: `src/speed_sample.hpp`, `src/speed_sample.cpp` — hashrate tracking, rolling average, MH/s formatting
- **Deps**: none
- [x] Status

### T12: CLI Argument Parser
- **Role**: Systems Engineer
- **Output**: `src/arg_parser.hpp` — parse --chain, --prefix, --suffix, --devices, --benchmark, --worker, --panel, --token
- **Deps**: none
- [x] Status

### T13: GPU Memory RAII Wrapper
- **Role**: Systems Engineer
- **Output**: `src/dispatch/gpu_memory.hpp` — template wrapper for cudaMalloc/cudaFree/cudaMemcpyAsync with pinned host memory
- **Deps**: none
- [x] Status

### T14: gRPC Protocol Definition
- **Role**: Protocol Engineer
- **Output**: `proto/hyperfanity.proto` — service defs for Register, GetJob, ReportProgress, ReportResult
- **Deps**: none
- [x] Status

### T15: Address Verification Script
- **Role**: Tooling Engineer
- **Output**: `tools/verify_address.py` — given private key + chain, derive and verify address (uses ecdsa, pynacl, base58 Python libs)
- **Deps**: none
- [x] Status

---

## WAVE 2 — Curve Operations + Chain Encoding (depends on Wave 1)

### T16: Secp256k1 Point Operations
- **Role**: GPU Math Engineer
- **Output**: `kernels/secp256k1/secp256k1_ops.cuh` — point_add, point_double, specialized constant subs (sub_gx, sub_gy)
- **Deps**: T03 (mp_uint256)
- [ ] Status

### T17: Ed25519 Group Operations (ge25519)
- **Role**: GPU Math Engineer
- **Output**: `kernels/ed25519/ge25519.cuh` — extended coords point add/double, Niels-form mixed add, point compression
- **Deps**: T08 (fe25519)
- [ ] Status

### T18: TRX Address Encoding
- **Role**: Chain Specialist
- **Output**: `src/chain/tron.hpp`, `src/chain/tron.cpp`, `src/chain/chain.hpp` — fromHash(20 bytes) → T... address
- **Deps**: T05 (sha256), T07 (base58)
- [ ] Status

### T19: ETH Address Encoding
- **Role**: Chain Specialist
- **Output**: `src/chain/ethereum.hpp`, `src/chain/ethereum.cpp` — fromHash(20 bytes) → 0x... address with EIP-55 checksum
- **Deps**: T07 (base58 — for chain.hpp shared interface)
- [ ] Status

### T20: Solana Address Encoding
- **Role**: Chain Specialist
- **Output**: `src/chain/solana.hpp`, `src/chain/solana.cpp` — fromPublicKey(32 bytes) → Base58 address
- **Deps**: T07 (base58)
- [ ] Status

### T21: Scoring Mode Definitions
- **Role**: Systems Architect
- **Output**: `src/scoring/mode.hpp`, `src/scoring/mode.cpp`, `src/scoring/scorer.hpp` — prefix, suffix, matching, benchmark modes
- **Deps**: T02 (types)
- [ ] Status

---

## WAVE 3 — GPU Kernels + Dispatcher (depends on Wave 2)

### T22: Secp256k1 Init Kernel
- **Role**: GPU Kernel Engineer
- **Output**: `kernels/secp256k1/secp256k1_init.cu` — initialize points from random seeds using precomputed table
- **Deps**: T16 (secp256k1_ops), T09 (precomp table)
- [ ] Status

### T23: Secp256k1 Batch Inverse Kernel
- **Role**: GPU Kernel Engineer
- **Output**: `kernels/secp256k1/secp256k1_inverse.cu` — Algorithm 2.11 batch modular inverse
- **Deps**: T16 (secp256k1_ops)
- [ ] Status

### T24: Secp256k1 Iterate Kernel
- **Role**: GPU Kernel Engineer
- **Output**: `kernels/secp256k1/secp256k1_iterate.cu` — iterative point addition + Keccak-256 hash, store 20-byte address hash
- **Deps**: T16 (secp256k1_ops), T04 (keccak)
- [ ] Status

### T25: Ed25519 Keygen Kernel
- **Role**: GPU Kernel Engineer
- **Output**: `kernels/ed25519/ed25519_keygen.cu` — full keypair: random seed → SHA-512 → clamp → scalar × B → compressed pubkey
- **Deps**: T17 (ge25519), T06 (sha512 GPU), T10 (ed25519 precomp)
- [ ] Status

### T26: Ed25519 Batch Inverse Kernel
- **Role**: GPU Kernel Engineer
- **Output**: `kernels/ed25519/ed25519_inverse.cu` — Algorithm 2.11 over fe25519 field
- **Deps**: T17 (ge25519)
- [ ] Status

### T27: Ed25519 Iterate Kernel
- **Role**: GPU Kernel Engineer
- **Output**: `kernels/ed25519/ed25519_iterate.cu` — iterative base-point addition, store 32-byte compressed pubkey
- **Deps**: T17 (ge25519), T10 (ed25519 precomp)
- [ ] Status

### T28: Scoring Kernels
- **Role**: GPU Kernel Engineer
- **Output**: `kernels/scoring/score_prefix.cu`, `score_matching.cu`, `score_benchmark.cu`
- **Deps**: T02 (types), T21 (mode definitions)
- [ ] Status

### T29: CUDA Dispatcher
- **Role**: Systems Architect
- **Output**: `src/dispatch/dispatcher.hpp`, `src/dispatch/dispatcher.cpp`, `src/dispatch/gpu_device.hpp`, `src/dispatch/gpu_device.cpp`
- **Deps**: T13 (gpu_memory), T02 (types), T11 (speed_sample), T21 (scoring modes)
- [ ] Status

---

## WAVE 4 — Integration: Standalone CLI (depends on Wave 3)

### T30: Main Entry Point (Standalone CLI)
- **Role**: Integration Engineer
- **Output**: `src/main.cpp` — parse args, select chain, init dispatcher, run, print results
- **Deps**: T12 (arg_parser), T29 (dispatcher), T18 (tron), T19 (eth), T20 (solana), T22-T28 (all kernels)
- [ ] Status

### T31: Unit Tests — Crypto Primitives
- **Role**: Test Engineer
- **Output**: `tests/test_mp_uint256.cu`, `tests/test_keccak.cu`, `tests/test_base58.cpp`
- **Deps**: T03 (mp_uint256), T04 (keccak), T07 (base58), T01 (cmake)
- [ ] Status

### T32: Unit Tests — Curve Operations
- **Role**: Test Engineer
- **Output**: `tests/test_secp256k1.cu`, `tests/test_ed25519.cu`
- **Deps**: T16 (secp256k1_ops), T17 (ge25519), T01 (cmake)
- [ ] Status

### T33: Integration Tests — Address Derivation
- **Role**: Test Engineer
- **Output**: `tests/test_tron_address.cpp`, `tests/test_solana_address.cpp`, `tests/test_ethereum_address.cpp`
- **Deps**: T18 (tron), T19 (eth), T20 (solana), T01 (cmake)
- [ ] Status

---

## WAVE 5 — Panel + Worker + Docker (depends on Wave 4)

### T34: Panel Service (Go)
- **Role**: Backend Engineer
- **Output**: `panel/main.go`, `panel/api.go`, `panel/job_manager.go`, `panel/go.mod` — gRPC server, job queue, worker registry, web UI
- **Deps**: T14 (proto), T30 (working CLI to validate protocol)
- [ ] Status

### T35: Worker Mode (C++)
- **Role**: Integration Engineer
- **Output**: `src/worker.hpp`, `src/worker.cpp` — connects to panel via gRPC, pulls jobs, streams progress, reports results
- **Deps**: T14 (proto), T29 (dispatcher), T30 (main.cpp for integration)
- [ ] Status

### T36: Docker — Worker Image
- **Role**: DevOps Engineer
- **Output**: `Dockerfile.worker` — multi-stage build, nvidia/cuda base, builds hyperfanity binary
- **Deps**: T01 (cmake), T30 (main builds)
- [ ] Status

### T37: Docker — Panel Image
- **Role**: DevOps Engineer
- **Output**: `Dockerfile.panel` — multi-stage Go build, alpine runtime
- **Deps**: T34 (panel service)
- [ ] Status

### T38: Docker Compose + Deploy Scripts
- **Role**: DevOps Engineer
- **Output**: `docker-compose.yml`, `scripts/deploy-worker.sh`, `scripts/deploy-runpod.sh`
- **Deps**: T36 (worker dockerfile), T37 (panel dockerfile)
- [ ] Status

---

## Dependency Graph (Waves)

```
WAVE 1 (15 tasks, all parallel)
  T01 T02 T03 T04 T05 T06 T07 T08 T09 T10 T11 T12 T13 T14 T15

WAVE 2 (6 tasks, all parallel)
  T16←T03  T17←T08  T18←T05,T07  T19←T07  T20←T07  T21←T02

WAVE 3 (8 tasks, all parallel)
  T22←T16,T09  T23←T16  T24←T16,T04  T25←T17,T06,T10
  T26←T17  T27←T17,T10  T28←T02,T21  T29←T13,T02,T11,T21

WAVE 4 (4 tasks, all parallel)
  T30←T12,T29,T18-T20,T22-T28  T31←T03,T04,T07,T01
  T32←T16,T17,T01  T33←T18,T19,T20,T01

WAVE 5 (5 tasks, partial parallel)
  T34←T14,T30  T35←T14,T29,T30
  T36←T01,T30  T37←T34  T38←T36,T37
```
