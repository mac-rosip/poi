# HANDOFF: Hyperfanity Wave 1 File Writing

## What This Project Is

Hyperfanity is a CUDA vanity wallet address generator for TRX, Solana, and ETH. It uses a Panel + Worker architecture with Docker deployment. The full plan is at:
- `/Users/dddd/.claude/plans/crispy-kindling-kurzweil.md`
- `TASKS.md` in this repo (38 tasks across 5 waves)

## Current State

**Git repo**: `/Users/dddd/hype/hyperfanity/` (initialized, one commit)

### Files Already Written to Disk (8 files)
```
.gitignore
CMakeLists.txt                    # T01 - done
TASKS.md                          # Task tracker
proto/hyperfanity.proto           # T14 - done
src/types.hpp                     # T02 - done
src/hex_utils.hpp                 # T02 - done
src/speed_sample.hpp              # T11 - done
src/speed_sample.cpp              # T11 - done
```

### Files NOT Yet Written (agents completed, code in output files)

All 15 Wave 1 agents ran and produced correct code, but couldn't write files (background agents lack write permission). The code lives in agent output transcript files. Each transcript is JSONL; the code is embedded in assistant messages (usually in Write tool call parameters or code blocks).

**Agent output files** are at `/private/tmp/claude/-Users-dddd-hype/tasks/`:

| Task | File(s) to Create | Agent Output File |
|------|-------------------|-------------------|
| T03 | `kernels/common/mp_uint256.cuh` | `aba5843.output` |
| T04 | `kernels/common/keccak.cuh` | `a904127.output` |
| T05 | `src/crypto/sha256.hpp`, `src/crypto/sha256.cpp` | `ad5ae17.output` |
| T06 | `src/crypto/sha512.hpp`, `src/crypto/sha512.cpp`, `kernels/ed25519/sha512.cuh` | `a1cc3db.output` |
| T07 | `src/chain/base58.hpp`, `src/chain/base58.cpp` | `ab11d08.output` |
| T08 | `kernels/ed25519/fe25519.cuh` | `af850c0.output` |
| T09 | `tools/gen_precomp_secp256k1.py` | `a1452df.output` |
| T10 | `tools/gen_precomp_ed25519.py` | `aa6d2e2.output` |
| T12 | `src/arg_parser.hpp` | `a71806e.output` |
| T13 | `src/dispatch/gpu_memory.hpp` | `afcd08f.output` |
| T15 | `tools/verify_address.py` | `a84d92c.output` |

### Code Already Extracted (from previous extraction round)

The previous session extracted clean source code for these files before running out of context. The code was extracted but NOT written to disk:

1. **mp_uint256.cuh** - 256-bit multiprecision arithmetic for secp256k1 field. Functions: mp_sub, mp_add, mp_mod_sub, mp_mod_add, mp_mul_512 (schoolbook), mp_reduce_hi (secp256k1 identity: 2^256 = 2^32+977 mod p), mp_mod_mul, mp_mod_sqr, mp_mod_inverse (Fermat's little theorem addition chain), mp_is_zero, mp_cmp, mp_copy, mp_set_ui.

2. **keccak.cuh** - Keccak-f[1600] permutation + Keccak-256 (Ethereum variant, 0x01 padding NOT SHA-3's 0x06). Structs: ethash_hash union. Functions: sha3_keccakf, keccak256 (general), keccak256_64 (hot path for 64-byte pubkey), keccak256_64_q (uint64_t input variant).

3. **sha256.hpp/cpp** - Host-side SHA-256 (FIPS 180-4). Namespace: `crypto`. Functions: `sha256()`, `double_sha256()`.

4. **sha512.hpp/cpp + sha512.cuh** - Host SHA-512 + GPU SHA-512. GPU version has `device_sha512_32()` optimized single-block path for Ed25519 32-byte seed.

## What You Need To Do

### Immediate Task: Write Remaining Wave 1 Files

1. **Read each agent output file** listed in the table above
2. **Extract the source code** from the JSONL transcript (look for Write tool calls or code blocks in assistant messages)
3. **Write each file** to the correct path under `/Users/dddd/hype/hyperfanity/`
4. **Create directories** if they don't exist:
   - `src/crypto/`
   - `src/chain/`
   - `src/dispatch/`
   - `kernels/common/`
   - `kernels/secp256k1/`
   - `kernels/ed25519/`
   - `tools/`

### Known Issue: base58.cpp SHA-256 API Mismatch
The base58 agent's code references `sha256::hash()` but the sha256 module uses `crypto::sha256()` and `crypto::double_sha256()`. Fix the include/namespace when writing base58.cpp.

### After Writing Files

1. **Update TASKS.md** — mark T01-T15 as `[x]` complete
2. **Git commit** Wave 1: `git add` all new files and commit
3. **Proceed to Wave 2** (6 tasks: T16-T21) — see TASKS.md for deps and descriptions

## Wave 2 Tasks (Next)

| Task | Description | Dependencies |
|------|-------------|-------------|
| T16 | secp256k1 point ops (point_add, point_double) | T03 (mp_uint256) |
| T17 | Ed25519 group ops (ge25519, extended coords) | T08 (fe25519) |
| T18 | TRX address encoding | T05 (sha256), T07 (base58) |
| T19 | ETH address encoding + EIP-55 | T07 (base58/chain.hpp) |
| T20 | Solana address encoding | T07 (base58) |
| T21 | Scoring mode definitions | T02 (types) |

Launch all 6 in parallel once Wave 1 files are on disk and committed.

## Agent Orchestration Pattern

- Use `Task` tool with `subagent_type=general-purpose` for code generation agents
- Use `model=haiku` for simple tasks (encoding, tools), `model=sonnet` for medium (chain encoding), default opus for complex (GPU math kernels)
- Agents should be given: the file(s) they depend on (read from disk), the target file path, and a clear specification
- Agents run as background tasks but CANNOT write files — extract code from their output and write from the main thread
- Git commit after each wave

## Key Architecture Notes

- **secp256k1** (TRX+ETH): privkey × G → uncompressed pubkey → Keccak-256 → address hash
- **Ed25519** (Solana): seed → SHA-512 → clamp → scalar × B → compressed pubkey = address
- **Batch inverse (Algorithm 2.11)**: N inversions → 1 inversion + 3N muls — critical perf optimization
- **mp_number**: 8×uint32_t, little-endian word order
- **fe25519**: 5×uint64_t, radix-2^51
- CUDA not OpenCL: `__global__` not `__kernel`, `__umulhi` not `mul_hi`, `blockIdx.x*blockDim.x+threadIdx.x` not `get_global_id(0)`
