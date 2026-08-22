---
name: abi-superpower-wdbx-compute
description: WDBX compute superpower. Cycle-free accelerator contracts, deterministic CPU SIMD fallback, scoped Metal execution, output-checked CoreML inference, and honest CUDA/Vulkan boundaries.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["info", "select", "backend", "remote"]
      description: "Compute action"
    - name: "backend"
      type: "string"
      enum: ["cpu", "gpu", "npu", "tpu", "auto"]
      description: "Target backend"
    - name: "endpoint"
      type: "string"
      description: "Remote compute endpoint (TPU)"
---

> **WDBX moved out of this repository on 2026-08-22.** It now lives in the
> sibling repo `~/dev/active/wdbx` together with `abi-compute`,
> `abi-foundation`, `abi-core`, and `abi-telemetry`; `abi` consumes them by
> relative path. Source paths below therefore read `../wdbx/crates/...`. Run
> WDBX-only tests from that repo (`cargo test --workspace`), and `abi`'s gate
> (`./tools/check.sh`) from here.
>
> Under the Abbey System Constitution
> (`docs/superpowers/specs/2026-08-22-abbey-system-constitution.md`) WDBX is the
> **provenance-aware episodic substrate**, not a vector store. Most of the
> evidence half of its specification is unimplemented; the measured gap list is
> in `docs/superpowers/specs/2026-08-22-wdbx-conformance-gap-analysis.md`. Do not
> describe an episodic capability as Current on the strength of the vector-store
> features that do exist.

# ABI Superpower: WDBX Compute

Exposes the cycle-free `abi-compute` accelerator contract and WDBX selector.
Unsupported or unverified paths degrade to deterministic CPU SIMD. Supported
macOS builds can execute Metal dot/cosine/norm/batch-cosine; capability becomes
runtime-verified only after successful CPU-oracle parity. CoreML evidence is an
output-checked tiny-model inference under a `.cpuAndNeuralEngine` request, not
placement or residency proof. CUDA/Vulkan runtime remains unverified.

## Actions

### info
Show compute backend matrix (availability + dispatch status):
```
abi wdbx compute info
```

Output includes:
- CPU: scalar, AVX2, AVX-512, NEON through Rust `std::simd`
- GPU: Metal/CUDA/Vulkan five-state evidence; compilation is not execution
- NPU: ANE hardware presence plus output-checked CoreML inference, not residency
- TPU: Report-only endpoint metadata (`ABI_REMOTE_COMPUTE_ENDPOINT`) plus a
  separately tested reference TCP transport

### select
Backend selection is internal to the Rust compute API. There is no public
`select` subcommand; `abi wdbx compute info` reports the requested/effective
selection and its fallback reason.

### backend
There is no public per-backend subcommand. Use `abi backends` for the complete
capability matrix or `abi wdbx compute info` for WDBX selection.

### remote
`ABI_REMOTE_COMPUTE_ENDPOINT` opts `abi wdbx compute info` into one authenticated,
bounded DOT probe. The reference transport attempts the configured endpoint and
falls back to deterministic local CPU when the endpoint is unavailable. This is
execution evidence for the small reference request, not a production remote-TPU
service or a general WDBX offload path.

## Backend Matrix (Honest)

| Backend | Available | Dispatches | Reality |
|---------|-----------|------------|---------|
| CPU SIMD | Always | ✅ Always | Rust `std::simd` with host-matched width |
| Metal | Supported macOS builds | Scoped | Dot/cosine/norm/batch-cosine; runtime-verified only after CPU-oracle agreement, otherwise CPU fallback |
| CUDA | Adapter may compile | ❌ Unverified | Compilation/tool detection does not prove runtime initialization or execution |
| Vulkan | Adapter may compile | ❌ Unverified | Compilation/tool detection does not prove loader/device execution |
| CoreML / ANE | Optional inference helper | ❌ Residency unverified | Executes an output-checked tiny model under `.cpuAndNeuralEngine`; does not prove inference placement or ANE residency |
| Remote accelerator | Reference DOT only | Scoped probe | `remote_compute.rs` has an authenticated, timeout-bounded transport; the explicit environment endpoint is probed by `compute info` with deterministic CPU fallback |

## Implementation

| Component | Source | Role |
|-----------|--------|------|
| Contracts / CPU SIMD | `../wdbx/crates/abi-compute/src/` | Object-safe accelerator, five-state evidence, deterministic SIMD/top-k |
| WDBX batch search | `../wdbx/crates/abi-wdbx/src/v2/index.rs` | Accelerator injection with result validation and CPU parity/fallback |
| GPU adapters | `crates/abi-gpu/src/{adapters,metal_kernels}.rs` | Metal execution/oracle; CUDA/Vulkan unverified; CoreML output-checked without residency claim |
| NPU Detection | `../wdbx/crates/abi-compute/src/backend.rs` | `ane_hardware_present()` — hardware metadata only |
| Remote accelerator | `../wdbx/crates/abi-wdbx/src/remote_compute.rs` | Authenticated bounded reference DOT transport; `compute info` probes an explicitly configured endpoint and otherwise does no network I/O |
| Selection parity tests | `../wdbx/crates/abi-compute/src/cpu.rs`, `../wdbx/crates/abi-wdbx/src/v2/index.rs` | Accelerator requests preserve CPU-reference results |

## CLI Access

```
abi wdbx compute info
```

## Build and runtime boundary

`abi-compute`, `abi-wdbx`, and `abi-gpu` are normal Rust workspace crates. The
WDBX selector retains deterministic CPU fallback; no compiled/available adapter
is promoted to execution without later evidence states. The real CLI surface is
`abi wdbx compute info`.

## Claim Boundary

Per `docs/spec/wdbx-north-star.mdx` §3.3 and `docs/contracts/external-claims-audit.mdx`:
- ✅ Cycle-free object-safe accelerator contract and five-state evidence
- ✅ CPU SIMD parity across accelerator requests and fallback
- ✅ Scoped Metal numerical execution when initialized and oracle-verified
- ✅ ANE hardware detection and output-checked CoreML tiny-model inference
- ✅ Reference loopback TCP DOT transport with local fallback
- ⚠️ Metal execution remains supported-host/runtime scoped; availability alone is not execution evidence
- ❌ CUDA/Vulkan runtime execution is not verified
- ❌ CoreML inference under a compute-unit request is not placement or ANE residency
- ⚠️ `ABI_REMOTE_COMPUTE_ENDPOINT` drives only a bounded reference DOT probe with deterministic CPU fallback; it is not general WDBX offload or production remote acceleration
- ❌ No blanket accelerator speedup claim without reproducible benchmark evidence
