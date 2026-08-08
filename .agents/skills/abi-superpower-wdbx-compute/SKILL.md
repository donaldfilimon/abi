---
name: abi-superpower-wdbx-compute
description: WDBX compute backend superpower. CPU/GPU/NPU/TPU backend selector with deterministic CPU SIMD fallback. ANE detection, remote TPU dispatch.
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

# ABI Superpower: WDBX Compute

Exposes the WDBX compute backend selector as a superpower. It reports CPU
(scalar/AVX2/AVX-512/NEON), GPU (CUDA/Metal/Vulkan), NPU (ANE detection), and
the reference remote-compute path. Unsupported or unavailable native paths
degrade to deterministic CPU SIMD; the default `metal-kernels` Cargo feature
can link the Metal DOT path on supported macOS builds.

## Actions

### info
Show compute backend matrix (availability + dispatch status):
```
abi wdbx compute info
```

Output includes:
- CPU: scalar, AVX2, AVX-512, NEON through Rust `std::simd`
- GPU: CUDA, Metal, Vulkan capability rows; this selector reports native=false
  and resolves accelerator requests to CPU SIMD
- NPU: ANE hardware presence (truthful detection, no execution)
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
`ABI_REMOTE_COMPUTE_ENDPOINT` is report-only metadata for the public CLI. The
reference TCP DOT transport is library/test code, not a production remote-TPU
command.

## Backend Matrix (Honest)

| Backend | Available | Dispatches | Reality |
|---------|-----------|------------|---------|
| CPU SIMD | Always | ✅ Always | Rust `std::simd` with host-matched width |
| Metal | Capability row on macOS | ❌ No in this selector | `native=false`; effective backend is CPU SIMD. The separate `abi-gpu` layer may link Metal DOT |
| CUDA | Never | ❌ No | Capability reported, not linked |
| Vulkan | Never | ❌ No | Needs loader/ICD + SPIR-V; not linked |
| ANE (NPU) | On Apple Silicon | ❌ No | `compute::ane_hardware_present()` detects hardware; execution needs CoreML/ObjC |
| TPU (remote) | Reference only | ❌ Not production-wired | `remote_compute.rs` has a loopback-tested DOT transport; the environment endpoint is report-only |

## Implementation

| Component | Source | Role |
|-----------|--------|------|
| Backend Selector | `crates/abi-wdbx/src/compute.rs` | Dynamic CPU/GPU/NPU/TPU selection |
| CPU SIMD | `crates/abi-wdbx/src/compute.rs` | Rust `std::simd` DOT path and CPU selection |
| GPU status / Metal DOT | `crates/abi-gpu/src/lib.rs` + `metal_kernels.rs` | Separate GPU layer with CPU fallback |
| NPU Detection | `crates/abi-wdbx/src/compute.rs` | `ane_hardware_present()` — hardware metadata only |
| Remote TPU | `crates/abi-wdbx/src/remote_compute.rs` | Reference DOT transport; no production caller wires `ABI_REMOTE_COMPUTE_ENDPOINT` |
| Selection parity tests | `crates/abi-wdbx/src/compute.rs` | Accelerator requests preserve CPU-reference results |

## CLI Access

```
abi wdbx compute info
```

## Build and runtime boundary

`abi-wdbx` and `abi-gpu` are normal Rust workspace crates; there are no
`feat-wdbx` or `feat-gpu` switches. The WDBX selector reports every accelerator
as native=false and resolves it to deterministic CPU SIMD. Separately,
`abi-gpu` enables its Metal DOT feature by default on supported builds. The real
CLI surface is `abi wdbx compute info`.

## Claim Boundary

Per `docs/spec/wdbx-north-star.mdx` §3.3 and `docs/contracts/external-claims-audit.mdx`:
- ✅ Dynamic backend selector across CPU/GPU/NPU/TPU
- ✅ CPU SIMD parity across accelerator requests and fallback
- ✅ ANE hardware detection (truthful)
- ✅ Reference loopback TCP DOT transport with local fallback
- ⚠️ The separate `abi-gpu` layer may link Metal DOT on supported macOS builds
- ❌ WDBX selector does not natively dispatch Metal/CUDA/Vulkan/ANE/TPU
- ❌ `ABI_REMOTE_COMPUTE_ENDPOINT` is report-only in the public CLI
- ❌ ANE execution requires CoreML/ObjC (not available as a pure in-tree kernel) — disclosed non-goal
