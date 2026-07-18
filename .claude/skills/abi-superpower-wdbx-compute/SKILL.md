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

Exposes the WDBX compute backend selector as a superpower. Dynamic backend selection across CPU (scalar/AVX2/AVX-512/NEON), GPU (CUDA/Metal/Vulkan), NPU (ANE detection), and TPU (remote dispatch transport). **Always degrades to deterministic CPU SIMD path** — native accelerator kernels are NOT linked.

## Actions

### info
Show compute backend matrix (availability + dispatch status):
```
/abi-superpower-wdbx-compute info
```

Output includes:
- CPU: scalar, AVX2, AVX-512, NEON (host-detected via `std.simd.suggestVectorLength`)
- GPU: CUDA, Metal, Vulkan (capability reporting only — no native kernels linked)
- NPU: ANE hardware presence (truthful detection, no execution)
- TPU: Report-only endpoint metadata (`ABI_REMOTE_COMPUTE_ENDPOINT`) plus a
  separately tested reference TCP transport

### select
Select and initialize a backend:
```
/abi-superpower-wdbx-compute select --backend auto
```

Options: `cpu`, `gpu`, `npu`, `tpu`, `auto` (prefers Metal on macOS, else CPU SIMD)

### backend
Get details for a specific backend:
```
/abi-superpower-wdbx-compute backend --backend metal
```

### remote
Show or plan remote TPU reference transport configuration:
```
/abi-superpower-wdbx-compute remote --endpoint http://tpu-server:8080
```

## Backend Matrix (Honest)

| Backend | Available | Dispatches | Reality |
|---------|-----------|------------|---------|
| CPU SIMD | Always | ✅ Always | Vectorized `@Vector` with host-matched width |
| Metal (macOS) | On macOS | ⚠️ Only if `g_metal_context.init()` succeeds | Pure-Zig ObjC FFI, runtime MSL compile |
| CUDA | Never | ❌ No | Capability reported, not linked |
| Vulkan | Never | ❌ No | Needs loader/ICD + SPIR-V; not linked |
| ANE (NPU) | On Apple Silicon | ❌ No | `compute.aneHardwarePresent()` truthfully detects; needs CoreML/ObjC |
| TPU (remote) | Reference only | ❌ Not production-wired | `remote_compute.zig` has a loopback-tested DOT transport; the environment endpoint is report-only |

## Implementation

| Component | Source | Role |
|-----------|--------|------|
| Backend Selector | `src/features/wdbx/compute.zig` | Dynamic CPU/GPU/NPU/TPU selection |
| CPU SIMD | `src/features/wdbx/hnsw_distance.zig` | `@Vector` cosine, `std.simd.suggestVectorLength` |
| GPU | `src/features/gpu/vector_ops.zig` + `compute_api.zig` | `cosineSimilarity()` via Metal, CPU fallback |
| NPU Detection | `src/features/wdbx/compute.zig` | `aneHardwarePresent()` — truthful |
| Remote TPU | `src/features/wdbx/remote_compute.zig` | Reference DOT transport; no production caller wires `ABI_REMOTE_COMPUTE_ENDPOINT` |
| Parity Test | `src/features/gpu/compute_api.zig` | CPU/GPU dot-product parity |

## CLI Access

```
abi wdbx compute info
```

## Feature Gates

Requires `feat-wdbx=true` and `feat-gpu=true` (both default). When disabled, returns `FeatureDisabled`.

## Claim Boundary

Per `docs/spec/wdbx-north-star.mdx` §3.3 and `docs/contracts/external-claims-audit.mdx`:
- ✅ Dynamic backend selector across CPU/GPU/NPU/TPU
- ✅ CPU SIMD parity test against GPU path
- ✅ ANE hardware detection (truthful)
- ✅ Remote TPU dispatch transport (operator's endpoint)
- ❌ Native CUDA/Metal/Vulkan/ANE kernel execution NOT linked
- ❌ No bundled accelerator — TPU points at operator's own service
- ❌ ANE execution requires CoreML/ObjC (not pure Zig) — disclosed non-goal
