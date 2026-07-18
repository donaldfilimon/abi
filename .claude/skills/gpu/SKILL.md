---
name: gpu
description: Plan abi GPU/backend work — Metal on macOS, CPU SIMD fallback, and the disclosed honest-stub backends (accelerator, shaders, mlir, mobile). Use when asked about GPU/backends, why accelerated=false, or when planning backend work. Routes to backend-diagnostics and abi-superpower-gpu. Never claims native CUDA/ANE/Metal-kernel execution — those are disclosed non-goals.
---

# gpu

Entry point for abi's GPU/backend surface (`src/features/gpu/` + the four
honest-stub feature modules). Routes to specialists:

| You want to… | Use |
| --- | --- |
| Report GPU/accelerator/shader/MLIR status + compute matrix | `backend-diagnostics` |
| Deep-dive the GPU/Metal superpower | `abi-superpower-gpu` |

## Honest status (trust the source flags over any prose)
- **Metal on macOS is real**: linked at build time; `accelerated=false` is the
  normal state until `g_metal_context.init()` succeeds at runtime; mid-run
  failure degrades to CPU. No `-Dgpu-backend` option exists.
- **Honest stubs** (`available=false` / `native_dispatch=false` in each
  `src/features/*/mod.zig`): `accelerator` (selection report + CPU SIMD
  fallback only), `shaders` (validate + checksum, no compiler), `mlir`
  (textual lower only, no LLVM toolchain), `mobile` (profile report only, no
  runtime).
- **ANE execution is a disclosed non-goal** (100% Zig constraint; requires
  CoreML/ObjC). Native CUDA/Vulkan/Metal-kernel execution is not linked.

## Hard rule
Do not claim real shader compilation, MLIR/LLVM lowering, or native
accelerator dispatch — per `docs/contracts/external-claims-audit.mdx`.
