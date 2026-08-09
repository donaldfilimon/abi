---
name: gpu
description: Plan abi GPU/backend work — Metal on macOS, deterministic CPU fallback, and disclosed capability-only CUDA/Vulkan/ANE rows. Use when asked about GPU/backends, why accelerated=false, or when planning backend work. Routes to backend-diagnostics and abi-superpower-gpu and never promotes a compiled adapter to executed acceleration.
---

# gpu

Entry point for abi's GPU/backend surface (`crates/abi-gpu/src/` + the four
honest-stub feature modules). Routes to specialists:

| You want to… | Use |
| --- | --- |
| Report GPU/accelerator/shader/MLIR status + compute matrix | `backend-diagnostics` |
| Deep-dive the GPU/Metal superpower | `abi-superpower-gpu` |

## Honest status (trust the source flags over any prose)
- **Metal DOT is real when initialized** through `crates/abi-gpu`; failed or
  unavailable initialization degrades deterministically to CPU.
- **Capability-only rows** must report `initialized=false`, `executed=false`,
  and `runtime_verified=false` until runtime evidence exists. CUDA/Vulkan remain
  feature/target compile work, not local runtime proof on this Mac.
- **ANE/CoreML evidence is request plus successful inference**, not proof of ANE
  residency. Keep that distinction in capability output and documentation.

## Hard rule
Do not claim native dispatch from compilation or availability alone. Require an
initialized, executed, runtime-verified state and preserve CPU oracle parity.
