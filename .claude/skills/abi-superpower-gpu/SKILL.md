---
name: abi-superpower-gpu
description: GPU and Metal backend superpower. Reports backend status, vector ops, hardware capabilities.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["status", "ops", "hardware"]
      description: "GPU action"
    - name: "backend"
      type: "string"
      description: "GPU backend: metal, cuda, vulkan"
---

# ABI Superpower: GPU

Exposes GPU hardware and acceleration capabilities as a superpower.

## Actions

### status
Report GPU backend selection and fallback state:
```
/abi-superpower-gpu status
```

### ops
Vector operations via GPU/Metal acceleration:
```
/abi-superpower-gpu ops --input "vector data" --accumulate true
```

### hardware
System hardware report:
```
/abi-superpower-gpu hardware
```

## Implementation

Maps to:
- `crates/abi-gpu/src/reporting.rs` - Metal/CUDA/Vulkan detection
- `crates/abi-gpu/src/vector_ops.rs` - HNSW search acceleration
- `crates/abi-gpu/src/compute_api.rs` - SIMD/Metal kernels

## Feature Gate

Requires `feat-gpu=true` and native Metal/CUDA/Vulkan bindings.
When `accelerated=false` (CPU fallback), still reports metrics for transparency.