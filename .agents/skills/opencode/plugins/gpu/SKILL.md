---
name: gpu
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

# GPU Superpower Plugin

Core GPU capabilities for OpenCode within the ABI framework.

## Capabilities

- GPU subsystem integration
- Plugin framework registration
- Runtime lifecycle management
- Configuration and settings management
- Status monitoring and reporting

## Integration Points

- ABI's GPU subsystem integration
- OpenCode plugin framework integration
- Runtime lifecycle management
- Configuration and settings management

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
- `src/features/gpu/status.zig` - Metal/CUDA/Vulkan detection
- `src/features/gpu/vector_ops.zig` - HNSW search acceleration
- `src/features/gpu/compute.zig` - SIMD/Metal kernels

## Feature Gate

Requires `feat-gpu=true` and native Metal/CUDA/Vulkan bindings.
When `accelerated=false` (CPU fallback), still reports metrics for transparency.
