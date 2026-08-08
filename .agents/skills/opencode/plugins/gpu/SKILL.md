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
abi wdbx gpu info
```

### ops
Vector operations are a library surface, not a standalone CLI action. Validate
their native/fallback parity with `./tools/cargo.sh test -p abi-gpu`.

### hardware
System hardware report:
```
abi backends
```

## Implementation

Maps to:
- `crates/abi-gpu/src/lib.rs` - backend reporting, vector operations, and CPU fallback
- `crates/abi-gpu/src/metal_kernels.rs` - optional macOS Metal DOT kernel

## Build and runtime boundary

`abi-gpu` is a normal Rust workspace crate; there is no `feat-gpu` switch. Its
default `metal-kernels` Cargo feature links the Metal DOT path on supported
macOS builds. CUDA and Vulkan are capability reports only. Run `abi backends`
or `abi wdbx gpu info`; `accelerated=false` means deterministic CPU fallback.
