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
macOS builds. CUDA and Vulkan are capability reports only. Use `abi backends`
or `abi wdbx gpu info`; `accelerated=false` means deterministic CPU fallback.
