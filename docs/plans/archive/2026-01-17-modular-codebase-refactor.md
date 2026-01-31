---
title: "2026-01-17-modular-codebase-refactor"
tags: []
---
# Modular Codebase Refactor Design
> **Codebase Status:** Synced with repository as of 2026-01-30.

**Date:** 2026-01-17
**Status:** Complete (All Phases)
**Author:** Claude Code

## Executive Summary

This design document outlines a modular refactoring of the ABI codebase to improve maintainability, reduce coupling, and establish clearer module boundaries without breaking existing functionality.

## Goals

1. **Extract GPU backend common layer** - Reduce duplication across CUDA, Vulkan, Metal, WebGPU, OpenGL backends
2. **Establish module visibility markers** - Document public vs internal APIs
3. **Create backend factory pattern** - Unified backend instantiation
4. **Add dispatcher abstraction** - Single entry point for backend operations

## Current State Analysis

### GPU Backend Structure
```
src/compute/gpu/
├── mod.zig              # 268 lines - Main GPU module
├── backend.zig          # Backend detection
├── unified.zig          # Unified GPU API
├── backends/
│   ├── cuda/            # CUDA backend (multi-file)
│   ├── vulkan*.zig      # Vulkan backend (6 files)
│   ├── metal.zig        # Metal backend
│   ├── webgpu.zig       # WebGPU backend
│   ├── opengl.zig       # OpenGL backend
│   ├── opengles.zig     # OpenGL ES backend
│   └── webgl2.zig       # WebGL2 backend
```

### Identified Duplication
Each backend implements similar patterns:
- `init()` / `deinit()` lifecycle
- `compileKernel()` / `destroyKernel()`
- `launchKernel()` execution
- `allocateDeviceMemory()` / `freeDeviceMemory()`
- `memcpyHostToDevice()` / `memcpyDeviceToHost()`

## Proposed Architecture

### 1. Backend VTable Interface

Create a common vtable interface that all backends implement:

```zig
// src/compute/gpu/backend_vtable.zig
pub const BackendVTable = struct {
    // Lifecycle
    init: *const fn () anyerror!void,
    deinit: *const fn () void,

    // Kernel operations
    compileKernel: *const fn (source: []const u8, config: KernelConfig) anyerror!*CompiledKernel,
    launchKernel: *const fn (kernel: *CompiledKernel, config: LaunchConfig) anyerror!void,
    destroyKernel: *const fn (kernel: *CompiledKernel) void,

    // Memory operations
    allocate: *const fn (size: usize) anyerror!*DeviceMemory,
    free: *const fn (mem: *DeviceMemory) void,
    copyToDevice: *const fn (dst: *DeviceMemory, src: []const u8) anyerror!void,
    copyFromDevice: *const fn (dst: []u8, src: *DeviceMemory) anyerror!void,

    // Metadata
    name: []const u8,
    backend_type: Backend,
};
```

### 2. Backend Factory

```zig
// src/compute/gpu/backend_factory.zig
pub const BackendFactory = struct {
    pub fn create(backend: Backend, allocator: Allocator) !*BackendVTable {
        return switch (backend) {
            .cuda => cuda.getVTable(),
            .vulkan => vulkan.getVTable(),
            .metal => metal.getVTable(),
            .webgpu => webgpu.getVTable(),
            .opengl => opengl.getVTable(),
            .opengles => opengles.getVTable(),
            .stdgpu => stdgpu.getVTable(),
            else => error.UnsupportedBackend,
        };
    }

    pub fn createBest(allocator: Allocator) !*BackendVTable {
        // Auto-detect best available backend
    }
};
```

### 3. Unified Dispatcher

```zig
// src/compute/gpu/dispatcher.zig
pub const Dispatcher = struct {
    backends: []const *BackendVTable,
    active: *BackendVTable,
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: DispatcherConfig) !Dispatcher {
        // Initialize with preferred backend order
    }

    pub fn execute(self: *Dispatcher, kernel: *CompiledKernel, config: LaunchConfig) !void {
        return self.active.launchKernel(kernel, config);
    }

    pub fn allocateBuffer(self: *Dispatcher, size: usize) !*DeviceMemory {
        return self.active.allocate(size);
    }
};
```

### 4. Builtin Kernels Registry

```zig
// src/compute/gpu/builtin_kernels.zig
pub const BuiltinKernels = struct {
    vectorAdd: ?*CompiledKernel = null,
    matrixMultiply: ?*CompiledKernel = null,
    softmax: ?*CompiledKernel = null,
    reduceSum: ?*CompiledKernel = null,

    pub fn init(dispatcher: *Dispatcher) !BuiltinKernels {
        // Pre-compile common kernels
    }

    pub fn deinit(self: *BuiltinKernels, dispatcher: *Dispatcher) void {
        // Clean up compiled kernels
    }
};
```

## Implementation Plan

### Phase 1: Create Infrastructure (This Session)
1. Create `backend_factory.zig` with factory pattern
2. Create `dispatcher.zig` for unified dispatch
3. Create `builtin_kernels.zig` for common operations
4. Add VTable to CUDA backend as reference implementation

### Phase 2: Migrate Backends (Future)
1. Add VTable to Vulkan backend
2. Add VTable to Metal backend
3. Add VTable to WebGPU backend
4. Add VTable to OpenGL/OpenGL ES backends

### Phase 3: Simplify API Surface (Future)
1. Update `unified.zig` to use dispatcher
2. Reduce exports in `mod.zig`
3. Add module visibility documentation

## Files to Create

| File | Purpose |
|------|---------|
| `src/compute/gpu/backend_factory.zig` | Backend instantiation factory |
| `src/compute/gpu/dispatcher.zig` | Unified dispatch layer |
| `src/compute/gpu/builtin_kernels.zig` | Pre-compiled common kernels |
| `src/compute/gpu/backends/cuda/vtable.zig` | CUDA vtable implementation |

## Success Criteria

1. All existing tests pass
2. `zig build` succeeds with all feature combinations
3. New abstractions are backward-compatible
4. No performance regression in GPU operations

## Completion Status

### Phase 1 Complete (2026-01-17)

- [x] Created `backend_factory.zig` with factory pattern
- [x] Created `dispatcher.zig` for unified dispatch
- [x] Created `builtin_kernels.zig` for common operations
- [x] Created `backends/cuda/vtable.zig` CUDA vtable implementation
- [x] Added module visibility documentation to GPU module
- [x] Wired new abstractions into `mod.zig`
- [x] All 51 tests passing
- [x] Build verified

### Phase 2 Complete (2026-01-17)

- [x] Added VTable to Vulkan backend (`backends/vulkan_vtable.zig`)
- [x] Added VTable to Metal backend (`backends/metal_vtable.zig`)
- [x] Added VTable to WebGPU backend (`backends/webgpu_vtable.zig`)
- [x] Wired all VTables into `backend_factory.zig`
- [x] All 51 tests passing
- [x] Build verified

### Phase 3 Complete (2026-01-17)

- [x] Updated `unified.zig` to use dispatcher
  - Added `dispatcher` field to `Gpu` struct
  - Integrated `KernelDispatcher` into `vectorAdd`, `matrixMultiply`, `reduceSum`, `dotProduct`, `softmax`
  - Added `getDispatcher()` and `getDispatcherStats()` public methods
- [x] Fixed Zig 0.16 compatibility issues in DSL module
  - Updated `codegen/common.zig` to use `bufPrint`/`allocPrint` instead of deprecated `.writer()`
  - Updated `kernel.zig` to use named `BindingKey` struct instead of anonymous struct
- [x] All 51 tests passing
- [x] Build verified

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing API | Keep old exports, add new layer on top |
| Performance overhead from vtable | Use comptime dispatch where possible |
| Incomplete backend coverage | Start with CUDA, expand incrementally |

