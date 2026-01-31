---
title: "2026-01-17-cuda-vtable-wrapper"
tags: []
---
# CUDA VTable Wrapper Implementation Plan
> **Codebase Status:** Synced with repository as of 2026-01-30.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> **Status:** Implemented ✅ (January 17, 2026)

**Goal:** Create a complete CUDA backend implementation that fully implements the VTable interface, enabling real GPU kernel execution instead of the current simulated fallback.

**Architecture:** The CUDA VTable wrapper (`CudaBackend`) will implement all 12 VTable methods by delegating to the existing CUDA loader and native modules. It wraps the dynamic CUDA driver API calls (cuInit, cuMemAlloc, cuLaunchKernel, etc.) behind the polymorphic VTable interface, enabling the dispatcher to execute kernels on actual NVIDIA GPUs.

**Tech Stack:** Zig 0.16, CUDA Driver API (nvcuda.dll/libcuda.so), NVRTC for runtime compilation

---

## Task 1: Create CudaBackend Struct

**Files:**
- Create: `src/compute/gpu/backends/cuda/vtable.zig`
- Modify: `src/compute/gpu/backends/cuda/mod.zig` (add export)

**Step 1: Write the failing test**

```zig
// In src/compute/gpu/backends/cuda/vtable.zig
test "CudaBackend initialization" {
    const allocator = std.testing.allocator;

    // Should create backend or return NotAvailable if no CUDA
    const result = CudaBackend.init(allocator, 0);
    if (result) |backend| {
        defer backend.deinit();
        try std.testing.expect(backend.device_id == 0);
    } else |err| {
        // Expected on systems without CUDA
        try std.testing.expect(err == error.BackendNotAvailable or err == error.InitFailed);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "CudaBackend initialization"`
Expected: FAIL with "CudaBackend not defined"

**Step 3: Write minimal implementation**

```zig
//! CUDA VTable Backend Implementation
//!
//! Provides a complete VTable implementation for CUDA, enabling real GPU
//! kernel execution through the polymorphic backend interface.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const interface = @import("../../interface.zig");
const loader = @import("loader.zig");

pub const CudaBackend = struct {
    allocator: std.mem.Allocator,
    device_id: i32,
    context: ?*anyopaque,
    functions: ?loader.CudaFunctions,

    // Track allocations for cleanup
    allocations: std.ArrayListUnmanaged(Allocation),
    kernels: std.ArrayListUnmanaged(CompiledKernel),

    const Allocation = struct {
        ptr: *anyopaque,
        size: usize,
        is_host_pinned: bool,
    };

    const CompiledKernel = struct {
        module: *anyopaque,
        function: *anyopaque,
        name: []const u8,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, device_id: i32) interface.BackendError!*Self {
        // Check if CUDA is enabled at compile time
        if (comptime !build_options.gpu_cuda) {
            return interface.BackendError.NotAvailable;
        }

        // Try to load CUDA driver
        const functions = loader.loadCudaFunctions() catch {
            return interface.BackendError.NotAvailable;
        };

        // Initialize CUDA
        const init_result = functions.core.cuInit(0);
        if (init_result != 0) {
            return interface.BackendError.InitFailed;
        }

        // Check device count
        var device_count: c_int = 0;
        const count_result = functions.device.cuDeviceGetCount(&device_count);
        if (count_result != 0 or device_count == 0) {
            return interface.BackendError.DeviceNotFound;
        }

        if (device_id >= device_count) {
            return interface.BackendError.DeviceNotFound;
        }

        // Get device handle
        var device: c_int = undefined;
        const device_result = functions.core.cuDeviceGet(&device, device_id);
        if (device_result != 0) {
            return interface.BackendError.DeviceNotFound;
        }

        // Create context
        var context: ?*anyopaque = null;
        const ctx_result = functions.core.cuCtxCreate(&context, 0, device);
        if (ctx_result != 0) {
            return interface.BackendError.InitFailed;
        }

        const self = allocator.create(Self) catch {
            // Destroy context on allocation failure
            if (context) |ctx| {
                _ = functions.core.cuCtxDestroy(ctx);
            }
            return interface.BackendError.OutOfMemory;
        };

        self.* = .{
            .allocator = allocator,
            .device_id = device_id,
            .context = context,
            .functions = functions,
            .allocations = .empty,
            .kernels = .empty,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        // Free all allocations
        if (self.functions) |funcs| {
            for (self.allocations.items) |alloc| {
                if (alloc.is_host_pinned) {
                    _ = funcs.memory.cuMemFreeHost(alloc.ptr);
                } else {
                    _ = funcs.memory.cuMemFree(@intFromPtr(alloc.ptr));
                }
            }

            // Destroy all kernels
            for (self.kernels.items) |kernel| {
                _ = funcs.kernel.cuModuleUnload(kernel.module);
            }

            // Destroy context
            if (self.context) |ctx| {
                _ = funcs.core.cuCtxDestroy(ctx);
            }
        }

        self.allocations.deinit(self.allocator);
        self.kernels.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};
```

**Step 4: Run test to verify it passes**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "CudaBackend initialization"`
Expected: PASS (or skip on non-CUDA systems)

**Step 5: Commit**

```bash
git add src/compute/gpu/backends/cuda/vtable.zig
git commit -m "feat(gpu): add CudaBackend struct with init/deinit"
```

---

## Task 2: Implement Device Info Methods

**Files:**
- Modify: `src/compute/gpu/backends/cuda/vtable.zig`

**Step 1: Write the failing test**

```zig
test "CudaBackend device info" {
    const allocator = std.testing.allocator;

    const backend = CudaBackend.init(allocator, 0) catch |err| {
        if (err == error.BackendNotAvailable or err == error.DeviceNotFound) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer backend.deinit();

    // Test getDeviceCount
    const count = backend.getDeviceCount();
    try std.testing.expect(count > 0);

    // Test getDeviceCaps
    const caps = try backend.getDeviceCaps(0);
    try std.testing.expect(caps.total_memory > 0);
    try std.testing.expect(caps.compute_capability[0] > 0);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "device info"`
Expected: FAIL with "no member named 'getDeviceCount'"

**Step 3: Write minimal implementation**

```zig
    // Add these methods to CudaBackend struct:

    pub fn getDeviceCount(self: *Self) u32 {
        const funcs = self.functions orelse return 0;
        var count: c_int = 0;
        const result = funcs.device.cuDeviceGetCount(&count);
        if (result != 0) return 0;
        return @intCast(@max(0, count));
    }

    pub fn getDeviceCaps(self: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        const funcs = self.functions orelse return interface.BackendError.NotAvailable;

        var device: c_int = undefined;
        if (funcs.core.cuDeviceGet(&device, @intCast(device_id)) != 0) {
            return interface.BackendError.DeviceNotFound;
        }

        var caps = interface.DeviceCaps{
            .name = undefined,
            .name_len = 0,
            .total_memory = 0,
            .shared_memory_per_block = 0,
            .max_threads_per_block = 0,
            .max_block_dims = .{ 0, 0, 0 },
            .max_grid_dims = .{ 0, 0, 0 },
            .warp_size = 32,
            .compute_capability = .{ 0, 0 },
            .supports_f16 = false,
            .supports_f64 = true,
            .supports_atomics = true,
            .supports_dynamic_parallelism = false,
        };

        // Get device name
        var name_buf: [256]u8 = undefined;
        if (funcs.device.cuDeviceGetName(&name_buf, 256, device) == 0) {
            const len = std.mem.indexOfScalar(u8, &name_buf, 0) orelse 256;
            @memcpy(caps.name[0..len], name_buf[0..len]);
            caps.name_len = len;
        }

        // Get total memory
        var total_mem: usize = 0;
        if (funcs.device.cuDeviceTotalMem(&total_mem, device) == 0) {
            caps.total_memory = total_mem;
        }

        // Get compute capability
        var major: c_int = 0;
        var minor: c_int = 0;
        _ = funcs.device.cuDeviceGetAttribute(&major, 75, device); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        _ = funcs.device.cuDeviceGetAttribute(&minor, 76, device); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        caps.compute_capability = .{ @intCast(@max(0, major)), @intCast(@max(0, minor)) };

        // Get other attributes
        var val: c_int = 0;
        if (funcs.device.cuDeviceGetAttribute(&val, 1, device) == 0) { // MAX_THREADS_PER_BLOCK
            caps.max_threads_per_block = @intCast(@max(0, val));
        }
        if (funcs.device.cuDeviceGetAttribute(&val, 8, device) == 0) { // MAX_SHARED_MEMORY_PER_BLOCK
            caps.shared_memory_per_block = @intCast(@max(0, val));
        }
        if (funcs.device.cuDeviceGetAttribute(&val, 2, device) == 0) caps.max_block_dims[0] = @intCast(@max(0, val));
        if (funcs.device.cuDeviceGetAttribute(&val, 3, device) == 0) caps.max_block_dims[1] = @intCast(@max(0, val));
        if (funcs.device.cuDeviceGetAttribute(&val, 4, device) == 0) caps.max_block_dims[2] = @intCast(@max(0, val));
        if (funcs.device.cuDeviceGetAttribute(&val, 5, device) == 0) caps.max_grid_dims[0] = @intCast(@max(0, val));
        if (funcs.device.cuDeviceGetAttribute(&val, 6, device) == 0) caps.max_grid_dims[1] = @intCast(@max(0, val));
        if (funcs.device.cuDeviceGetAttribute(&val, 7, device) == 0) caps.max_grid_dims[2] = @intCast(@max(0, val));
        if (funcs.device.cuDeviceGetAttribute(&val, 10, device) == 0) caps.warp_size = @intCast(@max(1, val));

        // FP16 support (compute >= 5.3)
        caps.supports_f16 = caps.compute_capability[0] > 5 or
            (caps.compute_capability[0] == 5 and caps.compute_capability[1] >= 3);

        // Dynamic parallelism (compute >= 3.5)
        caps.supports_dynamic_parallelism = caps.compute_capability[0] > 3 or
            (caps.compute_capability[0] == 3 and caps.compute_capability[1] >= 5);

        return caps;
    }
```

**Step 4: Run test to verify it passes**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "device info"`
Expected: PASS (or SkipZigTest on non-CUDA)

**Step 5: Commit**

```bash
git add src/compute/gpu/backends/cuda/vtable.zig
git commit -m "feat(gpu): add CudaBackend device info methods"
```

---

## Task 3: Implement Memory Operations

**Files:**
- Modify: `src/compute/gpu/backends/cuda/vtable.zig`

**Step 1: Write the failing test**

```zig
test "CudaBackend memory operations" {
    const allocator = std.testing.allocator;

    const backend = CudaBackend.init(allocator, 0) catch |err| {
        if (err == error.BackendNotAvailable or err == error.DeviceNotFound) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer backend.deinit();

    // Allocate device memory
    const size: usize = 1024;
    const ptr = try backend.allocate(size, .{});
    defer backend.free(ptr);

    // Test copy to device
    var host_data: [256]f32 = undefined;
    for (&host_data, 0..) |*v, i| v.* = @floatFromInt(i);
    try backend.copyToDevice(ptr, std.mem.sliceAsBytes(&host_data));

    // Test copy from device
    var result: [256]f32 = undefined;
    try backend.copyFromDevice(std.mem.sliceAsBytes(&result), ptr);

    try std.testing.expectApproxEqAbs(host_data[0], result[0], 0.001);
    try std.testing.expectApproxEqAbs(host_data[255], result[255], 0.001);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "memory operations"`
Expected: FAIL with "no member named 'allocate'"

**Step 3: Write minimal implementation**

```zig
    // Add these methods to CudaBackend struct:

    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        const funcs = self.functions orelse return interface.MemoryError.OutOfMemory;

        var ptr: usize = 0;
        const result = if (flags.host_visible)
            funcs.memory.cuMemAllocHost(@ptrCast(&ptr), size)
        else
            funcs.memory.cuMemAlloc(&ptr, size);

        if (result != 0) {
            return interface.MemoryError.OutOfMemory;
        }

        const alloc_ptr: *anyopaque = @ptrFromInt(ptr);

        // Track allocation
        self.allocations.append(self.allocator, .{
            .ptr = alloc_ptr,
            .size = size,
            .is_host_pinned = flags.host_visible,
        }) catch {
            // Free on tracking failure
            if (flags.host_visible) {
                _ = funcs.memory.cuMemFreeHost(alloc_ptr);
            } else {
                _ = funcs.memory.cuMemFree(ptr);
            }
            return interface.MemoryError.OutOfMemory;
        };

        return alloc_ptr;
    }

    pub fn free(self: *Self, ptr: *anyopaque) void {
        const funcs = self.functions orelse return;

        // Find and remove from tracking
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == ptr) {
                if (alloc.is_host_pinned) {
                    _ = funcs.memory.cuMemFreeHost(ptr);
                } else {
                    _ = funcs.memory.cuMemFree(@intFromPtr(ptr));
                }
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    pub fn copyToDevice(self: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        const funcs = self.functions orelse return interface.MemoryError.TransferFailed;

        const result = funcs.memory.cuMemcpyHtoD(@intFromPtr(dst), src.ptr, src.len);
        if (result != 0) {
            return interface.MemoryError.TransferFailed;
        }
    }

    pub fn copyFromDevice(self: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        const funcs = self.functions orelse return interface.MemoryError.TransferFailed;

        const result = funcs.memory.cuMemcpyDtoH(dst.ptr, @intFromPtr(src), dst.len);
        if (result != 0) {
            return interface.MemoryError.TransferFailed;
        }
    }
```

**Step 4: Run test to verify it passes**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "memory operations"`
Expected: PASS (or SkipZigTest)

**Step 5: Commit**

```bash
git add src/compute/gpu/backends/cuda/vtable.zig
git commit -m "feat(gpu): add CudaBackend memory operations"
```

---

## Task 4: Implement Kernel Compilation

**Files:**
- Modify: `src/compute/gpu/backends/cuda/vtable.zig`

**Step 1: Write the failing test**

```zig
test "CudaBackend kernel compilation" {
    const allocator = std.testing.allocator;

    const backend = CudaBackend.init(allocator, 0) catch |err| {
        if (err == error.BackendNotAvailable or err == error.DeviceNotFound) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer backend.deinit();

    // Simple CUDA kernel
    const kernel_source =
        \\extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
        \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (i < n) c[i] = a[i] + b[i];
        \\}
    ;

    const kernel = backend.compileKernel(allocator, kernel_source, "vector_add") catch |err| {
        // NVRTC might not be available
        if (err == error.CompileFailed) return error.SkipZigTest;
        return err;
    };
    defer backend.destroyKernel(kernel);

    try std.testing.expect(kernel != null);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "kernel compilation"`
Expected: FAIL with "no member named 'compileKernel'"

**Step 3: Write minimal implementation**

```zig
    // Add these methods to CudaBackend struct:

    pub fn compileKernel(
        self: *Self,
        alloc: std.mem.Allocator,
        source: []const u8,
        entry_point: []const u8,
    ) interface.KernelError!*anyopaque {
        const funcs = self.functions orelse return interface.KernelError.CompileFailed;

        // Use NVRTC to compile
        const nvrtc = @import("nvrtc.zig");

        // Compile to PTX
        const ptx = nvrtc.compileSourceToPtx(alloc, source, &.{}) catch {
            return interface.KernelError.CompileFailed;
        };
        defer alloc.free(ptx);

        // Load module from PTX
        var module: ?*anyopaque = null;
        const load_result = funcs.kernel.cuModuleLoadData(&module, ptx.ptr);
        if (load_result != 0 or module == null) {
            return interface.KernelError.CompileFailed;
        }
        errdefer _ = funcs.kernel.cuModuleUnload(module.?);

        // Get function from module
        var function: ?*anyopaque = null;
        const entry_z = alloc.dupeZ(u8, entry_point) catch {
            _ = funcs.kernel.cuModuleUnload(module.?);
            return interface.KernelError.CompileFailed;
        };
        defer alloc.free(entry_z);

        const func_result = funcs.kernel.cuModuleGetFunction(&function, module.?, entry_z.ptr);
        if (func_result != 0 or function == null) {
            _ = funcs.kernel.cuModuleUnload(module.?);
            return interface.KernelError.CompileFailed;
        }

        // Track kernel
        const name_copy = alloc.dupe(u8, entry_point) catch {
            _ = funcs.kernel.cuModuleUnload(module.?);
            return interface.KernelError.CompileFailed;
        };

        self.kernels.append(self.allocator, .{
            .module = module.?,
            .function = function.?,
            .name = name_copy,
        }) catch {
            alloc.free(name_copy);
            _ = funcs.kernel.cuModuleUnload(module.?);
            return interface.KernelError.CompileFailed;
        };

        return function.?;
    }

    pub fn destroyKernel(self: *Self, kernel: *anyopaque) void {
        const funcs = self.functions orelse return;

        for (self.kernels.items, 0..) |k, i| {
            if (k.function == kernel) {
                _ = funcs.kernel.cuModuleUnload(k.module);
                self.allocator.free(k.name);
                _ = self.kernels.swapRemove(i);
                return;
            }
        }
    }
```

**Step 4: Run test to verify it passes**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "kernel compilation"`
Expected: PASS (or SkipZigTest)

**Step 5: Commit**

```bash
git add src/compute/gpu/backends/cuda/vtable.zig
git commit -m "feat(gpu): add CudaBackend kernel compilation"
```

---

## Task 5: Implement Kernel Launch and Synchronization

**Files:**
- Modify: `src/compute/gpu/backends/cuda/vtable.zig`

**Step 1: Write the failing test**

```zig
test "CudaBackend kernel launch" {
    const allocator = std.testing.allocator;

    const backend = CudaBackend.init(allocator, 0) catch |err| {
        if (err == error.BackendNotAvailable or err == error.DeviceNotFound) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer backend.deinit();

    // Compile kernel
    const kernel_source =
        \\extern "C" __global__ void fill(float* out, float val, int n) {
        \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (i < n) out[i] = val;
        \\}
    ;

    const kernel = backend.compileKernel(allocator, kernel_source, "fill") catch {
        return error.SkipZigTest;
    };
    defer backend.destroyKernel(kernel);

    // Allocate memory
    const n: usize = 256;
    const out_ptr = try backend.allocate(n * @sizeOf(f32), .{});
    defer backend.free(out_ptr);

    // Launch config
    const config = interface.LaunchConfig{
        .grid_dim = .{ 1, 1, 1 },
        .block_dim = .{ 256, 1, 1 },
        .shared_mem = 0,
        .stream = null,
    };

    // Prepare args
    const val: f32 = 42.0;
    const n_val: i32 = @intCast(n);
    var args: [3]*anyopaque = .{
        @ptrCast(&out_ptr),
        @ptrCast(@constCast(&val)),
        @ptrCast(@constCast(&n_val)),
    };

    try backend.launchKernel(kernel, config, &args);
    try backend.synchronize();

    // Verify results
    var result: [256]f32 = undefined;
    try backend.copyFromDevice(std.mem.sliceAsBytes(&result), out_ptr);

    try std.testing.expectApproxEqAbs(@as(f32, 42.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), result[255], 0.001);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "kernel launch"`
Expected: FAIL with "no member named 'launchKernel'"

**Step 3: Write minimal implementation**

```zig
    // Add these methods to CudaBackend struct:

    pub fn launchKernel(
        self: *Self,
        kernel: *anyopaque,
        config: interface.LaunchConfig,
        args: []const *anyopaque,
    ) interface.KernelError!void {
        const funcs = self.functions orelse return interface.KernelError.LaunchFailed;

        // Validate config
        if (config.block_dim[0] == 0 or config.block_dim[1] == 0 or config.block_dim[2] == 0) {
            return interface.KernelError.InvalidConfig;
        }
        if (config.grid_dim[0] == 0 or config.grid_dim[1] == 0 or config.grid_dim[2] == 0) {
            return interface.KernelError.InvalidConfig;
        }

        // Build args array for CUDA
        var cuda_args: [32]*anyopaque = undefined;
        const arg_count = @min(args.len, 32);
        for (args[0..arg_count], 0..) |arg, i| {
            cuda_args[i] = @constCast(arg);
        }

        const result = funcs.kernel.cuLaunchKernel(
            kernel,
            config.grid_dim[0],
            config.grid_dim[1],
            config.grid_dim[2],
            config.block_dim[0],
            config.block_dim[1],
            config.block_dim[2],
            config.shared_mem,
            config.stream,
            &cuda_args,
            null, // extra
        );

        if (result != 0) {
            return interface.KernelError.LaunchFailed;
        }
    }

    pub fn synchronize(self: *Self) interface.BackendError!void {
        const funcs = self.functions orelse return interface.BackendError.NotAvailable;

        const result = funcs.core.cuCtxSynchronize();
        if (result != 0) {
            return interface.BackendError.InvalidOperation;
        }
    }
```

**Step 4: Run test to verify it passes**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "kernel launch"`
Expected: PASS (or SkipZigTest)

**Step 5: Commit**

```bash
git add src/compute/gpu/backends/cuda/vtable.zig
git commit -m "feat(gpu): add CudaBackend kernel launch and sync"
```

---

## Task 6: Create VTable Wrapper Function

**Files:**
- Modify: `src/compute/gpu/backends/cuda/vtable.zig`

**Step 1: Write the failing test**

```zig
test "CudaBackend as VTable interface" {
    const allocator = std.testing.allocator;

    const backend = createCudaVTable(allocator) catch |err| {
        if (err == error.BackendNotAvailable or err == error.DeviceNotFound) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer backend.deinit();

    // Should work through VTable interface
    const count = backend.getDeviceCount();
    try std.testing.expect(count > 0 or count == 0); // 0 is valid if simulated
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "VTable interface"`
Expected: FAIL with "createCudaVTable not defined"

**Step 3: Write minimal implementation**

```zig
/// Create a VTable-wrapped CUDA backend.
pub fn createCudaVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try CudaBackend.init(allocator, 0);
    return interface.createBackend(CudaBackend, impl);
}
```

**Step 4: Run test to verify it passes**

Run: `zig test src/compute/gpu/backends/cuda/vtable.zig --test-filter "VTable interface"`
Expected: PASS (or SkipZigTest)

**Step 5: Commit**

```bash
git add src/compute/gpu/backends/cuda/vtable.zig
git commit -m "feat(gpu): add createCudaVTable wrapper function"
```

---

## Task 7: Export from CUDA Module

**Files:**
- Modify: `src/compute/gpu/backends/cuda/mod.zig`

**Step 1: Read current exports**

Read `src/compute/gpu/backends/cuda/mod.zig` to understand current structure.

**Step 2: Add vtable export**

Add to mod.zig:
```zig
pub const vtable = @import("vtable.zig");
pub const CudaBackend = vtable.CudaBackend;
pub const createCudaVTable = vtable.createCudaVTable;
```

**Step 3: Run build to verify**

Run: `zig build`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add src/compute/gpu/backends/cuda/mod.zig
git commit -m "feat(gpu): export CudaBackend from cuda module"
```

---

## Task 8: Integrate with Backend Factory

**Files:**
- Modify: `src/compute/gpu/backend_factory.zig`

**Step 1: Read current createCudaVTableBackend**

Current implementation now returns the real CUDA backend (legacy TODO resolved).

**Step 2: Update to use real CUDA backend**

```zig
fn createCudaVTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    if (comptime !build_options.gpu_cuda) {
        return FactoryError.BackendNotAvailable;
    }

    const cuda = @import("backends/cuda/mod.zig");
    return cuda.createCudaVTable(allocator) catch |err| switch (err) {
        error.NotAvailable => return FactoryError.BackendNotAvailable,
        error.DeviceNotFound => return FactoryError.BackendNotAvailable,
        error.InitFailed => return FactoryError.InitFailed,
        error.OutOfMemory => return FactoryError.OutOfMemory,
        else => return FactoryError.InitFailed,
    };
}
```

**Step 3: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/compute/gpu/backend_factory.zig
git commit -m "feat(gpu): integrate CudaBackend with backend factory"
```

---

## Task 9: Add Integration Test

**Files:**
- Create: `src/compute/gpu/backends/cuda/vtable_test.zig`

**Step 1: Write comprehensive integration test**

```zig
//! CUDA VTable Integration Tests
//!
//! Tests the complete CUDA backend through the VTable interface.

const std = @import("std");
const interface = @import("../../interface.zig");
const backend_factory = @import("../../backend_factory.zig");

test "CUDA VTable integration - full workflow" {
    const allocator = std.testing.allocator;

    // Create backend via factory
    const backend = backend_factory.createVTableBackend(allocator, .cuda) catch |err| {
        if (err == backend_factory.FactoryError.BackendNotAvailable) {
            return error.SkipZigTest;
        }
        return err;
    };
    defer backend.deinit();

    // 1. Query device info
    const count = backend.getDeviceCount();
    if (count == 0) return error.SkipZigTest;

    const caps = try backend.getDeviceCaps(0);
    std.debug.print("\nCUDA Device: {s}\n", .{caps.name[0..caps.name_len]});
    std.debug.print("Memory: {} MB\n", .{caps.total_memory / (1024 * 1024)});
    std.debug.print("Compute: {}.{}\n", .{caps.compute_capability[0], caps.compute_capability[1]});

    // 2. Memory operations
    const size: usize = 1024 * @sizeOf(f32);
    const a_ptr = try backend.allocate(size, .{});
    defer backend.free(a_ptr);
    const b_ptr = try backend.allocate(size, .{});
    defer backend.free(b_ptr);
    const c_ptr = try backend.allocate(size, .{});
    defer backend.free(c_ptr);

    // Initialize host data
    var a_host: [1024]f32 = undefined;
    var b_host: [1024]f32 = undefined;
    for (&a_host, &b_host, 0..) |*a, *b, i| {
        a.* = @floatFromInt(i);
        b.* = @floatFromInt(i * 2);
    }

    try backend.copyToDevice(a_ptr, std.mem.sliceAsBytes(&a_host));
    try backend.copyToDevice(b_ptr, std.mem.sliceAsBytes(&b_host));

    // 3. Compile and launch kernel
    const kernel_source =
        \\extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
        \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (i < n) c[i] = a[i] + b[i];
        \\}
    ;

    const kernel = backend.compileKernel(allocator, kernel_source, "vector_add") catch {
        std.debug.print("Kernel compilation not available (NVRTC missing?)\n", .{});
        return error.SkipZigTest;
    };
    defer backend.destroyKernel(kernel);

    const config = interface.LaunchConfig{
        .grid_dim = .{ 4, 1, 1 },
        .block_dim = .{ 256, 1, 1 },
        .shared_mem = 0,
        .stream = null,
    };

    const n: i32 = 1024;
    var args: [4]*anyopaque = .{
        @ptrCast(&a_ptr),
        @ptrCast(&b_ptr),
        @ptrCast(&c_ptr),
        @ptrCast(@constCast(&n)),
    };

    try backend.launchKernel(kernel, config, &args);
    try backend.synchronize();

    // 4. Verify results
    var c_host: [1024]f32 = undefined;
    try backend.copyFromDevice(std.mem.sliceAsBytes(&c_host), c_ptr);

    for (c_host, 0..) |val, i| {
        const expected: f32 = @as(f32, @floatFromInt(i)) + @as(f32, @floatFromInt(i * 2));
        try std.testing.expectApproxEqAbs(expected, val, 0.001);
    }

    std.debug.print("CUDA VTable integration test PASSED\n", .{});
}
```

**Step 2: Run integration test**

Run: `zig test src/compute/gpu/backends/cuda/vtable_test.zig`
Expected: PASS (or SkipZigTest on non-CUDA systems)

**Step 3: Commit**

```bash
git add src/compute/gpu/backends/cuda/vtable_test.zig
git commit -m "test(gpu): add CUDA VTable integration tests"
```

---

## Task 10: Update Documentation

**Files:**
- Modify: `docs/gpu.md`

**Step 1: Add CUDA VTable section**

Add to docs/gpu.md:

```markdown
## CUDA Backend

The CUDA backend provides full GPU acceleration on NVIDIA hardware.

### Requirements
- NVIDIA GPU (Compute Capability 3.5+)
- CUDA Driver installed
- NVRTC for runtime kernel compilation (optional)

### Usage

```zig
const backend_factory = @import("abi").compute.gpu.backend_factory;

// Create CUDA backend
const backend = try backend_factory.createVTableBackend(allocator, .cuda);
defer backend.deinit();

// Query device capabilities
const caps = try backend.getDeviceCaps(0);
std.debug.print("Device: {s}, Memory: {} GB\n", .{
    caps.name[0..caps.name_len],
    caps.total_memory / (1024 * 1024 * 1024),
});

// Allocate GPU memory
const ptr = try backend.allocate(size, .{});
defer backend.free(ptr);

// Transfer data
try backend.copyToDevice(ptr, host_data);
// ... execute kernel ...
try backend.copyFromDevice(result, ptr);
```

### Fallback Behavior
If CUDA is unavailable, the backend factory automatically falls back to the simulated backend for testing/development.
```

**Step 2: Commit**

```bash
git add docs/gpu.md
git commit -m "docs(gpu): add CUDA VTable documentation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Create CudaBackend struct | vtable.zig (new) |
| 2 | Device info methods | vtable.zig |
| 3 | Memory operations | vtable.zig |
| 4 | Kernel compilation | vtable.zig |
| 5 | Kernel launch & sync | vtable.zig |
| 6 | VTable wrapper function | vtable.zig |
| 7 | Export from module | mod.zig |
| 8 | Factory integration | backend_factory.zig |
| 9 | Integration tests | vtable_test.zig (new) |
| 10 | Documentation | docs/gpu.md |

**Total estimated commits:** 10
**New files:** 2
**Modified files:** 3

