//! Integration with Zig 0.16's std.gpu facilities
//!
//! Provides a bridge between our backend interface and Zig's standard library
//! GPU abstraction. This module enables:
//!
//! - Automatic GPU target detection (SPIR-V)
//! - CPU fallback for non-GPU targets
//! - Integration with Vulkan backend for SPIR-V execution
//!
//! ## Memory Ownership
//!
//! - `StdGpuDevice` owns its internal state; call `deinit()` when done
//! - `StdGpuBuffer` owns allocated memory; call `deinit()` when done
//! - `StdGpuQueue` owns its state; call `deinit()` when done
//! - `compileShaderToSpirv(allocator, ...)` returns allocated memory;
//!   **caller must free** with `allocator.free(result)`
//!
//! ## GPU Execution Path
//!
//! When targeting SPIR-V (via `-target spirv64-unknown`), the kernels in
//! `std_gpu_kernels.zig` compile to native GPU code that can be executed
//! via Vulkan compute pipelines.
//!
//! Example:
//! ```zig
//! var device = try initStdGpuDevice(allocator);
//! defer device.deinit();
//!
//! var buffer = try device.createBuffer(.{ .size = 1024 });
//! defer buffer.deinit();
//! ```

const std = @import("std");
const builtin = @import("builtin");
const std_gpu_kernels = @import("../std_gpu_kernels.zig");

// Import dispatch types for BuiltinKernel enum and Buffer
const dsl_kernel = @import("../dsl/kernel.zig");
const unified_buffer = @import("../unified_buffer.zig");

// Inline GPU target detection (same as std_gpu.zig)
const is_gpu_target = builtin.cpu.arch.isSpirV();
const std_gpu_available = @hasDecl(std, "gpu");

pub const StdGpuError = error{
    InitializationFailed,
    QueueCreationFailed,
    BufferCreationFailed,
    ShaderCompilationFailed,
    PipelineCreationFailed,
    OutOfMemory,
    StdGpuNotAvailable,
};

/// Check if std.gpu is available in this Zig version
pub fn isStdGpuAvailable() bool {
    return std_gpu_available;
}

/// Check if we're running on a GPU target
pub fn isGpuTarget() bool {
    return is_gpu_target;
}

/// Wrapper around std.gpu.Device (compatibility layer)
pub const StdGpuDevice = struct {
    allocator: std.mem.Allocator,
    /// True if running CPU emulation (non-SPIR-V target)
    is_emulated: bool,

    pub fn deinit(self: *StdGpuDevice) void {
        _ = self;
        // Cleanup when std.gpu is available
    }

    pub fn createQueue(self: *StdGpuDevice) !StdGpuQueue {
        return StdGpuQueue{
            .allocator = self.allocator,
            .is_emulated = self.is_emulated,
        };
    }

    pub fn createBuffer(self: *StdGpuDevice, desc: BufferDescriptor) !StdGpuBuffer {
        // Allocate CPU-side buffer for data staging or emulation
        const buffer_data = try self.allocator.alloc(u8, desc.size);
        errdefer self.allocator.free(buffer_data);

        return StdGpuBuffer{
            .data = buffer_data,
            .size = desc.size,
            .allocator = self.allocator,
            .is_emulated = self.is_emulated,
        };
    }

    /// Create a typed buffer for compute operations
    pub fn createTypedBuffer(self: *StdGpuDevice, comptime T: type, count: usize) !TypedBuffer(T) {
        const size = count * @sizeOf(T);
        // Use regular alloc - typed buffer handles alignment via @alignCast at access time
        const buffer_data = try self.allocator.alloc(u8, size);
        errdefer self.allocator.free(buffer_data);

        return TypedBuffer(T){
            .data = buffer_data,
            .count = count,
            .allocator = self.allocator,
            .is_emulated = self.is_emulated,
        };
    }

    /// Execute vector addition using std_gpu kernels
    pub fn vectorAdd(
        self: *StdGpuDevice,
        a: []const f32,
        b: []const f32,
        result: []f32,
    ) !void {
        if (a.len != b.len or a.len != result.len) {
            return error.SizeMismatch;
        }

        // CPU fallback implementation (used for both emulated and real GPU until
        // Vulkan dispatch is integrated)
        _ = self;
        vectorAddCpu(a, b, result);
    }

    /// Execute matrix multiplication using std_gpu kernels
    pub fn matrixMul(
        self: *StdGpuDevice,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        if (a.len != m * k or b.len != k * n or c.len != m * n) {
            return error.SizeMismatch;
        }

        // CPU fallback implementation
        _ = self;
        matrixMulCpu(a, b, c, m, n, k);
    }
};

pub const StdGpuQueue = struct {
    allocator: std.mem.Allocator,
    is_emulated: bool,

    pub fn deinit(self: *StdGpuQueue) void {
        _ = self;
    }

    pub fn submit(self: *StdGpuQueue) !void {
        if (!self.is_emulated) {
            // On real GPU, this would submit command buffer
        }
        // On emulated device, operations execute synchronously
    }

    pub fn waitIdle(self: *StdGpuQueue) !void {
        _ = self;
        // On emulated device, already idle
    }
};

pub const BufferDescriptor = struct {
    size: usize,
    usage: BufferUsage = .{},
};

pub const BufferUsage = struct {
    storage: bool = false,
    uniform: bool = false,
    copy_dst: bool = false,
    copy_src: bool = false,
};

pub const StdGpuBuffer = struct {
    data: []u8,
    size: usize,
    allocator: std.mem.Allocator,
    is_emulated: bool,

    pub fn deinit(self: *StdGpuBuffer) void {
        self.allocator.free(self.data);
    }

    pub fn write(self: *StdGpuBuffer, offset: usize, data: []const u8) !void {
        if (offset + data.len > self.size) {
            return error.BufferTooSmall;
        }

        @memcpy(self.data[offset..][0..data.len], data);
    }

    pub fn read(self: *StdGpuBuffer, offset: usize, data: []u8) !void {
        if (offset + data.len > self.size) {
            return error.BufferTooSmall;
        }

        @memcpy(data, self.data[offset..][0..data.len]);
    }

    /// Get typed view of buffer data
    pub fn asSlice(self: *StdGpuBuffer, comptime T: type) []T {
        const ptr: [*]T = @ptrCast(@alignCast(self.data.ptr));
        return ptr[0 .. self.size / @sizeOf(T)];
    }
};

/// Typed buffer for specific element types
pub fn TypedBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []u8,
        count: usize,
        allocator: std.mem.Allocator,
        is_emulated: bool,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
        }

        pub fn asSlice(self: *Self) []T {
            const ptr: [*]T = @ptrCast(@alignCast(self.data.ptr));
            return ptr[0..self.count];
        }

        pub fn asConstSlice(self: *const Self) []const T {
            const ptr: [*]const T = @ptrCast(@alignCast(self.data.ptr));
            return ptr[0..self.count];
        }
    };
}

/// Initialize a std.gpu device (or CPU fallback)
pub fn initStdGpuDevice(allocator: std.mem.Allocator) !StdGpuDevice {
    // Check if we're on a GPU target
    const is_emulated = !is_gpu_target;

    return StdGpuDevice{
        .allocator = allocator,
        .is_emulated = is_emulated,
    };
}

// ============================================================================
// CPU Fallback Implementations
// ============================================================================

/// Execute vector addition on CPU
fn vectorAddCpu(a: []const f32, b: []const f32, result: []f32) void {
    for (a, b, result) |av, bv, *rv| {
        rv.* = av + bv;
    }
}

/// Execute matrix multiplication on CPU
fn matrixMulCpu(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0.0;
            for (0..k) |i| {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

/// Compile SPIR-V shader using std.gpu
///
/// Returns allocated SPIR-V bytecode. **Caller owns the returned memory**
/// and must free it with `allocator.free(result)` when done.
///
/// NOTE: For actual GPU execution, compile the kernel module for SPIR-V:
/// ```bash
/// zig build-obj -target spirv64-unknown -O ReleaseFast src/features/gpu/std_gpu_kernels.zig
/// ```
/// Then load the resulting .spv file with the Vulkan backend.
pub fn compileShaderToSpirv(
    allocator: std.mem.Allocator,
    source: []const u8,
    entry_point: []const u8,
) ![]const u32 {
    _ = source;
    _ = entry_point;

    // Return SPIR-V header as placeholder
    // Actual compilation requires:
    // 1. Build kernel module with -target spirv64-unknown
    // 2. Extract entry point from resulting .spv file
    const spirv_header = [_]u32{
        0x07230203, // SPIR-V magic
        0x00010000, // Version 1.0
        0x00000000, // Generator
        0x00000001, // Bound
        0x00000000, // Schema
    };

    const result = try allocator.alloc(u32, spirv_header.len);
    @memcpy(result, &spirv_header);
    return result;
}

/// Get kernel info for Vulkan pipeline creation
pub const KernelInfo = struct {
    name: []const u8,
    workgroup_size: [3]u32,
    num_buffers: u32,
};

/// List of available kernels and their configurations
pub const available_kernels = [_]KernelInfo{
    .{ .name = "vectorAdd", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 4 },
    .{ .name = "vectorSub", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 4 },
    .{ .name = "vectorMul", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 4 },
    .{ .name = "vectorScale", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 4 },
    .{ .name = "vectorFMA", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 5 },
    .{ .name = "reduceSum", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 3 },
    .{ .name = "reduceMax", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 3 },
    .{ .name = "matrixMul", .workgroup_size = .{ 16, 16, 1 }, .num_buffers = 6 },
    .{ .name = "matrixMulTiled", .workgroup_size = .{ 16, 16, 1 }, .num_buffers = 6 },
    .{ .name = "relu", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 3 },
    .{ .name = "sigmoid", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 3 },
    .{ .name = "silu", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 3 },
    .{ .name = "softmaxNumerator", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 4 },
    .{ .name = "softmaxNormalize", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 3 },
    .{ .name = "rmsNorm", .workgroup_size = .{ 256, 1, 1 }, .num_buffers = 5 },
};

/// Calculate number of workgroups needed
pub fn calculateWorkgroups(total_elements: u32, workgroup_size: u32) u32 {
    return (total_elements + workgroup_size - 1) / workgroup_size;
}

/// Calculate 2D workgroup counts for matrix operations
pub fn calculateWorkgroups2D(rows: u32, cols: u32, tile_size: u32) [2]u32 {
    return .{
        (cols + tile_size - 1) / tile_size,
        (rows + tile_size - 1) / tile_size,
    };
}

// ============================================================================
// Native Zig Kernel Registry
// ============================================================================

/// Registry that maps BuiltinKernel enum values to native Zig kernel
/// implementations from std_gpu_kernels.zig.
///
/// When the active backend is `.stdgpu` or `.simulated`, the dispatch
/// coordinator tries this registry before falling back to the DSL-generated
/// kernel pipeline. On CPU targets the SPIR-V kernels degrade to regular
/// Zig function calls, so we invoke the dedicated CPU fallback routines
/// which operate on slices for efficiency.
pub const StdGpuKernelRegistry = struct {
    pub const BuiltinKernel = dsl_kernel.BuiltinKernel;
    pub const Buffer = unified_buffer.Buffer;

    /// Execute a built-in kernel using native Zig implementations.
    /// Returns `true` if the kernel was handled natively, `false` if not
    /// supported (caller should fall through to the DSL pipeline).
    pub fn executeBuiltin(
        kernel: BuiltinKernel,
        buffers: []const *Buffer,
        global_size: [3]u32,
        uniforms: []const *const anyopaque,
        uniform_sizes: []const usize,
    ) bool {
        switch (kernel) {
            .vector_add => {
                if (buffers.len < 3) return false;
                const a = bufSliceF32(buffers[0]) orelse return false;
                const b = bufSliceF32(buffers[1]) orelse return false;
                const r = bufSliceMutF32(buffers[2]) orelse return false;
                std_gpu_kernels.vectorAddCpu(a, b, r);
                return true;
            },
            .vector_sub => {
                if (buffers.len < 3) return false;
                const a = bufSliceF32(buffers[0]) orelse return false;
                const b = bufSliceF32(buffers[1]) orelse return false;
                const r = bufSliceMutF32(buffers[2]) orelse return false;
                const len = @min(a.len, @min(b.len, r.len));
                for (0..len) |i| {
                    r[i] = a[i] - b[i];
                }
                return true;
            },
            .vector_mul => {
                if (buffers.len < 3) return false;
                const a = bufSliceF32(buffers[0]) orelse return false;
                const b = bufSliceF32(buffers[1]) orelse return false;
                const r = bufSliceMutF32(buffers[2]) orelse return false;
                std_gpu_kernels.vectorMulCpu(a, b, r);
                return true;
            },
            .vector_scale => {
                if (buffers.len < 2) return false;
                const a = bufSliceF32(buffers[0]) orelse return false;
                const r = bufSliceMutF32(buffers[1]) orelse return false;
                const scalar = readUniform(f32, uniforms, uniform_sizes, 0) orelse 1.0;
                const len = @min(a.len, r.len);
                for (0..len) |i| {
                    r[i] = scalar * a[i];
                }
                return true;
            },
            .matrix_multiply => {
                if (buffers.len < 3) return false;
                const a = bufSliceF32(buffers[0]) orelse return false;
                const b = bufSliceF32(buffers[1]) orelse return false;
                const c = bufSliceMutF32(buffers[2]) orelse return false;
                // Dimensions encoded in global_size: [0]=n, [1]=m, [2]=k
                const m: usize = global_size[1];
                const n: usize = global_size[0];
                const k: usize = global_size[2];
                std_gpu_kernels.matrixMulCpu(a, b, c, m, n, k);
                return true;
            },
            .reduce_sum => {
                if (buffers.len < 2) return false;
                const input = bufSliceF32(buffers[0]) orelse return false;
                const result = bufSliceMutF32(buffers[1]) orelse return false;
                var sum: f32 = 0;
                for (input) |v| sum += v;
                if (result.len > 0) result[0] = sum;
                return true;
            },
            .reduce_max => {
                if (buffers.len < 2) return false;
                const input = bufSliceF32(buffers[0]) orelse return false;
                const result = bufSliceMutF32(buffers[1]) orelse return false;
                if (input.len == 0) return false;
                var max_val: f32 = input[0];
                for (input[1..]) |v| {
                    if (v > max_val) max_val = v;
                }
                if (result.len > 0) result[0] = max_val;
                return true;
            },
            .relu => {
                if (buffers.len < 2) return false;
                const input = bufSliceF32(buffers[0]) orelse return false;
                const output = bufSliceMutF32(buffers[1]) orelse return false;
                const len = @min(input.len, output.len);
                for (0..len) |i| {
                    output[i] = @max(input[i], 0.0);
                }
                return true;
            },
            .sigmoid => {
                if (buffers.len < 2) return false;
                const input = bufSliceF32(buffers[0]) orelse return false;
                const output = bufSliceMutF32(buffers[1]) orelse return false;
                const len = @min(input.len, output.len);
                for (0..len) |i| {
                    output[i] = 1.0 / (1.0 + @exp(-input[i]));
                }
                return true;
            },
            .silu => {
                if (buffers.len < 2) return false;
                const input = bufSliceF32(buffers[0]) orelse return false;
                const output = bufSliceMutF32(buffers[1]) orelse return false;
                const len = @min(input.len, output.len);
                for (0..len) |i| {
                    const x = input[i];
                    output[i] = x / (1.0 + @exp(-x));
                }
                return true;
            },
            .softmax => {
                if (buffers.len < 2) return false;
                const input = bufSliceF32(buffers[0]) orelse return false;
                const output = bufSliceMutF32(buffers[1]) orelse return false;
                if (input.len == 0) return false;
                const len = @min(input.len, output.len);
                // Find max for numerical stability
                var max_val: f32 = input[0];
                for (input[1..]) |v| {
                    if (v > max_val) max_val = v;
                }
                // exp(x - max) and sum
                var sum: f32 = 0;
                for (0..len) |i| {
                    output[i] = @exp(input[i] - max_val);
                    sum += output[i];
                }
                // Normalize
                for (0..len) |i| {
                    output[i] /= sum;
                }
                return true;
            },
            .rms_norm => {
                if (buffers.len < 3) return false;
                const input = bufSliceF32(buffers[0]) orelse return false;
                const weight = bufSliceF32(buffers[1]) orelse return false;
                const output = bufSliceMutF32(buffers[2]) orelse return false;
                const rms = readUniform(f32, uniforms, uniform_sizes, 0) orelse 1.0;
                const epsilon = readUniform(f32, uniforms, uniform_sizes, 1) orelse 1e-5;
                const len = @min(input.len, @min(weight.len, output.len));
                const rms_eps = rms + epsilon;
                for (0..len) |i| {
                    output[i] = weight[i] * input[i] / rms_eps;
                }
                return true;
            },
            else => return false,
        }
    }

    // ---- Helper functions ----

    /// Interpret a Buffer's host_data as a const f32 slice.
    fn bufSliceF32(buf: *const Buffer) ?[]const f32 {
        const data = buf.host_data orelse return null;
        if (data.len < @sizeOf(f32)) return null;
        const aligned: []align(@alignOf(f32)) u8 = @alignCast(data);
        const slice = std.mem.bytesAsSlice(f32, aligned);
        return slice;
    }

    /// Interpret a Buffer's host_data as a mutable f32 slice.
    fn bufSliceMutF32(buf: *const Buffer) ?[]f32 {
        const data = buf.host_data orelse return null;
        if (data.len < @sizeOf(f32)) return null;
        const aligned: []align(@alignOf(f32)) u8 = @alignCast(data);
        return std.mem.bytesAsSlice(f32, aligned);
    }

    /// Safely read a typed uniform parameter.
    fn readUniform(
        comptime T: type,
        uniforms: []const *const anyopaque,
        uniform_sizes: []const usize,
        index: usize,
    ) ?T {
        if (index >= uniforms.len) return null;
        if (index < uniform_sizes.len and uniform_sizes[index] != @sizeOf(T)) return null;
        return @as(*const T, @ptrCast(@alignCast(uniforms[index]))).*;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "StdGpuKernelRegistry unsupported kernel returns false" {
    const handled = StdGpuKernelRegistry.executeBuiltin(
        .conv2d,
        &.{},
        .{ 1, 1, 1 },
        &.{},
        &.{},
    );
    try std.testing.expect(!handled);
}

test "StdGpuKernelRegistry insufficient buffers returns false" {
    // vector_add requires 3 buffers; passing 0 should return false
    const handled = StdGpuKernelRegistry.executeBuiltin(
        .vector_add,
        &.{},
        .{ 4, 1, 1 },
        &.{},
        &.{},
    );
    try std.testing.expect(!handled);
}

test "std.gpu device initialization" {
    const allocator = std.testing.allocator;

    var device = try initStdGpuDevice(allocator);
    defer device.deinit();

    // Device should initialize successfully (emulated on non-GPU targets)
    try std.testing.expect(device.is_emulated);
}

test "std.gpu buffer creation and access" {
    const allocator = std.testing.allocator;

    var device = try initStdGpuDevice(allocator);
    defer device.deinit();

    var buffer = try device.createBuffer(.{
        .size = 1024,
        .usage = .{ .storage = true, .copy_dst = true },
    });
    defer buffer.deinit();

    try std.testing.expectEqual(@as(usize, 1024), buffer.size);

    // Test buffer read/write
    const test_data = "Hello, GPU!";
    try buffer.write(0, test_data);

    var read_buffer: [32]u8 = undefined;
    try buffer.read(0, read_buffer[0..test_data.len]);

    try std.testing.expectEqualStrings(test_data, read_buffer[0..test_data.len]);
}

test "std.gpu typed buffer" {
    const allocator = std.testing.allocator;

    var device = try initStdGpuDevice(allocator);
    defer device.deinit();

    var buffer = try device.createTypedBuffer(f32, 4);
    defer buffer.deinit();

    const slice = buffer.asSlice();
    slice[0] = 1.0;
    slice[1] = 2.0;
    slice[2] = 3.0;
    slice[3] = 4.0;

    try std.testing.expectApproxEqRel(@as(f32, 1.0), slice[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 4.0), slice[3], 1e-6);
}

test "std.gpu vector operations" {
    const allocator = std.testing.allocator;

    var device = try initStdGpuDevice(allocator);
    defer device.deinit();

    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var result: [4]f32 = undefined;

    try device.vectorAdd(&a, &b, &result);

    try std.testing.expectApproxEqRel(@as(f32, 6.0), result[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 8.0), result[1], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 10.0), result[2], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 12.0), result[3], 1e-6);
}

test "std.gpu matrix operations" {
    const allocator = std.testing.allocator;

    var device = try initStdGpuDevice(allocator);
    defer device.deinit();

    // 2x3 * 3x2 = 2x2
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var c: [4]f32 = undefined;

    try device.matrixMul(&a, &b, &c, 2, 2, 3);

    try std.testing.expectApproxEqRel(@as(f32, 58.0), c[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 64.0), c[1], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 139.0), c[2], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 154.0), c[3], 1e-6);
}

test "workgroup calculation" {
    // 1D workgroups
    try std.testing.expectEqual(@as(u32, 4), calculateWorkgroups(1000, 256));
    try std.testing.expectEqual(@as(u32, 1), calculateWorkgroups(256, 256));
    try std.testing.expectEqual(@as(u32, 2), calculateWorkgroups(257, 256));

    // 2D workgroups
    const wg = calculateWorkgroups2D(100, 200, 16);
    try std.testing.expectEqual(@as(u32, 13), wg[0]); // ceil(200/16)
    try std.testing.expectEqual(@as(u32, 7), wg[1]); // ceil(100/16)
}
