//! GPU Dispatch Types
//!
//! Error types, configuration structs, and handles used by the kernel dispatcher.

const std = @import("std");
const backend_mod = @import("../backend.zig");
const device_mod = @import("../device.zig");
const unified_buffer = @import("../unified_buffer.zig");

pub const Backend = backend_mod.Backend;
pub const Device = device_mod.Device;
pub const Buffer = unified_buffer.Buffer;

/// Errors that can occur during kernel dispatch.
pub const DispatchError = error{
    NoBackendAvailable,
    KernelNotFound,
    KernelCompilationFailed,
    InvalidConfiguration,
    BufferNotReady,
    DeviceMismatch,
    OutOfMemory,
    ExecutionFailed,
    UnsupportedOperation,
    BackendNotInitialized,
    InvalidArguments,
    TimerFailed,
    LaunchQueueFull,
    BatchTooLarge,
};

/// Handle to a compiled kernel.
pub const CompiledKernelHandle = struct {
    /// Backend-specific handle.
    handle: ?*anyopaque,
    /// Kernel name for identification.
    name: []const u8,
    /// Backend this kernel was compiled for.
    backend: Backend,
    /// Workgroup size used during compilation.
    workgroup_size: [3]u32,
    /// Number of buffer parameters expected.
    buffer_count: u8,
    /// Number of uniform parameters expected.
    uniform_count: u8,

    pub fn isValid(self: *const CompiledKernelHandle) bool {
        return self.handle != null or self.backend == .stdgpu;
    }
};

/// Configuration for kernel execution.
pub const LaunchConfig = struct {
    /// Global work size (total threads).
    global_size: [3]u32 = .{ 1, 1, 1 },
    /// Local work size (workgroup/block size). null = auto-calculate.
    local_size: ?[3]u32 = null,
    /// Shared memory size in bytes.
    shared_memory: u32 = 0,
    /// Optional stream handle for async execution.
    stream: ?*anyopaque = null,

    /// Calculate grid dimensions from global size and local size.
    pub fn gridDimensions(self: *const LaunchConfig) [3]u32 {
        const local = self.local_size orelse .{ 256, 1, 1 };
        return .{
            (self.global_size[0] + local[0] - 1) / local[0],
            (self.global_size[1] + local[1] - 1) / local[1],
            (self.global_size[2] + local[2] - 1) / local[2],
        };
    }

    /// Create config for 1D kernel execution.
    pub fn for1D(element_count: usize, workgroup_size: u32) LaunchConfig {
        return .{
            .global_size = .{ @intCast(element_count), 1, 1 },
            .local_size = .{ workgroup_size, 1, 1 },
        };
    }

    /// Create config for 2D kernel execution (e.g., matrices).
    pub fn for2D(width: usize, height: usize, tile_x: u32, tile_y: u32) LaunchConfig {
        return .{
            .global_size = .{ @intCast(width), @intCast(height), 1 },
            .local_size = .{ tile_x, tile_y, 1 },
        };
    }
};

/// Arguments for kernel execution.
pub const KernelArgs = struct {
    /// Buffer arguments (device memory pointers).
    buffers: []const *Buffer = &.{},
    /// Uniform arguments (small constant values).
    uniforms: []const *const anyopaque = &.{},
    /// Uniform sizes in bytes for each uniform.
    uniform_sizes: []const usize = &.{},
};

/// Result of kernel execution.
pub const ExecutionResult = struct {
    /// Execution time in nanoseconds.
    execution_time_ns: u64,
    /// Number of elements processed.
    elements_processed: usize,
    /// Bytes transferred (input + output).
    bytes_transferred: usize,
    /// Backend used for execution.
    backend: Backend,
    /// Device ID used.
    device_id: u32,
    /// Whether kernel executed on GPU (vs CPU fallback).
    gpu_executed: bool,

    /// Get throughput in GB/s.
    pub fn throughputGBps(self: *const ExecutionResult) f64 {
        if (self.execution_time_ns == 0) return 0;
        const bytes_per_sec = @as(f64, @floatFromInt(self.bytes_transferred)) /
            (@as(f64, @floatFromInt(self.execution_time_ns)) / 1_000_000_000.0);
        return bytes_per_sec / (1024 * 1024 * 1024);
    }

    /// Get elements per second.
    pub fn elementsPerSecond(self: *const ExecutionResult) f64 {
        if (self.execution_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.elements_processed)) /
            (@as(f64, @floatFromInt(self.execution_time_ns)) / 1_000_000_000.0);
    }
};

/// Queued kernel launch for batching.
pub const QueuedLaunch = struct {
    kernel: *const CompiledKernelHandle,
    config: LaunchConfig,
    args: KernelArgs,
};
