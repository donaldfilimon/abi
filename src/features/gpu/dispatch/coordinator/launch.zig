//! Kernel Launch/Dispatch Logic
//!
//! Backend launch and cuBLAS execution helpers for the KernelDispatcher.

const std = @import("std");
const build_options = @import("build_options");
const interface = @import("../../interface.zig");
const dispatch_types = @import("../types.zig");
const unified_buffer = @import("../../unified_buffer.zig");
const backend_shared = @import("../../backends/shared.zig");

const DispatchError = dispatch_types.DispatchError;
const CompiledKernelHandle = dispatch_types.CompiledKernelHandle;
const LaunchConfig = dispatch_types.LaunchConfig;
const KernelArgs = dispatch_types.KernelArgs;

// Conditionally import CUDA/cuBLAS for optimized BLAS operations
const cublas = if (build_options.feat_gpu and build_options.gpu_cuda and backend_shared.dynlibSupported)
    @import("../../backends/cuda/cublas.zig")
else
    struct {
        pub const CublasContext = void;
        pub fn isAvailable() bool {
            return false;
        }
    };

/// Maximum dimension size for safe matrix operations (prevents overflow).
pub const MAX_MATRIX_DIM: u32 = 32768; // 32K x 32K max

/// Safely cast u32 to i32 with overflow check.
pub fn safeCastToI32(val: u32) DispatchError!i32 {
    if (val > std.math.maxInt(i32)) {
        return DispatchError.InvalidArguments;
    }
    return @intCast(val);
}

/// Validate matrix dimensions to prevent overflow in stride calculations.
pub fn validateMatrixDimensions(m: u32, n: u32, k: u32) DispatchError!void {
    if (m > MAX_MATRIX_DIM or n > MAX_MATRIX_DIM or k > MAX_MATRIX_DIM) {
        std.log.err("Matrix dimensions exceed safe limit: m={}, n={}, k={} (max={})", .{ m, n, k, MAX_MATRIX_DIM });
        return DispatchError.InvalidArguments;
    }
    // Check for potential overflow in stride calculations (i64 is safe for dimensions up to 32K)
    const max_stride = @as(u64, m) * @as(u64, k);
    if (max_stride > std.math.maxInt(i64)) {
        return DispatchError.InvalidArguments;
    }
}

/// Safely read a uniform parameter with type checking.
pub fn readUniformAs(comptime T: type, args: KernelArgs, index: usize, default: T) T {
    if (index >= args.uniforms.len) return default;
    if (index >= args.uniform_sizes.len) {
        // No size info, use pointer cast but log warning
        std.log.debug("Uniform at index {} has no size info, using type assumption", .{index});
        return @as(*const T, @ptrCast(@alignCast(args.uniforms[index]))).*;
    }
    // Validate size matches expected type
    if (args.uniform_sizes[index] != @sizeOf(T)) {
        std.log.warn("Uniform size mismatch at index {}: expected {} bytes, got {}", .{
            index, @sizeOf(T), args.uniform_sizes[index],
        });
        return default;
    }
    return @as(*const T, @ptrCast(@alignCast(args.uniforms[index]))).*;
}

/// Execute on the actual backend.
pub fn launchOnBackend(
    allocator: std.mem.Allocator,
    backend_interface: ?interface.Backend,
    kernel: CompiledKernelHandle,
    config: LaunchConfig,
    args: KernelArgs,
) DispatchError!void {
    const bi = backend_interface orelse return DispatchError.BackendNotInitialized;
    const handle = kernel.handle orelse return DispatchError.KernelNotFound;

    // Build argument list for backend, allocating device memory as needed
    var arg_ptrs = std.ArrayListUnmanaged(*anyopaque).empty;
    defer arg_ptrs.deinit(allocator);

    for (args.buffers) |buf| {
        var device_ptr: *anyopaque = undefined;

        // Allocate device memory if buffer doesn't have one
        if (!buf.hasDeviceHandle()) {
            const size = buf.getSize();
            device_ptr = bi.allocate(size, .{}) catch {
                std.log.debug("Failed to allocate {} bytes device memory", .{size});
                return DispatchError.OutOfMemory;
            };

            // Copy host data to device
            if (buf.getHostBytes()) |host_data| {
                bi.copyToDevice(device_ptr, host_data) catch {
                    bi.free(device_ptr);
                    return DispatchError.BufferNotReady;
                };
            }

            // Store device handle in buffer for reuse
            buf.setDeviceHandle(device_ptr, bi);
            buf.clearHostDirty();
        } else {
            device_ptr = buf.getDevicePtr() catch return DispatchError.BufferNotReady;
        }

        arg_ptrs.append(allocator, device_ptr) catch return DispatchError.OutOfMemory;
    }

    // Create launch config for backend
    const grid = config.gridDimensions();
    const local = config.local_size orelse kernel.workgroup_size;
    const backend_config = interface.LaunchConfig{
        .grid_x = grid[0],
        .grid_y = grid[1],
        .grid_z = grid[2],
        .block_x = local[0],
        .block_y = local[1],
        .block_z = local[2],
        .shared_memory = config.shared_memory,
    };

    // Launch kernel
    bi.launchKernel(
        handle,
        backend_config,
        arg_ptrs.items,
    ) catch return DispatchError.ExecutionFailed;

    // Synchronize to ensure completion
    bi.synchronize() catch |err| {
        std.log.warn("GPU synchronization failed: {t}", .{err});
        return DispatchError.ExecutionFailed;
    };
}
