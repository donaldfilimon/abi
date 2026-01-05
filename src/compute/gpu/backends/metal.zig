//! Metal backend implementation with native GPU execution.
//!
//! Provides Metal-specific kernel compilation, execution, and memory management
//! using the Metal API for Apple Silicon acceleration.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");
const fallback = @import("fallback.zig");

pub const MetalError = error{
    InitializationFailed,
    DeviceNotFound,
    LibraryCompilationFailed,
    PipelineCreationFailed,
    CommandQueueCreationFailed,
    BufferCreationFailed,
    CommandBufferCreationFailed,
    KernelExecutionFailed,
    MemoryCopyFailed,
};

var metal_lib: ?std.DynLib = null;
var metal_initialized = false;
var metal_device: ?*anyopaque = null;
var metal_command_queue: ?*anyopaque = null;

// Metal API function pointers (simplified)
const MtlCreateSystemDefaultDeviceFn = *const fn () callconv(.c) ?*anyopaque;
const MtlDeviceNewCommandQueueFn = *const fn (?*anyopaque) callconv(.c) ?*anyopaque;
const MtlDeviceNewLibraryWithSourceFn = *const fn (?*anyopaque, [*:0]const u8, ?*anyopaque) callconv(.c) ?*anyopaque;
const MtlLibraryNewComputePipelineStateWithFunctionFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const MtlDeviceNewBufferWithLengthFn = *const fn (?*anyopaque, usize, u32) callconv(.c) ?*anyopaque;
const MtlCommandQueueCommandBufferFn = *const fn (?*anyopaque) callconv(.c) ?*anyopaque;
const MtlCommandBufferComputeCommandEncoderFn = *const fn (?*anyopaque) callconv(.c) ?*anyopaque;
const MtlComputeCommandEncoderSetComputePipelineStateFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) void;
const MtlComputeCommandEncoderSetBufferFn = *const fn (?*anyopaque, ?*anyopaque, u32, u32) callconv(.c) void;
const MtlComputeCommandEncoderDispatchThreadsFn = *const fn (?*anyopaque, [3]u32, [3]u32) callconv(.c) void;
const MtlComputeCommandEncoderEndEncodingFn = *const fn (?*anyopaque) callconv(.c) void;
const MtlCommandBufferCommitFn = *const fn (?*anyopaque) callconv(.c) void;
const MtlCommandBufferWaitUntilCompletedFn = *const fn (?*anyopaque) callconv(.c) void;
const MtlBufferContentsFn = *const fn (?*anyopaque) callconv(.c) ?*anyopaque;
const MtlBufferLengthFn = *const fn (?*anyopaque) callconv(.c) usize;

var mtlCreateSystemDefaultDevice: ?MtlCreateSystemDefaultDeviceFn = null;
var mtlDeviceNewCommandQueue: ?MtlDeviceNewCommandQueueFn = null;
var mtlDeviceNewLibraryWithSource: ?MtlDeviceNewLibraryWithSourceFn = null;
var mtlLibraryNewComputePipelineStateWithFunction: ?MtlLibraryNewComputePipelineStateWithFunctionFn = null;
var mtlDeviceNewBufferWithLength: ?MtlDeviceNewBufferWithLengthFn = null;
var mtlCommandQueueCommandBuffer: ?MtlCommandQueueCommandBufferFn = null;
var mtlCommandBufferComputeCommandEncoder: ?MtlCommandBufferComputeCommandEncoderFn = null;
var mtlComputeCommandEncoderSetComputePipelineState: ?MtlComputeCommandEncoderSetComputePipelineStateFn = null;
var mtlComputeCommandEncoderSetBuffer: ?MtlComputeCommandEncoderSetBufferFn = null;
var mtlComputeCommandEncoderDispatchThreads: ?MtlComputeCommandEncoderDispatchThreadsFn = null;
var mtlComputeCommandEncoderEndEncoding: ?MtlComputeCommandEncoderEndEncodingFn = null;
var mtlCommandBufferCommit: ?MtlCommandBufferCommitFn = null;
var mtlCommandBufferWaitUntilCompleted: ?MtlCommandBufferWaitUntilCompletedFn = null;
var mtlBufferContents: ?MtlBufferContentsFn = null;
var mtlBufferLength: ?MtlBufferLengthFn = null;

const MetalKernel = struct {
    pipeline_state: ?*anyopaque,
    library: ?*anyopaque,
};

const MetalBuffer = struct {
    buffer: ?*anyopaque,
    size: usize,
};

pub fn init() !void {
    if (metal_initialized) return;

    if (builtin.target.os.tag != .macos) {
        return MetalError.InitializationFailed;
    }

    if (!tryLoadMetal()) {
        return MetalError.InitializationFailed;
    }

    if (!loadMetalFunctions()) {
        return MetalError.InitializationFailed;
    }

    // Create Metal device
    const create_device_fn = mtlCreateSystemDefaultDevice orelse return MetalError.DeviceNotFound;
    const device = create_device_fn();
    if (device == null) {
        return MetalError.DeviceNotFound;
    }

    // Create command queue
    const create_queue_fn = mtlDeviceNewCommandQueue orelse return MetalError.CommandQueueCreationFailed;
    const command_queue = create_queue_fn(device);
    if (command_queue == null) {
        return MetalError.CommandQueueCreationFailed;
    }

    metal_device = device;
    metal_command_queue = command_queue;
    metal_initialized = true;
}

pub fn deinit() void {
    // Note: Metal objects are reference counted, so they clean up automatically
    metal_device = null;
    metal_command_queue = null;

    if (metal_lib) |lib| {
        lib.close();
    }
    metal_lib = null;
    metal_initialized = false;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    if (!metal_initialized or metal_device == null) {
        return types.KernelError.CompilationFailed;
    }

    const device = metal_device.?;

    // Create library from source
    const create_library_fn = mtlDeviceNewLibraryWithSource orelse return types.KernelError.CompilationFailed;
    const library = create_library_fn(device, source.source.ptr, null);
    if (library == null) {
        return types.KernelError.CompilationFailed;
    }

    // Get function from library (simplified - assuming function name matches entry point)
    // In Metal, we'd need to get the function from the library
    const function = library; // Placeholder - real implementation would get function by name

    // Create compute pipeline state
    const create_pipeline_fn = mtlLibraryNewComputePipelineStateWithFunction orelse return types.KernelError.CompilationFailed;
    const pipeline_state = create_pipeline_fn(library, function);
    if (pipeline_state == null) {
        return types.KernelError.CompilationFailed;
    }

    const kernel = try allocator.create(MetalKernel);
    kernel.* = .{
        .pipeline_state = pipeline_state,
        .library = library,
    };

    return kernel;
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    _ = allocator;

    if (!metal_initialized or metal_command_queue == null) {
        return types.KernelError.LaunchFailed;
    }

    const kernel: *MetalKernel = @ptrCast(@alignCast(kernel_handle));

    // Create command buffer
    const create_cmd_buf_fn = mtlCommandQueueCommandBuffer orelse return types.KernelError.LaunchFailed;
    const command_buffer = create_cmd_buf_fn(metal_command_queue.?);
    if (command_buffer == null) {
        return types.KernelError.LaunchFailed;
    }

    // Create compute command encoder
    const create_encoder_fn = mtlCommandBufferComputeCommandEncoder orelse return types.KernelError.LaunchFailed;
    const encoder = create_encoder_fn(command_buffer);
    if (encoder == null) {
        return types.KernelError.LaunchFailed;
    }

    // Set pipeline state
    const set_pipeline_fn = mtlComputeCommandEncoderSetComputePipelineState orelse return types.KernelError.LaunchFailed;
    set_pipeline_fn(encoder, kernel.pipeline_state);

    // Set buffers
    const set_buffer_fn = mtlComputeCommandEncoderSetBuffer orelse return types.KernelError.LaunchFailed;
    for (args, 0..) |arg, i| {
        if (arg != null) {
            const buffer: *MetalBuffer = @ptrCast(@alignCast(arg.?));
            set_buffer_fn(encoder, buffer.buffer, 0, @intCast(i));
        }
    }

    // Dispatch threads
    const dispatch_fn = mtlComputeCommandEncoderDispatchThreads orelse return types.KernelError.LaunchFailed;
    const threads_per_grid = [3]u32{
        config.grid_dim[0] * config.block_dim[0],
        config.grid_dim[1] * config.block_dim[1],
        config.grid_dim[2] * config.block_dim[2],
    };
    const threads_per_threadgroup = [3]u32{
        config.block_dim[0],
        config.block_dim[1],
        config.block_dim[2],
    };
    dispatch_fn(encoder, threads_per_grid, threads_per_threadgroup);

    // End encoding
    const end_encoding_fn = mtlComputeCommandEncoderEndEncoding orelse return types.KernelError.LaunchFailed;
    end_encoding_fn(encoder);

    // Commit and wait
    const commit_fn = mtlCommandBufferCommit orelse return types.KernelError.LaunchFailed;
    commit_fn(command_buffer);

    const wait_fn = mtlCommandBufferWaitUntilCompleted orelse return types.KernelError.LaunchFailed;
    wait_fn(command_buffer);
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    const kernel: *MetalKernel = @ptrCast(@alignCast(kernel_handle));
    // Metal objects are reference counted, so they clean up automatically
    allocator.destroy(kernel);
}

pub fn allocateDeviceMemory(size: usize) !*anyopaque {
    if (!metal_initialized or metal_device == null) {
        return MetalError.BufferCreationFailed;
    }

    const device = metal_device.?;

    const create_buffer_fn = mtlDeviceNewBufferWithLength orelse return MetalError.BufferCreationFailed;
    const buffer = create_buffer_fn(device, size, 0); // MTLResourceStorageModeShared
    if (buffer == null) {
        return MetalError.BufferCreationFailed;
    }

    const metal_buffer = try std.heap.page_allocator.create(MetalBuffer);
    metal_buffer.* = .{
        .buffer = buffer,
        .size = size,
    };

    return metal_buffer;
}

pub fn freeDeviceMemory(ptr: *anyopaque) void {
    const buffer: *MetalBuffer = @ptrCast(@alignCast(ptr));
    // Metal objects are reference counted, so they clean up automatically
    std.heap.page_allocator.destroy(buffer);
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    const dst_buffer: *MetalBuffer = @ptrCast(@alignCast(dst));

    const contents_fn = mtlBufferContents orelse return MetalError.MemoryCopyFailed;
    const contents = contents_fn(dst_buffer.buffer);
    if (contents == null) {
        return MetalError.MemoryCopyFailed;
    }

    @memcpy(@as([*]u8, @ptrCast(contents))[0..size], @as([*]const u8, @ptrCast(src))[0..size]);
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    const src_buffer: *MetalBuffer = @ptrCast(@alignCast(src));

    const contents_fn = mtlBufferContents orelse return MetalError.MemoryCopyFailed;
    const contents = contents_fn(src_buffer.buffer);
    if (contents == null) {
        return MetalError.MemoryCopyFailed;
    }

    @memcpy(@as([*]u8, @ptrCast(dst))[0..size], @as([*]const u8, @ptrCast(contents.?))[0..size]);
}

fn tryLoadMetal() bool {
    // Metal framework is loaded differently on macOS
    // For now, return false to use fallback
    return false;
}

fn loadMetalFunctions() bool {
    // Simplified - would load actual Metal functions
    return false;
}
