//! GPU Renderer Buffer Operations
//!
//! High-level buffer management operations for the GPU renderer

const std = @import("std");

const config_mod = @import("config.zig");
const buffers = @import("buffers.zig");
const pipelines = @import("pipelines.zig");

const BufferUsage = config_mod.BufferUsage;
const GpuError = config_mod.GpuError;
const Buffer = buffers.Buffer;
const BufferManager = buffers.BufferManager;
const ComputePipeline = pipelines.ComputePipeline;
const RendererStats = pipelines.RendererStats;

const GPURenderer = @import("renderer.zig").GPURenderer;

/// Create a GPU buffer with the specified size and usage
pub fn createBuffer(renderer: *GPURenderer, size: usize, usage: BufferUsage) !u32 {
    if (renderer.buffer_manager == null) return GpuError.InitializationFailed;

    const handle_id = renderer.next_handle_id;
    renderer.next_handle_id += 1;

    const gpu_buffer = try renderer.buffer_manager.?.createBuffer(u8, @as(u64, @intCast(size)), usage);
    const buffer = Buffer.init(gpu_buffer, size, usage, handle_id);

    try renderer.buffers.append(renderer.allocator, buffer);

    // Update stats
    renderer.stats.buffers_created += 1;
    renderer.stats.bytes_current += @as(u64, @intCast(size));
    if (renderer.stats.bytes_current > renderer.stats.bytes_peak) {
        renderer.stats.bytes_peak = renderer.stats.bytes_current;
    }
    return @intCast(handle_id);
}

/// Create a buffer initialized with data
pub fn createBufferWithData(renderer: *GPURenderer, comptime T: type, data: []const T, usage: BufferUsage) !u32 {
    if (renderer.buffer_manager == null) return GpuError.InitializationFailed;

    const handle_id = renderer.next_handle_id;
    renderer.next_handle_id += 1;

    const gpu_buffer = try renderer.buffer_manager.?.createBufferWithData(T, data, usage);
    const size_bytes: usize = data.len * @sizeOf(T);
    const buffer = Buffer.init(gpu_buffer, size_bytes, usage, handle_id);

    try renderer.buffers.append(renderer.allocator, buffer);

    // Update stats
    renderer.stats.buffers_created += 1;
    renderer.stats.bytes_current += @as(u64, @intCast(size_bytes));
    if (renderer.stats.bytes_current > renderer.stats.bytes_peak) {
        renderer.stats.bytes_peak = renderer.stats.bytes_current;
    }
    renderer.stats.bytes_written += @as(u64, @intCast(size_bytes));
    return @intCast(handle_id);
}

/// Destroy a buffer by handle
pub fn destroyBuffer(renderer: *GPURenderer, handle: u32) !void {
    if (renderer.findBufferIndex(handle)) |idx| {
        var buffer = renderer.buffers.items[idx];
        buffer.deinit(renderer.allocator);
        _ = renderer.buffers.orderedRemove(idx);

        // Update stats
        renderer.stats.buffers_destroyed += 1;
        renderer.stats.bytes_current -= @as(u64, @intCast(buffer.size));
    } else {
        return GpuError.HandleNotFound;
    }
}

/// Write data to a GPU buffer
pub fn writeBuffer(renderer: *GPURenderer, handle: u32, data: anytype) !void {
    const buffer = renderer.findBuffer(handle) orelse return GpuError.HandleNotFound;

    const bytes = switch (@typeInfo(@TypeOf(data))) {
        .Pointer => |ptr| blk: {
            if (ptr.size == .Slice) {
                break :blk std.mem.sliceAsBytes(data);
            } else {
                @compileError("writeBuffer expects a slice, got a pointer");
            }
        },
        .Array => std.mem.sliceAsBytes(&data),
        else => @compileError("writeBuffer expects a slice or array"),
    };

    if (bytes.len > buffer.size) return GpuError.BufferOverflow;

    try buffer.gpu_buffer.write(bytes);

    // Update stats
    renderer.stats.bytes_written += @as(u64, @intCast(bytes.len));
}

/// Read data from a GPU buffer
pub fn readBuffer(renderer: *GPURenderer, handle: u32, allocator: std.mem.Allocator) ![]u8 {
    const buffer = renderer.findBuffer(handle) orelse return GpuError.HandleNotFound;

    const result = try allocator.alloc(u8, buffer.size);
    errdefer allocator.free(result);

    try buffer.gpu_buffer.read(result);

    // Update stats
    renderer.stats.bytes_read += @as(u64, @intCast(buffer.size));

    return result;
}

/// Get a typed slice from a buffer
pub fn getBufferSlice(renderer: *GPURenderer, handle: u32, comptime T: type, count: usize) ![]T {
    const buffer = renderer.findBuffer(handle) orelse return GpuError.HandleNotFound;

    const expected_size = count * @sizeOf(T);
    if (expected_size > buffer.size) return GpuError.BufferOverflow;

    return buffer.gpu_buffer.getSlice(T, count);
}

/// Copy data between buffers
pub fn copyBuffer(renderer: *GPURenderer, src_handle: u32, dst_handle: u32) !usize {
    const src_buffer = renderer.findBuffer(src_handle) orelse return GpuError.HandleNotFound;
    const dst_buffer = renderer.findBuffer(dst_handle) orelse return GpuError.HandleNotFound;

    const copy_size = @min(src_buffer.size, dst_buffer.size);

    try src_buffer.gpu_buffer.copyTo(dst_buffer.gpu_buffer, copy_size);

    // Update stats
    renderer.stats.bytes_copied += @as(u64, @intCast(copy_size));

    return copy_size;
}
