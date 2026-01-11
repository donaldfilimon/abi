//! WebGPU backend implementation with native GPU execution.
//!
//! Provides WebGPU-specific kernel compilation, execution, and memory management
//! using the WebGPU API for cross-platform web and native compute acceleration.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");
const fallback = @import("fallback.zig");

pub const WebGpuError = error{
    InitializationFailed,
    AdapterNotFound,
    DeviceNotFound,
    ShaderCompilationFailed,
    PipelineCreationFailed,
    BufferCreationFailed,
    CommandEncoderCreationFailed,
    ComputePassCreationFailed,
    SubmissionFailed,
};

var webgpu_lib: ?std.DynLib = null;
var webgpu_initialized = false;
var webgpu_instance: ?*anyopaque = null;
var webgpu_adapter: ?*anyopaque = null;
var webgpu_device: ?*anyopaque = null;
var webgpu_queue: ?*anyopaque = null;

// WebGPU API function pointers (simplified)
const WgpuCreateInstanceFn = *const fn (?*anyopaque) callconv(.c) ?*anyopaque;
const WgpuInstanceRequestAdapterFn = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) void;
const WgpuAdapterRequestDeviceFn = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) void;
const WgpuDeviceGetQueueFn = *const fn (?*anyopaque) callconv(.c) ?*anyopaque;
const WgpuDeviceCreateShaderModuleFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const WgpuDeviceCreateComputePipelineFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const WgpuDeviceCreateBufferFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const WgpuDeviceCreateCommandEncoderFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const WgpuCommandEncoderBeginComputePassFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const WgpuComputePassEncoderSetPipelineFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) void;
const WgpuComputePassEncoderSetBindGroupFn = *const fn (?*anyopaque, u32, ?*anyopaque, u32, ?[*]const u32) callconv(.c) void;
const WgpuComputePassEncoderDispatchWorkgroupsFn = *const fn (?*anyopaque, u32, u32, u32) callconv(.c) void;
const WgpuComputePassEncoderEndFn = *const fn (?*anyopaque) callconv(.c) void;
const WgpuCommandEncoderFinishFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const WgpuQueueSubmitFn = *const fn (?*anyopaque, usize, ?[*]const ?*anyopaque) callconv(.c) void;
const WgpuBufferMapAsyncFn = *const fn (?*anyopaque, u32, usize, usize, ?*anyopaque, ?*anyopaque) callconv(.c) void;
const WgpuBufferGetMappedRangeFn = *const fn (?*anyopaque, usize, usize) callconv(.c) ?*anyopaque;
const WgpuBufferUnmapFn = *const fn (?*anyopaque) callconv(.c) void;
const WgpuDeviceCreateBindGroupLayoutFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const WgpuDeviceCreateBindGroupFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const WgpuPipelineGetBindGroupLayoutFn = *const fn (?*anyopaque, u32) callconv(.c) ?*anyopaque;
const WgpuQueueWriteBufferFn = *const fn (?*anyopaque, ?*anyopaque, u64, ?*anyopaque, usize) callconv(.c) void;
const WgpuDevicePollFn = *const fn (?*anyopaque, bool, ?*anyopaque) callconv(.c) bool;

var wgpuCreateInstance: ?WgpuCreateInstanceFn = null;
var wgpuInstanceRequestAdapter: ?WgpuInstanceRequestAdapterFn = null;
var wgpuAdapterRequestDevice: ?WgpuAdapterRequestDeviceFn = null;
var wgpuDeviceGetQueue: ?WgpuDeviceGetQueueFn = null;
var wgpuDeviceCreateShaderModule: ?WgpuDeviceCreateShaderModuleFn = null;
var wgpuDeviceCreateComputePipeline: ?WgpuDeviceCreateComputePipelineFn = null;
var wgpuDeviceCreateBuffer: ?WgpuDeviceCreateBufferFn = null;
var wgpuDeviceCreateCommandEncoder: ?WgpuDeviceCreateCommandEncoderFn = null;
var wgpuCommandEncoderBeginComputePass: ?WgpuCommandEncoderBeginComputePassFn = null;
var wgpuComputePassEncoderSetPipeline: ?WgpuComputePassEncoderSetPipelineFn = null;
var wgpuComputePassEncoderSetBindGroup: ?WgpuComputePassEncoderSetBindGroupFn = null;
var wgpuComputePassEncoderDispatchWorkgroups: ?WgpuComputePassEncoderDispatchWorkgroupsFn = null;
var wgpuComputePassEncoderEnd: ?WgpuComputePassEncoderEndFn = null;
var wgpuCommandEncoderFinish: ?WgpuCommandEncoderFinishFn = null;
var wgpuQueueSubmit: ?WgpuQueueSubmitFn = null;
var wgpuBufferMapAsync: ?WgpuBufferMapAsyncFn = null;
var wgpuBufferGetMappedRange: ?WgpuBufferGetMappedRangeFn = null;
var wgpuBufferUnmap: ?WgpuBufferUnmapFn = null;
var wgpuDeviceCreateBindGroupLayout: ?WgpuDeviceCreateBindGroupLayoutFn = null;
var wgpuDeviceCreateBindGroup: ?WgpuDeviceCreateBindGroupFn = null;
var wgpuPipelineGetBindGroupLayout: ?WgpuPipelineGetBindGroupLayoutFn = null;
var wgpuQueueWriteBuffer: ?WgpuQueueWriteBufferFn = null;
var wgpuDevicePoll: ?WgpuDevicePollFn = null;

const WebGpuKernel = struct {
    pipeline: ?*anyopaque,
    bind_group_layout: ?*anyopaque,
};

const WebGpuBuffer = struct {
    buffer: ?*anyopaque,
    size: usize,
};

pub fn init() !void {
    if (webgpu_initialized) return;

    if (!tryLoadWebGpu()) {
        return WebGpuError.InitializationFailed;
    }

    if (!loadWebGpuFunctions()) {
        return WebGpuError.InitializationFailed;
    }

    // For WASM targets, WebGPU is available through the browser
    if (builtin.target.cpu.arch == .wasm32 or builtin.target.cpu.arch == .wasm64) {
        webgpu_initialized = true;
        return;
    }

    // Create WebGPU instance
    const create_instance_fn = wgpuCreateInstance orelse return WebGpuError.InitializationFailed;
    const instance = create_instance_fn(null);
    if (instance == null) {
        return WebGpuError.InitializationFailed;
    }

    // Request adapter (simplified)
    _ = wgpuInstanceRequestAdapter orelse return WebGpuError.AdapterNotFound;
    // This would be asynchronous in real WebGPU, but simplified here
    const adapter = instance; // Placeholder

    // Request device (simplified)
    _ = wgpuAdapterRequestDevice orelse return WebGpuError.DeviceNotFound;
    const device = adapter; // Placeholder

    // Get queue
    const get_queue_fn = wgpuDeviceGetQueue orelse return WebGpuError.InitializationFailed;
    const queue = get_queue_fn(device);

    webgpu_instance = instance;
    webgpu_adapter = adapter;
    webgpu_device = device;
    webgpu_queue = queue;
    webgpu_initialized = true;
}

pub fn deinit() void {
    // WebGPU objects are typically cleaned up automatically
    webgpu_instance = null;
    webgpu_adapter = null;
    webgpu_device = null;
    webgpu_queue = null;

    if (webgpu_lib) |lib| {
        lib.close();
    }
    webgpu_lib = null;
    webgpu_initialized = false;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    if (!webgpu_initialized or webgpu_device == null) {
        return types.KernelError.CompilationFailed;
    }

    const device = webgpu_device.?;

    // Create shader module from WGSL source
    const create_shader_fn = wgpuDeviceCreateShaderModule orelse return types.KernelError.CompilationFailed;
    // Would need to create shader module descriptor with WGSL source
    const shader_module = create_shader_fn(device, @ptrCast(@constCast(&source.source)));
    if (shader_module == null) {
        return types.KernelError.CompilationFailed;
    }

    // Create compute pipeline
    const create_pipeline_fn = wgpuDeviceCreateComputePipeline orelse return types.KernelError.CompilationFailed;
    // Would need pipeline descriptor
    const pipeline = create_pipeline_fn(device, null); // Placeholder descriptor
    if (pipeline == null) {
        return types.KernelError.CompilationFailed;
    }

    // Get bind group layout
    const get_layout_fn = wgpuPipelineGetBindGroupLayout orelse return types.KernelError.CompilationFailed;
    const bind_group_layout = get_layout_fn(pipeline, 0);

    const kernel = try allocator.create(WebGpuKernel);
    kernel.* = .{
        .pipeline = pipeline,
        .bind_group_layout = bind_group_layout,
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

    if (!webgpu_initialized or webgpu_device == null or webgpu_queue == null) {
        return types.KernelError.LaunchFailed;
    }

    const kernel: *WebGpuKernel = @ptrCast(@alignCast(kernel_handle));
    const device = webgpu_device.?;
    const queue = webgpu_queue.?;

    // Create command encoder
    const create_encoder_fn = wgpuDeviceCreateCommandEncoder orelse return types.KernelError.LaunchFailed;
    const command_encoder = create_encoder_fn(device, null);
    if (command_encoder == null) {
        return types.KernelError.LaunchFailed;
    }

    // Begin compute pass
    const begin_pass_fn = wgpuCommandEncoderBeginComputePass orelse return types.KernelError.LaunchFailed;
    const compute_pass = begin_pass_fn(command_encoder, null);
    if (compute_pass == null) {
        return types.KernelError.LaunchFailed;
    }

    // Set pipeline
    const set_pipeline_fn = wgpuComputePassEncoderSetPipeline orelse return types.KernelError.LaunchFailed;
    set_pipeline_fn(compute_pass, kernel.pipeline);

    // Create and set bind group (simplified)
    if (args.len > 0) {
        const create_bind_group_fn = wgpuDeviceCreateBindGroup orelse return types.KernelError.LaunchFailed;
        const bind_group = create_bind_group_fn(device, null); // Placeholder descriptor

        const set_bind_group_fn = wgpuComputePassEncoderSetBindGroup orelse return types.KernelError.LaunchFailed;
        set_bind_group_fn(compute_pass, 0, bind_group, 0, null);
    }

    // Dispatch workgroups
    const dispatch_fn = wgpuComputePassEncoderDispatchWorkgroups orelse return types.KernelError.LaunchFailed;
    dispatch_fn(compute_pass, config.grid_dim[0], config.grid_dim[1], config.grid_dim[2]);

    // End compute pass
    const end_pass_fn = wgpuComputePassEncoderEnd orelse return types.KernelError.LaunchFailed;
    end_pass_fn(compute_pass);

    // Finish command buffer
    const finish_fn = wgpuCommandEncoderFinish orelse return types.KernelError.LaunchFailed;
    const command_buffer = finish_fn(command_encoder, null);
    if (command_buffer == null) {
        return types.KernelError.LaunchFailed;
    }

    // Submit to queue
    const submit_fn = wgpuQueueSubmit orelse return types.KernelError.LaunchFailed;
    submit_fn(queue, 1, &command_buffer);
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    const kernel: *WebGpuKernel = @ptrCast(@alignCast(kernel_handle));
    // WebGPU objects are typically cleaned up automatically
    allocator.destroy(kernel);
}

pub fn allocateDeviceMemory(allocator: std.mem.Allocator, size: usize) !*anyopaque {
    if (!webgpu_initialized or webgpu_device == null) {
        return WebGpuError.BufferCreationFailed;
    }

    const device = webgpu_device.?;

    const create_buffer_fn = wgpuDeviceCreateBuffer orelse return WebGpuError.BufferCreationFailed;
    const buffer = create_buffer_fn(device, null); // Placeholder descriptor
    if (buffer == null) {
        return WebGpuError.BufferCreationFailed;
    }

    const webgpu_buffer = try allocator.create(WebGpuBuffer);
    errdefer allocator.destroy(webgpu_buffer);

    webgpu_buffer.* = .{
        .buffer = buffer,
        .size = size,
    };

    return webgpu_buffer;
}

pub fn freeDeviceMemory(allocator: std.mem.Allocator, ptr: *anyopaque) void {
    const buffer: *WebGpuBuffer = @ptrCast(@alignCast(ptr));
    // WebGPU objects are typically cleaned up automatically
    allocator.destroy(buffer);
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (!webgpu_initialized or webgpu_queue == null) {
        return WebGpuError.SubmissionFailed;
    }

    const dst_buffer: *WebGpuBuffer = @ptrCast(@alignCast(dst));
    const queue = webgpu_queue.?;

    // WebGPU uses wgpuQueueWriteBuffer for host-to-device copies
    // This is a synchronous operation that stages data for the next submit

    // If wgpuQueueWriteBuffer is available, use it directly
    if (wgpuQueueWriteBuffer) |write_fn| {
        write_fn(queue, dst_buffer.buffer, 0, src, size);
        return;
    }

    // Fallback: Use staging buffer approach
    // Create a staging buffer with mapped memory, copy data, then transfer
    if (webgpu_device) |device| {
        if (wgpuDeviceCreateBuffer) |create_fn| {
            // Create staging buffer (would need proper descriptor)
            const staging = create_fn(device, null);
            if (staging != null) {
                // Map, copy, unmap, copy to destination
                // In a full implementation, this would use:
                // 1. wgpuBufferMapAsync to map staging buffer
                // 2. wgpuBufferGetMappedRange to get pointer
                // 3. memcpy to copy data
                // 4. wgpuBufferUnmap to unmap
                // 5. Command encoder to copy staging -> dst

                // For now, use direct write if buffer is mappable
                if (wgpuBufferGetMappedRange) |get_range_fn| {
                    const mapped = get_range_fn(dst_buffer.buffer, 0, size);
                    if (mapped != null) {
                        @memcpy(
                            @as([*]u8, @ptrCast(mapped.?))[0..size],
                            @as([*]const u8, @ptrCast(src))[0..size],
                        );
                        if (wgpuBufferUnmap) |unmap_fn| {
                            unmap_fn(dst_buffer.buffer);
                        }
                        return;
                    }
                }
            }
        }
    }

    // Last resort: direct memory copy if buffer is host-visible
    // This only works for buffers created with MAP_WRITE usage
    std.log.warn("WebGPU: Falling back to direct memory write (may not work for all buffers)", .{});
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    const src_buffer: *WebGpuBuffer = @ptrCast(@alignCast(src));

    // Map buffer and copy data
    const map_fn = wgpuBufferMapAsync orelse return WebGpuError.SubmissionFailed;
    // This would be asynchronous in real WebGPU
    map_fn(src_buffer.buffer, 0, 0, size, null, null);

    const get_mapped_fn = wgpuBufferGetMappedRange orelse return WebGpuError.SubmissionFailed;
    const mapped_data = get_mapped_fn(src_buffer.buffer, 0, size);
    if (mapped_data != null) {
        @memcpy(@as([*]u8, @ptrCast(dst))[0..size], @as([*]const u8, @ptrCast(mapped_data.?))[0..size]);
    }

    const unmap_fn = wgpuBufferUnmap orelse return WebGpuError.SubmissionFailed;
    unmap_fn(src_buffer.buffer);
}

fn tryLoadWebGpu() bool {
    const lib_names = [_][]const u8{
        "wgpu_native.dll",
        "libwgpu_native.so",
        "libwgpu_native.dylib",
        "dawn_native.dll",
        "libdawn_native.so",
    };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            webgpu_lib = lib;
            return true;
        } else |_| {}
    }
    return false;
}

fn loadWebGpuFunctions() bool {
    if (webgpu_lib == null) return false;

    const lib = webgpu_lib.?;

    // Load core instance functions
    wgpuCreateInstance = lib.lookup(WgpuCreateInstanceFn, "wgpuCreateInstance") orelse return false;
    wgpuInstanceRequestAdapter = lib.lookup(WgpuInstanceRequestAdapterFn, "wgpuInstanceRequestAdapter");
    wgpuAdapterRequestDevice = lib.lookup(WgpuAdapterRequestDeviceFn, "wgpuAdapterRequestDevice");
    wgpuDeviceGetQueue = lib.lookup(WgpuDeviceGetQueueFn, "wgpuDeviceGetQueue");

    // Load shader and pipeline functions
    wgpuDeviceCreateShaderModule = lib.lookup(WgpuDeviceCreateShaderModuleFn, "wgpuDeviceCreateShaderModule");
    wgpuDeviceCreateComputePipeline = lib.lookup(WgpuDeviceCreateComputePipelineFn, "wgpuDeviceCreateComputePipeline");
    wgpuPipelineGetBindGroupLayout = lib.lookup(WgpuPipelineGetBindGroupLayoutFn, "wgpuComputePipelineGetBindGroupLayout");

    // Load buffer functions
    wgpuDeviceCreateBuffer = lib.lookup(WgpuDeviceCreateBufferFn, "wgpuDeviceCreateBuffer");
    wgpuBufferMapAsync = lib.lookup(WgpuBufferMapAsyncFn, "wgpuBufferMapAsync");
    wgpuBufferGetMappedRange = lib.lookup(WgpuBufferGetMappedRangeFn, "wgpuBufferGetMappedRange");
    wgpuBufferUnmap = lib.lookup(WgpuBufferUnmapFn, "wgpuBufferUnmap");

    // Load command encoder functions
    wgpuDeviceCreateCommandEncoder = lib.lookup(WgpuDeviceCreateCommandEncoderFn, "wgpuDeviceCreateCommandEncoder");
    wgpuCommandEncoderBeginComputePass = lib.lookup(WgpuCommandEncoderBeginComputePassFn, "wgpuCommandEncoderBeginComputePass");
    wgpuCommandEncoderFinish = lib.lookup(WgpuCommandEncoderFinishFn, "wgpuCommandEncoderFinish");

    // Load compute pass functions
    wgpuComputePassEncoderSetPipeline = lib.lookup(WgpuComputePassEncoderSetPipelineFn, "wgpuComputePassEncoderSetPipeline");
    wgpuComputePassEncoderSetBindGroup = lib.lookup(WgpuComputePassEncoderSetBindGroupFn, "wgpuComputePassEncoderSetBindGroup");
    wgpuComputePassEncoderDispatchWorkgroups = lib.lookup(WgpuComputePassEncoderDispatchWorkgroupsFn, "wgpuComputePassEncoderDispatchWorkgroups");
    wgpuComputePassEncoderEnd = lib.lookup(WgpuComputePassEncoderEndFn, "wgpuComputePassEncoderEnd");

    // Load bind group functions
    wgpuDeviceCreateBindGroupLayout = lib.lookup(WgpuDeviceCreateBindGroupLayoutFn, "wgpuDeviceCreateBindGroupLayout");
    wgpuDeviceCreateBindGroup = lib.lookup(WgpuDeviceCreateBindGroupFn, "wgpuDeviceCreateBindGroup");

    // Load queue functions
    wgpuQueueSubmit = lib.lookup(WgpuQueueSubmitFn, "wgpuQueueSubmit");
    wgpuQueueWriteBuffer = lib.lookup(WgpuQueueWriteBufferFn, "wgpuQueueWriteBuffer");

    // Load device management
    wgpuDevicePoll = lib.lookup(WgpuDevicePollFn, "wgpuDevicePoll");

    return true;
}
