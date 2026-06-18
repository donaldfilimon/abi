const builtin = @import("builtin");
const std = @import("std");
const sync = @import("../../foundation/sync.zig");

const objc = struct {
    extern fn objc_getClass(name: [*c]const u8) ?*anyopaque;
    extern fn sel_registerName(name: [*c]const u8) ?*anyopaque;
    extern fn objc_msgSend() void;
};

const MTLCreateSystemDefaultDeviceFn = fn () callconv(.c) ?*anyopaque;
const MTLCreateSystemDefaultDevice: ?*const MTLCreateSystemDefaultDeviceFn = if (builtin.target.os.tag == .macos)
    &struct {
        extern fn MTLCreateSystemDefaultDevice() ?*anyopaque;
    }.MTLCreateSystemDefaultDevice
else
    null;

const MTLSize = extern struct {
    width: usize,
    height: usize,
    depth: usize,
};

const MsgSendVoidRetId = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const MsgSendVoidRetVoid = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) void;
const MsgSendIdRetId = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const MsgSendIdErrRetId = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const MsgSendIdIdErrRetId = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const MsgSendPtrUsizeUsizeRetId = *const fn (?*anyopaque, ?*anyopaque, ?*const anyopaque, usize, usize) callconv(.c) ?*anyopaque;
const MsgSendVoidRetPtr = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const MsgSendIdUsizeUsizeRetVoid = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, usize, usize) callconv(.c) void;
const MsgSendMtlSizeMtlSizeRetVoid = *const fn (?*anyopaque, ?*anyopaque, MTLSize, MTLSize) callconv(.c) void;

fn createNSString(allocator: std.mem.Allocator, str: []const u8) !?*anyopaque {
    if (builtin.target.os.tag != .macos) return null;
    const class_nsstring = objc.objc_getClass("NSString");
    const sel_string_with_utf8 = objc.sel_registerName("stringWithUTF8String:");
    const MsgSendCstrRetId = *const fn (?*anyopaque, ?*anyopaque, [*c]const u8) callconv(.c) ?*anyopaque;
    const msg_send_cstr_ret_id = @as(MsgSendCstrRetId, @ptrCast(&objc.objc_msgSend));

    const c_str = try allocator.alloc(u8, str.len + 1);
    defer allocator.free(c_str);
    @memcpy(c_str[0..str.len], str);
    c_str[str.len] = 0;

    return msg_send_cstr_ret_id(class_nsstring, sel_string_with_utf8, c_str.ptr);
}

const MetalContext = struct {
    device: ?*anyopaque = null,
    queue: ?*anyopaque = null,
    dot_pipeline: ?*anyopaque = null,
    l2_pipeline: ?*anyopaque = null,
    initialized: bool = false,
    mutex: sync.SpinLock = .{},

    pub fn init(self: *MetalContext, allocator: std.mem.Allocator) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.initialized) return;

        if (builtin.target.os.tag != .macos) {
            return error.NotSupported;
        }

        const device = if (MTLCreateSystemDefaultDevice) |f| f() else null orelse return error.NoMetalDevice;
        self.device = device;

        const sel_newCommandQueue = objc.sel_registerName("newCommandQueue");
        const msg_send_void_ret_id = @as(MsgSendVoidRetId, @ptrCast(&objc.objc_msgSend));
        self.queue = msg_send_void_ret_id(device, sel_newCommandQueue) orelse return error.CreateQueueFailed;

        const source =
            \\#include <metal_stdlib>
            \\using namespace metal;
            \\
            \\kernel void dot_kernel(
            \\    device const float* a [[buffer(0)]],
            \\    device const float* b [[buffer(1)]],
            \\    device float* result [[buffer(2)]],
            \\    uint id [[thread_position_in_grid]]
            \\) {
            \\    result[id] = a[id] * b[id];
            \\}
            \\
            \\kernel void l2_kernel(
            \\    device const float* a [[buffer(0)]],
            \\    device const float* b [[buffer(1)]],
            \\    device float* result [[buffer(2)]],
            \\    uint id [[thread_position_in_grid]]
            \\) {
            \\    float diff = a[id] - b[id];
            \\    result[id] = diff * diff;
            \\}
        ;

        const source_nsstring = try createNSString(allocator, source) orelse return error.CreateStringFailed;

        const sel_newLibraryWithSource = objc.sel_registerName("newLibraryWithSource:options:error:");
        const msg_send_id_id_err_ret_id = @as(MsgSendIdIdErrRetId, @ptrCast(&objc.objc_msgSend));

        var err: ?*anyopaque = null;
        const library = msg_send_id_id_err_ret_id(device, sel_newLibraryWithSource, source_nsstring, null, @ptrCast(&err));
        if (library == null) {
            if (err) |err_obj| {
                const sel_localized_description = objc.sel_registerName("localizedDescription");
                const desc_nsstring = msg_send_void_ret_id(err_obj, sel_localized_description);
                if (desc_nsstring) |desc| {
                    const sel_utf8_string = objc.sel_registerName("UTF8String");
                    const MsgSendVoidRetCstr = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) [*c]const u8;
                    const msg_send_void_ret_cstr = @as(MsgSendVoidRetCstr, @ptrCast(&objc.objc_msgSend));
                    const cstr = msg_send_void_ret_cstr(desc, sel_utf8_string);
                    std.log.err("Metal MSL compilation failed: {s}", .{cstr});
                }
            }
            return error.MetalCompilationFailed;
        }

        const sel_newFunctionWithName = objc.sel_registerName("newFunctionWithName:");
        const msg_send_id_ret_id = @as(MsgSendIdRetId, @ptrCast(&objc.objc_msgSend));

        const dot_func_name = try createNSString(allocator, "dot_kernel") orelse return error.CreateStringFailed;
        const dot_func = msg_send_id_ret_id(library, sel_newFunctionWithName, dot_func_name) orelse return error.FunctionNotFound;

        const l2_func_name = try createNSString(allocator, "l2_kernel") orelse return error.CreateStringFailed;
        const l2_func = msg_send_id_ret_id(library, sel_newFunctionWithName, l2_func_name) orelse return error.FunctionNotFound;

        const sel_newComputePipelineState = objc.sel_registerName("newComputePipelineStateWithFunction:error:");
        const msg_send_id_err_ret_id = @as(MsgSendIdErrRetId, @ptrCast(&objc.objc_msgSend));

        err = null;
        self.dot_pipeline = msg_send_id_err_ret_id(device, sel_newComputePipelineState, dot_func, @ptrCast(&err));
        if (self.dot_pipeline == null) return error.CreatePipelineStateFailed;

        err = null;
        self.l2_pipeline = msg_send_id_err_ret_id(device, sel_newComputePipelineState, l2_func, @ptrCast(&err));
        if (self.l2_pipeline == null) return error.CreatePipelineStateFailed;

        // Release one-time init temporaries; the pipeline states and queue stay retained.
        const sel_release = objc.sel_registerName("release");
        const msg_send_void_ret_void = @as(MsgSendVoidRetVoid, @ptrCast(&objc.objc_msgSend));
        msg_send_void_ret_void(dot_func, sel_release);
        msg_send_void_ret_void(l2_func, sel_release);
        msg_send_void_ret_void(library, sel_release);
        msg_send_void_ret_void(dot_func_name, sel_release);
        msg_send_void_ret_void(l2_func_name, sel_release);
        msg_send_void_ret_void(source_nsstring, sel_release);

        self.initialized = true;
    }

    pub fn runKernel(self: *MetalContext, pipeline: ?*anyopaque, len: usize, a: []const f32, b: []const f32, res: []f32) !void {
        if (!self.initialized) return error.NotInitialized;

        const msg_send_void_ret_id = @as(MsgSendVoidRetId, @ptrCast(&objc.objc_msgSend));
        const msg_send_void_ret_void = @as(MsgSendVoidRetVoid, @ptrCast(&objc.objc_msgSend));
        const msg_send_id_ret_id = @as(MsgSendIdRetId, @ptrCast(&objc.objc_msgSend));
        const msg_send_ptr_usize_usize_ret_id = @as(MsgSendPtrUsizeUsizeRetId, @ptrCast(&objc.objc_msgSend));
        const msg_send_id_usize_usize_ret_void = @as(MsgSendIdUsizeUsizeRetVoid, @ptrCast(&objc.objc_msgSend));
        const msg_send_mtlsize_mtlsize_ret_void = @as(MsgSendMtlSizeMtlSizeRetVoid, @ptrCast(&objc.objc_msgSend));
        const msg_send_void_ret_ptr = @as(MsgSendVoidRetPtr, @ptrCast(&objc.objc_msgSend));

        const sel_commandBuffer = objc.sel_registerName("commandBuffer");
        const sel_computeCommandEncoder = objc.sel_registerName("computeCommandEncoder");
        const sel_setComputePipelineState = objc.sel_registerName("setComputePipelineState:");
        const sel_newBufferWithBytes = objc.sel_registerName("newBufferWithBytes:length:options:");
        const sel_setBufferOffsetAtIndex = objc.sel_registerName("setBuffer:offset:atIndex:");
        const sel_dispatch = objc.sel_registerName("dispatchThreads:threadsPerThreadgroup:");
        const sel_endEncoding = objc.sel_registerName("endEncoding");
        const sel_commit = objc.sel_registerName("commit");
        const sel_waitUntilCompleted = objc.sel_registerName("waitUntilCompleted");
        const sel_contents = objc.sel_registerName("contents");

        const byte_len = len * @sizeOf(f32);
        const buffer_a = msg_send_ptr_usize_usize_ret_id(self.device, sel_newBufferWithBytes, a.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        const buffer_b = msg_send_ptr_usize_usize_ret_id(self.device, sel_newBufferWithBytes, b.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        const buffer_res = msg_send_ptr_usize_usize_ret_id(self.device, sel_newBufferWithBytes, res.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;

        const cmd_buf = msg_send_void_ret_id(self.queue, sel_commandBuffer) orelse return error.CommandBufferCreationFailed;
        const encoder = msg_send_void_ret_id(cmd_buf, sel_computeCommandEncoder) orelse return error.EncoderCreationFailed;

        _ = msg_send_id_ret_id(encoder, sel_setComputePipelineState, pipeline);
        msg_send_id_usize_usize_ret_void(encoder, sel_setBufferOffsetAtIndex, buffer_a, 0, 0);
        msg_send_id_usize_usize_ret_void(encoder, sel_setBufferOffsetAtIndex, buffer_b, 0, 1);
        msg_send_id_usize_usize_ret_void(encoder, sel_setBufferOffsetAtIndex, buffer_res, 0, 2);

        const local_size = @min(len, 256);
        const threadgroups = MTLSize{ .width = len, .height = 1, .depth = 1 };
        const threads_per_threadgroup = MTLSize{ .width = local_size, .height = 1, .depth = 1 };

        msg_send_mtlsize_mtlsize_ret_void(encoder, sel_dispatch, threadgroups, threads_per_threadgroup);
        msg_send_void_ret_void(encoder, sel_endEncoding);
        msg_send_void_ret_void(cmd_buf, sel_commit);
        msg_send_void_ret_void(cmd_buf, sel_waitUntilCompleted);

        const res_ptr = msg_send_void_ret_ptr(buffer_res, sel_contents) orelse return error.ReadBufferFailed;
        const res_slice = @as([*]f32, @ptrCast(@alignCast(res_ptr)))[0..len];
        @memcpy(res, res_slice);

        const sel_release = objc.sel_registerName("release");
        msg_send_void_ret_void(buffer_a, sel_release);
        msg_send_void_ret_void(buffer_b, sel_release);
        msg_send_void_ret_void(buffer_res, sel_release);
    }
};

pub var g_metal_context = MetalContext{};
