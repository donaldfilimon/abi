const builtin = @import("builtin");
const std = @import("std");
const sync = @import("../../foundation/sync.zig");

// The objc runtime symbols only exist on macOS. Declare the externs only when
// targeting macOS so cross-compiles (Linux/Windows) don't emit undefined
// symbols at link time. Off-macOS this is an empty struct and the objc-using
// function bodies are comptime-dead (see the `comptime` early returns below).
const objc = if (builtin.target.os.tag == .macos) struct {
    extern fn objc_getClass(name: [*c]const u8) ?*anyopaque;
    extern fn sel_registerName(name: [*c]const u8) ?*anyopaque;
    extern fn objc_msgSend() void;
} else struct {};

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
const MsgSendUsizeUsizeRetId = *const fn (?*anyopaque, ?*anyopaque, usize, usize) callconv(.c) ?*anyopaque;
const MsgSendVoidRetPtr = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
const MsgSendIdUsizeUsizeRetVoid = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, usize, usize) callconv(.c) void;
const MsgSendPtrUsizeUsizeRetVoid = *const fn (?*anyopaque, ?*anyopaque, ?*const anyopaque, usize, usize) callconv(.c) void;
const MsgSendMtlSizeMtlSizeRetVoid = *const fn (?*anyopaque, ?*anyopaque, MTLSize, MTLSize) callconv(.c) void;

fn createNSString(allocator: std.mem.Allocator, str: []const u8) !?*anyopaque {
    if (comptime builtin.target.os.tag != .macos) return null;
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
    cosine_parts_pipeline: ?*anyopaque = null,
    reduce_sum_pipeline: ?*anyopaque = null,
    initialized: bool = false,
    mutex: sync.SpinLock = .{},

    pub fn init(self: *MetalContext, allocator: std.mem.Allocator) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.initialized) return;

        if (comptime builtin.target.os.tag != .macos) {
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
            \\
            \\kernel void cosine_parts_kernel(
            \\    device const float* a [[buffer(0)]],
            \\    device const float* b [[buffer(1)]],
            \\    device float* ab [[buffer(2)]],
            \\    device float* aa [[buffer(3)]],
            \\    device float* bb [[buffer(4)]],
            \\    uint id [[thread_position_in_grid]]
            \\) {
            \\    float av = a[id];
            \\    float bv = b[id];
            \\    ab[id] = av * bv;
            \\    aa[id] = av * av;
            \\    bb[id] = bv * bv;
            \\}
            \\
            \\// One partial sum per threadgroup (256 threads). Host sums the
            \\// partials. Not a multi-pass full-device tree reduce.
            \\kernel void reduce_sum_kernel(
            \\    device const float* in [[buffer(0)]],
            \\    device float* partials [[buffer(1)]],
            \\    constant uint& n [[buffer(2)]],
            \\    uint lid [[thread_position_in_threadgroup]],
            \\    uint tgid [[threadgroup_position_in_grid]]
            \\) {
            \\    threadgroup float shared[256];
            \\    const uint lsize = 256u;
            \\    uint i = tgid * lsize + lid;
            \\    shared[lid] = (i < n) ? in[i] : 0.0f;
            \\    threadgroup_barrier(mem_flags::mem_threadgroup);
            \\    for (uint stride = lsize / 2u; stride > 0u; stride >>= 1u) {
            \\        if (lid < stride) shared[lid] += shared[lid + stride];
            \\        threadgroup_barrier(mem_flags::mem_threadgroup);
            \\    }
            \\    if (lid == 0u) partials[tgid] = shared[0];
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

        const cosine_func_name = try createNSString(allocator, "cosine_parts_kernel") orelse return error.CreateStringFailed;
        const cosine_func = msg_send_id_ret_id(library, sel_newFunctionWithName, cosine_func_name) orelse return error.FunctionNotFound;

        const reduce_func_name = try createNSString(allocator, "reduce_sum_kernel") orelse return error.CreateStringFailed;
        const reduce_func = msg_send_id_ret_id(library, sel_newFunctionWithName, reduce_func_name) orelse return error.FunctionNotFound;

        const sel_newComputePipelineState = objc.sel_registerName("newComputePipelineStateWithFunction:error:");
        const msg_send_id_err_ret_id = @as(MsgSendIdErrRetId, @ptrCast(&objc.objc_msgSend));

        err = null;
        self.dot_pipeline = msg_send_id_err_ret_id(device, sel_newComputePipelineState, dot_func, @ptrCast(&err));
        if (self.dot_pipeline == null) return error.CreatePipelineStateFailed;

        err = null;
        self.l2_pipeline = msg_send_id_err_ret_id(device, sel_newComputePipelineState, l2_func, @ptrCast(&err));
        if (self.l2_pipeline == null) return error.CreatePipelineStateFailed;

        err = null;
        self.cosine_parts_pipeline = msg_send_id_err_ret_id(device, sel_newComputePipelineState, cosine_func, @ptrCast(&err));
        if (self.cosine_parts_pipeline == null) return error.CreatePipelineStateFailed;

        err = null;
        self.reduce_sum_pipeline = msg_send_id_err_ret_id(device, sel_newComputePipelineState, reduce_func, @ptrCast(&err));
        if (self.reduce_sum_pipeline == null) return error.CreatePipelineStateFailed;

        // Release one-time init temporaries; the pipeline states and queue stay retained.
        const sel_release = objc.sel_registerName("release");
        const msg_send_void_ret_void = @as(MsgSendVoidRetVoid, @ptrCast(&objc.objc_msgSend));
        msg_send_void_ret_void(dot_func, sel_release);
        msg_send_void_ret_void(l2_func, sel_release);
        msg_send_void_ret_void(cosine_func, sel_release);
        msg_send_void_ret_void(reduce_func, sel_release);
        msg_send_void_ret_void(library, sel_release);
        msg_send_void_ret_void(dot_func_name, sel_release);
        msg_send_void_ret_void(l2_func_name, sel_release);
        msg_send_void_ret_void(cosine_func_name, sel_release);
        msg_send_void_ret_void(reduce_func_name, sel_release);
        msg_send_void_ret_void(source_nsstring, sel_release);

        self.initialized = true;
    }

    pub fn runKernel(self: *MetalContext, pipeline: ?*anyopaque, len: usize, a: []const f32, b: []const f32, res: []f32) !void {
        if (comptime builtin.target.os.tag != .macos) return error.NotSupported;
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

    pub fn runCosinePartsKernel(
        self: *MetalContext,
        len: usize,
        a: []const f32,
        b: []const f32,
        ab: []f32,
        aa: []f32,
        bb: []f32,
    ) !void {
        if (comptime builtin.target.os.tag != .macos) return error.NotSupported;
        if (!self.initialized) return error.NotInitialized;
        if (ab.len < len or aa.len < len or bb.len < len) return error.BufferTooSmall;

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
        const sel_release = objc.sel_registerName("release");

        const byte_len = len * @sizeOf(f32);
        const buffer_a = msg_send_ptr_usize_usize_ret_id(self.device, sel_newBufferWithBytes, a.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        const buffer_b = msg_send_ptr_usize_usize_ret_id(self.device, sel_newBufferWithBytes, b.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        const buffer_ab = msg_send_ptr_usize_usize_ret_id(self.device, sel_newBufferWithBytes, ab.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        const buffer_aa = msg_send_ptr_usize_usize_ret_id(self.device, sel_newBufferWithBytes, aa.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        const buffer_bb = msg_send_ptr_usize_usize_ret_id(self.device, sel_newBufferWithBytes, bb.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;

        const cmd_buf = msg_send_void_ret_id(self.queue, sel_commandBuffer) orelse return error.CommandBufferCreationFailed;
        const encoder = msg_send_void_ret_id(cmd_buf, sel_computeCommandEncoder) orelse return error.EncoderCreationFailed;

        _ = msg_send_id_ret_id(encoder, sel_setComputePipelineState, self.cosine_parts_pipeline);
        msg_send_id_usize_usize_ret_void(encoder, sel_setBufferOffsetAtIndex, buffer_a, 0, 0);
        msg_send_id_usize_usize_ret_void(encoder, sel_setBufferOffsetAtIndex, buffer_b, 0, 1);
        msg_send_id_usize_usize_ret_void(encoder, sel_setBufferOffsetAtIndex, buffer_ab, 0, 2);
        msg_send_id_usize_usize_ret_void(encoder, sel_setBufferOffsetAtIndex, buffer_aa, 0, 3);
        msg_send_id_usize_usize_ret_void(encoder, sel_setBufferOffsetAtIndex, buffer_bb, 0, 4);

        const local_size = @min(len, 256);
        const threadgroups = MTLSize{ .width = len, .height = 1, .depth = 1 };
        const threads_per_threadgroup = MTLSize{ .width = local_size, .height = 1, .depth = 1 };
        msg_send_mtlsize_mtlsize_ret_void(encoder, sel_dispatch, threadgroups, threads_per_threadgroup);
        msg_send_void_ret_void(encoder, sel_endEncoding);
        msg_send_void_ret_void(cmd_buf, sel_commit);
        msg_send_void_ret_void(cmd_buf, sel_waitUntilCompleted);

        const ab_ptr = msg_send_void_ret_ptr(buffer_ab, sel_contents) orelse return error.ReadBufferFailed;
        const aa_ptr = msg_send_void_ret_ptr(buffer_aa, sel_contents) orelse return error.ReadBufferFailed;
        const bb_ptr = msg_send_void_ret_ptr(buffer_bb, sel_contents) orelse return error.ReadBufferFailed;
        @memcpy(ab[0..len], @as([*]f32, @ptrCast(@alignCast(ab_ptr)))[0..len]);
        @memcpy(aa[0..len], @as([*]f32, @ptrCast(@alignCast(aa_ptr)))[0..len]);
        @memcpy(bb[0..len], @as([*]f32, @ptrCast(@alignCast(bb_ptr)))[0..len]);

        msg_send_void_ret_void(buffer_a, sel_release);
        msg_send_void_ret_void(buffer_b, sel_release);
        msg_send_void_ret_void(buffer_ab, sel_release);
        msg_send_void_ret_void(buffer_aa, sel_release);
        msg_send_void_ret_void(buffer_bb, sel_release);
    }

    /// Threadgroup partial reduce on the GPU (256-wide), then host SIMD sum of
    /// partials. Not a multi-pass full-device tree reduce.
    pub fn runReduceSum(self: *MetalContext, values: []const f32) !f32 {
        if (comptime builtin.target.os.tag != .macos) return error.NotSupported;
        if (!self.initialized) return error.NotInitialized;
        if (values.len == 0) return 0;
        if (self.reduce_sum_pipeline == null) return error.NotInitialized;

        const tg_size: usize = 256;
        const num_groups = (values.len + tg_size - 1) / tg_size;

        const msg_send_void_ret_id = @as(MsgSendVoidRetId, @ptrCast(&objc.objc_msgSend));
        const msg_send_void_ret_void = @as(MsgSendVoidRetVoid, @ptrCast(&objc.objc_msgSend));
        const msg_send_id_ret_id = @as(MsgSendIdRetId, @ptrCast(&objc.objc_msgSend));
        const msg_send_ptr_usize_usize_ret_id = @as(MsgSendPtrUsizeUsizeRetId, @ptrCast(&objc.objc_msgSend));
        const msg_send_usize_usize_ret_id = @as(MsgSendUsizeUsizeRetId, @ptrCast(&objc.objc_msgSend));
        const msg_send_id_usize_usize_ret_void = @as(MsgSendIdUsizeUsizeRetVoid, @ptrCast(&objc.objc_msgSend));
        const msg_send_ptr_usize_usize_ret_void = @as(MsgSendPtrUsizeUsizeRetVoid, @ptrCast(&objc.objc_msgSend));
        const msg_send_mtlsize_mtlsize_ret_void = @as(MsgSendMtlSizeMtlSizeRetVoid, @ptrCast(&objc.objc_msgSend));
        const msg_send_void_ret_ptr = @as(MsgSendVoidRetPtr, @ptrCast(&objc.objc_msgSend));

        const sel_commandBuffer = objc.sel_registerName("commandBuffer");
        const sel_computeCommandEncoder = objc.sel_registerName("computeCommandEncoder");
        const sel_setComputePipelineState = objc.sel_registerName("setComputePipelineState:");
        const sel_newBufferWithBytes = objc.sel_registerName("newBufferWithBytes:length:options:");
        const sel_newBufferWithLength = objc.sel_registerName("newBufferWithLength:options:");
        const sel_setBufferOffsetAtIndex = objc.sel_registerName("setBuffer:offset:atIndex:");
        const sel_setBytes = objc.sel_registerName("setBytes:length:atIndex:");
        const sel_dispatch = objc.sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:");
        const sel_endEncoding = objc.sel_registerName("endEncoding");
        const sel_commit = objc.sel_registerName("commit");
        const sel_waitUntilCompleted = objc.sel_registerName("waitUntilCompleted");
        const sel_contents = objc.sel_registerName("contents");
        const sel_release = objc.sel_registerName("release");

        const in_bytes = values.len * @sizeOf(f32);
        const partial_bytes = num_groups * @sizeOf(f32);
        const buffer_in = msg_send_ptr_usize_usize_ret_id(self.device, sel_newBufferWithBytes, values.ptr, in_bytes, 0) orelse return error.BufferAllocationFailed;
        const buffer_partials = msg_send_usize_usize_ret_id(self.device, sel_newBufferWithLength, partial_bytes, 0) orelse return error.BufferAllocationFailed;

        const cmd_buf = msg_send_void_ret_id(self.queue, sel_commandBuffer) orelse return error.CommandBufferCreationFailed;
        const encoder = msg_send_void_ret_id(cmd_buf, sel_computeCommandEncoder) orelse return error.EncoderCreationFailed;

        _ = msg_send_id_ret_id(encoder, sel_setComputePipelineState, self.reduce_sum_pipeline);
        msg_send_id_usize_usize_ret_void(encoder, sel_setBufferOffsetAtIndex, buffer_in, 0, 0);
        msg_send_id_usize_usize_ret_void(encoder, sel_setBufferOffsetAtIndex, buffer_partials, 0, 1);
        var n_u32: u32 = @intCast(values.len);
        msg_send_ptr_usize_usize_ret_void(encoder, sel_setBytes, &n_u32, @sizeOf(u32), 2);

        const groups = MTLSize{ .width = num_groups, .height = 1, .depth = 1 };
        const threads = MTLSize{ .width = tg_size, .height = 1, .depth = 1 };
        msg_send_mtlsize_mtlsize_ret_void(encoder, sel_dispatch, groups, threads);
        msg_send_void_ret_void(encoder, sel_endEncoding);
        msg_send_void_ret_void(cmd_buf, sel_commit);
        msg_send_void_ret_void(cmd_buf, sel_waitUntilCompleted);

        const partial_ptr = msg_send_void_ret_ptr(buffer_partials, sel_contents) orelse return error.ReadBufferFailed;
        const partials = @as([*]f32, @ptrCast(@alignCast(partial_ptr)))[0..num_groups];

        var sum: f32 = 0;
        var i: usize = 0;
        const VLen = comptime std.simd.suggestVectorLength(f32) orelse 4;
        while (i + VLen <= partials.len) : (i += VLen) {
            const v: @Vector(VLen, f32) = partials[i..][0..VLen].*;
            sum += @reduce(.Add, v);
        }
        while (i < partials.len) : (i += 1) sum += partials[i];

        msg_send_void_ret_void(buffer_in, sel_release);
        msg_send_void_ret_void(buffer_partials, sel_release);
        return sum;
    }
};

pub var g_metal_context = MetalContext{};

test {
    std.testing.refAllDecls(@This());
}
