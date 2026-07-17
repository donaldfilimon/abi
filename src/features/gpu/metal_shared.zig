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

/// Shared ObjC msgSend casts + Metal selectors used by every dispatch path.
/// Constructed once per call site via `load()` so kernels share one helper path
/// instead of duplicating cast/selector boilerplate.
const MetalDispatch = struct {
    msg_void_id: MsgSendVoidRetId,
    msg_void_void: MsgSendVoidRetVoid,
    msg_id_id: MsgSendIdRetId,
    msg_bytes: MsgSendPtrUsizeUsizeRetId,
    msg_len: MsgSendUsizeUsizeRetId,
    msg_set_buf: MsgSendIdUsizeUsizeRetVoid,
    msg_set_bytes: MsgSendPtrUsizeUsizeRetVoid,
    msg_dispatch: MsgSendMtlSizeMtlSizeRetVoid,
    msg_contents: MsgSendVoidRetPtr,

    sel_command_buffer: ?*anyopaque,
    sel_encoder: ?*anyopaque,
    sel_set_pipeline: ?*anyopaque,
    sel_new_bytes: ?*anyopaque,
    sel_new_length: ?*anyopaque,
    sel_set_buffer: ?*anyopaque,
    sel_set_bytes: ?*anyopaque,
    sel_dispatch_threads: ?*anyopaque,
    sel_dispatch_groups: ?*anyopaque,
    sel_end: ?*anyopaque,
    sel_commit: ?*anyopaque,
    sel_wait: ?*anyopaque,
    sel_contents: ?*anyopaque,
    sel_release: ?*anyopaque,

    fn load() MetalDispatch {
        if (comptime builtin.target.os.tag != .macos) {
            return undefined;
        }
        return .{
            .msg_void_id = @ptrCast(&objc.objc_msgSend),
            .msg_void_void = @ptrCast(&objc.objc_msgSend),
            .msg_id_id = @ptrCast(&objc.objc_msgSend),
            .msg_bytes = @ptrCast(&objc.objc_msgSend),
            .msg_len = @ptrCast(&objc.objc_msgSend),
            .msg_set_buf = @ptrCast(&objc.objc_msgSend),
            .msg_set_bytes = @ptrCast(&objc.objc_msgSend),
            .msg_dispatch = @ptrCast(&objc.objc_msgSend),
            .msg_contents = @ptrCast(&objc.objc_msgSend),
            .sel_command_buffer = objc.sel_registerName("commandBuffer"),
            .sel_encoder = objc.sel_registerName("computeCommandEncoder"),
            .sel_set_pipeline = objc.sel_registerName("setComputePipelineState:"),
            .sel_new_bytes = objc.sel_registerName("newBufferWithBytes:length:options:"),
            .sel_new_length = objc.sel_registerName("newBufferWithLength:options:"),
            .sel_set_buffer = objc.sel_registerName("setBuffer:offset:atIndex:"),
            .sel_set_bytes = objc.sel_registerName("setBytes:length:atIndex:"),
            .sel_dispatch_threads = objc.sel_registerName("dispatchThreads:threadsPerThreadgroup:"),
            .sel_dispatch_groups = objc.sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"),
            .sel_end = objc.sel_registerName("endEncoding"),
            .sel_commit = objc.sel_registerName("commit"),
            .sel_wait = objc.sel_registerName("waitUntilCompleted"),
            .sel_contents = objc.sel_registerName("contents"),
            .sel_release = objc.sel_registerName("release"),
        };
    }

    fn release(self: MetalDispatch, obj: ?*anyopaque) void {
        self.msg_void_void(obj, self.sel_release);
    }

    fn commitAndWait(self: MetalDispatch, cmd_buf: ?*anyopaque) void {
        self.msg_void_void(cmd_buf, self.sel_commit);
        self.msg_void_void(cmd_buf, self.sel_wait);
    }

    fn readF32(self: MetalDispatch, buffer: ?*anyopaque) !f32 {
        const ptr = self.msg_contents(buffer, self.sel_contents) orelse return error.ReadBufferFailed;
        return @as([*]f32, @ptrCast(@alignCast(ptr)))[0];
    }

    fn copyBufferToHost(self: MetalDispatch, buffer: ?*anyopaque, dest: []f32) !void {
        const ptr = self.msg_contents(buffer, self.sel_contents) orelse return error.ReadBufferFailed;
        @memcpy(dest, @as([*]f32, @ptrCast(@alignCast(ptr)))[0..dest.len]);
    }

    /// Encode one 256-wide reduce pass into an existing command buffer.
    /// Returns the newly allocated partials buffer (caller owns) and its length.
    fn encodeReducePass(
        self: MetalDispatch,
        device: ?*anyopaque,
        pipeline: ?*anyopaque,
        cmd_buf: ?*anyopaque,
        buffer_in: ?*anyopaque,
        current_len: usize,
    ) !struct { buffer: ?*anyopaque, len: usize } {
        const tg_size: usize = 256;
        const num_groups = (current_len + tg_size - 1) / tg_size;
        const partial_bytes = num_groups * @sizeOf(f32);
        const buffer_out = self.msg_len(device, self.sel_new_length, partial_bytes, 0) orelse
            return error.BufferAllocationFailed;

        const encoder = self.msg_void_id(cmd_buf, self.sel_encoder) orelse {
            self.release(buffer_out);
            return error.EncoderCreationFailed;
        };

        _ = self.msg_id_id(encoder, self.sel_set_pipeline, pipeline);
        self.msg_set_buf(encoder, self.sel_set_buffer, buffer_in, 0, 0);
        self.msg_set_buf(encoder, self.sel_set_buffer, buffer_out, 0, 1);
        var n_u32: u32 = @intCast(current_len);
        self.msg_set_bytes(encoder, self.sel_set_bytes, &n_u32, @sizeOf(u32), 2);

        const groups = MTLSize{ .width = num_groups, .height = 1, .depth = 1 };
        const threads = MTLSize{ .width = tg_size, .height = 1, .depth = 1 };
        self.msg_dispatch(encoder, self.sel_dispatch_groups, groups, threads);
        self.msg_void_void(encoder, self.sel_end);

        return .{ .buffer = buffer_out, .len = num_groups };
    }

    /// Multi-pass reduce encoded into `cmd_buf` (no commit/wait). Returns the
    /// final single-element buffer; releases intermediate inputs.
    fn encodeReduceToScalar(
        self: MetalDispatch,
        device: ?*anyopaque,
        pipeline: ?*anyopaque,
        cmd_buf: ?*anyopaque,
        buffer_in_owned: ?*anyopaque,
        start_len: usize,
    ) !?*anyopaque {
        var buffer_in = buffer_in_owned;
        var current_len = start_len;
        errdefer self.release(buffer_in);

        while (current_len > 1) {
            const pass = try self.encodeReducePass(device, pipeline, cmd_buf, buffer_in, current_len);
            self.release(buffer_in);
            buffer_in = pass.buffer;
            current_len = pass.len;
        }
        return buffer_in;
    }
};

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
            \\// One partial sum per threadgroup (256 threads). Zig encodeReduce*
            \\// re-dispatches until a single scalar remains (multi-pass).
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

    fn encodeMapKernel(
        d: MetalDispatch,
        pipeline: ?*anyopaque,
        cmd_buf: ?*anyopaque,
        buffer_a: ?*anyopaque,
        buffer_b: ?*anyopaque,
        buffer_res: ?*anyopaque,
        len: usize,
    ) !void {
        const encoder = d.msg_void_id(cmd_buf, d.sel_encoder) orelse return error.EncoderCreationFailed;
        _ = d.msg_id_id(encoder, d.sel_set_pipeline, pipeline);
        d.msg_set_buf(encoder, d.sel_set_buffer, buffer_a, 0, 0);
        d.msg_set_buf(encoder, d.sel_set_buffer, buffer_b, 0, 1);
        d.msg_set_buf(encoder, d.sel_set_buffer, buffer_res, 0, 2);

        const local_size = @min(len, 256);
        const threadgroups = MTLSize{ .width = len, .height = 1, .depth = 1 };
        const threads_per_threadgroup = MTLSize{ .width = local_size, .height = 1, .depth = 1 };
        d.msg_dispatch(encoder, d.sel_dispatch_threads, threadgroups, threads_per_threadgroup);
        d.msg_void_void(encoder, d.sel_end);
    }

    /// Map kernel then multi-pass reduce on-GPU in one command buffer.
    /// Avoids host copy + re-upload between map and reduce.
    pub fn runMapAndReduce(
        self: *MetalContext,
        pipeline: ?*anyopaque,
        len: usize,
        a: []const f32,
        b: []const f32,
    ) !f32 {
        if (comptime builtin.target.os.tag != .macos) return error.NotSupported;
        if (!self.initialized) return error.NotInitialized;
        if (self.reduce_sum_pipeline == null) return error.NotInitialized;
        if (len == 0) return 0;

        const d = MetalDispatch.load();
        const byte_len = len * @sizeOf(f32);

        const buffer_a = d.msg_bytes(self.device, d.sel_new_bytes, a.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_a);
        const buffer_b = d.msg_bytes(self.device, d.sel_new_bytes, b.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_b);
        // Ownership of buffer_res transfers to encodeReduceToScalar (no defer).
        const buffer_res = d.msg_len(self.device, d.sel_new_length, byte_len, 0) orelse return error.BufferAllocationFailed;

        const cmd_buf = d.msg_void_id(self.queue, d.sel_command_buffer) orelse {
            d.release(buffer_res);
            return error.CommandBufferCreationFailed;
        };
        encodeMapKernel(d, pipeline, cmd_buf, buffer_a, buffer_b, buffer_res, len) catch |e| {
            d.release(buffer_res);
            return e;
        };

        const scalar_buf = try d.encodeReduceToScalar(
            self.device,
            self.reduce_sum_pipeline,
            cmd_buf,
            buffer_res,
            len,
        );

        d.commitAndWait(cmd_buf);
        const sum = d.readF32(scalar_buf) catch |e| {
            d.release(scalar_buf);
            return e;
        };
        d.release(scalar_buf);
        return sum;
    }

    pub fn runKernel(self: *MetalContext, pipeline: ?*anyopaque, len: usize, a: []const f32, b: []const f32, res: []f32) !void {
        if (comptime builtin.target.os.tag != .macos) return error.NotSupported;
        if (!self.initialized) return error.NotInitialized;

        const d = MetalDispatch.load();
        const byte_len = len * @sizeOf(f32);

        const buffer_a = d.msg_bytes(self.device, d.sel_new_bytes, a.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_a);
        const buffer_b = d.msg_bytes(self.device, d.sel_new_bytes, b.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_b);
        const buffer_res = d.msg_bytes(self.device, d.sel_new_bytes, res.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_res);

        const cmd_buf = d.msg_void_id(self.queue, d.sel_command_buffer) orelse return error.CommandBufferCreationFailed;
        try encodeMapKernel(d, pipeline, cmd_buf, buffer_a, buffer_b, buffer_res, len);
        d.commitAndWait(cmd_buf);
        try d.copyBufferToHost(buffer_res, res[0..len]);
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

        const d = MetalDispatch.load();
        const byte_len = len * @sizeOf(f32);

        const buffer_a = d.msg_bytes(self.device, d.sel_new_bytes, a.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_a);
        const buffer_b = d.msg_bytes(self.device, d.sel_new_bytes, b.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_b);
        const buffer_ab = d.msg_bytes(self.device, d.sel_new_bytes, ab.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_ab);
        const buffer_aa = d.msg_bytes(self.device, d.sel_new_bytes, aa.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_aa);
        const buffer_bb = d.msg_bytes(self.device, d.sel_new_bytes, bb.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_bb);

        const cmd_buf = d.msg_void_id(self.queue, d.sel_command_buffer) orelse return error.CommandBufferCreationFailed;
        const encoder = d.msg_void_id(cmd_buf, d.sel_encoder) orelse return error.EncoderCreationFailed;

        _ = d.msg_id_id(encoder, d.sel_set_pipeline, self.cosine_parts_pipeline);
        d.msg_set_buf(encoder, d.sel_set_buffer, buffer_a, 0, 0);
        d.msg_set_buf(encoder, d.sel_set_buffer, buffer_b, 0, 1);
        d.msg_set_buf(encoder, d.sel_set_buffer, buffer_ab, 0, 2);
        d.msg_set_buf(encoder, d.sel_set_buffer, buffer_aa, 0, 3);
        d.msg_set_buf(encoder, d.sel_set_buffer, buffer_bb, 0, 4);

        const local_size = @min(len, 256);
        const threadgroups = MTLSize{ .width = len, .height = 1, .depth = 1 };
        const threads_per_threadgroup = MTLSize{ .width = local_size, .height = 1, .depth = 1 };
        d.msg_dispatch(encoder, d.sel_dispatch_threads, threadgroups, threads_per_threadgroup);
        d.msg_void_void(encoder, d.sel_end);
        d.commitAndWait(cmd_buf);

        try d.copyBufferToHost(buffer_ab, ab[0..len]);
        try d.copyBufferToHost(buffer_aa, aa[0..len]);
        try d.copyBufferToHost(buffer_bb, bb[0..len]);
    }

    pub const CosineSums = struct {
        ab: f32,
        aa: f32,
        bb: f32,
    };

    /// Fused cosine parts + three on-GPU reduces in one command buffer.
    /// Returns three scalars only — no full-array host round-trips.
    pub fn runCosineFused(self: *MetalContext, len: usize, a: []const f32, b: []const f32) !CosineSums {
        if (comptime builtin.target.os.tag != .macos) return error.NotSupported;
        if (!self.initialized) return error.NotInitialized;
        if (self.cosine_parts_pipeline == null or self.reduce_sum_pipeline == null) return error.NotInitialized;
        if (len == 0) return .{ .ab = 0, .aa = 0, .bb = 0 };

        const d = MetalDispatch.load();
        const byte_len = len * @sizeOf(f32);

        const buffer_a = d.msg_bytes(self.device, d.sel_new_bytes, a.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_a);
        const buffer_b = d.msg_bytes(self.device, d.sel_new_bytes, b.ptr, byte_len, 0) orelse return error.BufferAllocationFailed;
        defer d.release(buffer_b);

        const buffer_ab = d.msg_len(self.device, d.sel_new_length, byte_len, 0) orelse return error.BufferAllocationFailed;
        const buffer_aa = d.msg_len(self.device, d.sel_new_length, byte_len, 0) orelse {
            d.release(buffer_ab);
            return error.BufferAllocationFailed;
        };
        const buffer_bb = d.msg_len(self.device, d.sel_new_length, byte_len, 0) orelse {
            d.release(buffer_ab);
            d.release(buffer_aa);
            return error.BufferAllocationFailed;
        };

        const cmd_buf = d.msg_void_id(self.queue, d.sel_command_buffer) orelse {
            d.release(buffer_ab);
            d.release(buffer_aa);
            d.release(buffer_bb);
            return error.CommandBufferCreationFailed;
        };

        {
            const encoder = d.msg_void_id(cmd_buf, d.sel_encoder) orelse {
                d.release(buffer_ab);
                d.release(buffer_aa);
                d.release(buffer_bb);
                return error.EncoderCreationFailed;
            };
            _ = d.msg_id_id(encoder, d.sel_set_pipeline, self.cosine_parts_pipeline);
            d.msg_set_buf(encoder, d.sel_set_buffer, buffer_a, 0, 0);
            d.msg_set_buf(encoder, d.sel_set_buffer, buffer_b, 0, 1);
            d.msg_set_buf(encoder, d.sel_set_buffer, buffer_ab, 0, 2);
            d.msg_set_buf(encoder, d.sel_set_buffer, buffer_aa, 0, 3);
            d.msg_set_buf(encoder, d.sel_set_buffer, buffer_bb, 0, 4);
            const local_size = @min(len, 256);
            const threadgroups = MTLSize{ .width = len, .height = 1, .depth = 1 };
            const threads_per_threadgroup = MTLSize{ .width = local_size, .height = 1, .depth = 1 };
            d.msg_dispatch(encoder, d.sel_dispatch_threads, threadgroups, threads_per_threadgroup);
            d.msg_void_void(encoder, d.sel_end);
        }

        // Each encodeReduceToScalar takes ownership of its input buffer.
        const ab_scalar = d.encodeReduceToScalar(self.device, self.reduce_sum_pipeline, cmd_buf, buffer_ab, len) catch |e| {
            d.release(buffer_aa);
            d.release(buffer_bb);
            return e;
        };
        const aa_scalar = d.encodeReduceToScalar(self.device, self.reduce_sum_pipeline, cmd_buf, buffer_aa, len) catch |e| {
            d.release(ab_scalar);
            d.release(buffer_bb);
            return e;
        };
        const bb_scalar = d.encodeReduceToScalar(self.device, self.reduce_sum_pipeline, cmd_buf, buffer_bb, len) catch |e| {
            d.release(ab_scalar);
            d.release(aa_scalar);
            return e;
        };

        d.commitAndWait(cmd_buf);
        const sums = CosineSums{
            .ab = d.readF32(ab_scalar) catch |e| {
                d.release(ab_scalar);
                d.release(aa_scalar);
                d.release(bb_scalar);
                return e;
            },
            .aa = d.readF32(aa_scalar) catch |e| {
                d.release(ab_scalar);
                d.release(aa_scalar);
                d.release(bb_scalar);
                return e;
            },
            .bb = d.readF32(bb_scalar) catch |e| {
                d.release(ab_scalar);
                d.release(aa_scalar);
                d.release(bb_scalar);
                return e;
            },
        };
        d.release(ab_scalar);
        d.release(aa_scalar);
        d.release(bb_scalar);
        return sums;
    }

    /// Multi-pass threadgroup reduce on the GPU (256-wide). All passes encode
    /// into one command buffer; a single waitUntilCompleted at the end.
    /// Callers fall back to host `sumF32` if Metal dispatch fails.
    pub fn runReduceSum(self: *MetalContext, values: []const f32) !f32 {
        if (comptime builtin.target.os.tag != .macos) return error.NotSupported;
        if (!self.initialized) return error.NotInitialized;
        if (values.len == 0) return 0;
        if (values.len == 1) return values[0];
        if (self.reduce_sum_pipeline == null) return error.NotInitialized;

        const d = MetalDispatch.load();
        const in_bytes = values.len * @sizeOf(f32);
        const buffer_in = d.msg_bytes(self.device, d.sel_new_bytes, values.ptr, in_bytes, 0) orelse
            return error.BufferAllocationFailed;

        const cmd_buf = d.msg_void_id(self.queue, d.sel_command_buffer) orelse {
            d.release(buffer_in);
            return error.CommandBufferCreationFailed;
        };

        const scalar_buf = d.encodeReduceToScalar(
            self.device,
            self.reduce_sum_pipeline,
            cmd_buf,
            buffer_in,
            values.len,
        ) catch |e| {
            // encodeReduceToScalar errdefer releases buffer_in on failure mid-chain;
            // if the failure is before any pass, buffer_in is still owned.
            // encodeReduceToScalar's errdefer handles it.
            return e;
        };

        d.commitAndWait(cmd_buf);
        const sum = d.readF32(scalar_buf) catch |e| {
            d.release(scalar_buf);
            return e;
        };
        d.release(scalar_buf);
        return sum;
    }
};

pub var g_metal_context = MetalContext{};

test {
    std.testing.refAllDecls(@This());
}
