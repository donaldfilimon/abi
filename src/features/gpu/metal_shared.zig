//! Shared Metal context for vector_ops / backends / compute_api.
//! ObjC FFI helpers live in metal_objc.zig.
const builtin = @import("builtin");
const std = @import("std");
const sync = @import("../../foundation/sync.zig");
const metal_objc = @import("metal_objc.zig");

const objc = metal_objc.objc;
const MTLCreateSystemDefaultDevice = metal_objc.MTLCreateSystemDefaultDevice;
const MTLSize = metal_objc.MTLSize;
const MetalDispatch = metal_objc.MetalDispatch;
const createNSString = metal_objc.createNSString;
const MsgSendVoidRetId = metal_objc.MsgSendVoidRetId;
const MsgSendIdIdErrRetId = metal_objc.MsgSendIdIdErrRetId;
const MsgSendIdErrRetId = metal_objc.MsgSendIdErrRetId;
const MsgSendIdRetId = metal_objc.MsgSendIdRetId;
const MsgSendUsizeUsizeRetId = metal_objc.MsgSendUsizeUsizeRetId;
const MsgSendVoidRetPtr = metal_objc.MsgSendVoidRetPtr;
const MsgSendIdUsizeUsizeRetVoid = metal_objc.MsgSendIdUsizeUsizeRetVoid;
const MsgSendPtrUsizeUsizeRetVoid = metal_objc.MsgSendPtrUsizeUsizeRetVoid;
const MsgSendMtlSizeMtlSizeRetVoid = metal_objc.MsgSendMtlSizeMtlSizeRetVoid;
const MsgSendPtrUsizeUsizeRetId = metal_objc.MsgSendPtrUsizeUsizeRetId;
const MsgSendVoidRetVoid = metal_objc.MsgSendVoidRetVoid;

const MetalContext = struct {
    device: ?*anyopaque = null,
    queue: ?*anyopaque = null,
    dot_pipeline: ?*anyopaque = null,
    l2_pipeline: ?*anyopaque = null,
    cosine_parts_pipeline: ?*anyopaque = null,
    batch_cosine_pipeline: ?*anyopaque = null,
    reduce_sum_pipeline: ?*anyopaque = null,
    softmax_pipeline: ?*anyopaque = null,
    softmax_norm_pipeline: ?*anyopaque = null,
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
            \\// One threadgroup per candidate row (candidate `dim`-length vectors
            \\// are laid out contiguously in `candidates`). Threads stride over
            \\// the dimension accumulating partial qc/cc sums, then a shared-memory
            \\// tree reduction (same pattern as reduce_sum_kernel) collapses to one
            \\// value per threadgroup; thread 0 divides by the precomputed query
            \\// norm and writes the cosine similarity for that candidate.
            \\kernel void batch_cosine_kernel(
            \\    device const float* query [[buffer(0)]],
            \\    device const float* candidates [[buffer(1)]],
            \\    device float* out [[buffer(2)]],
            \\    constant uint& dim [[buffer(3)]],
            \\    constant float& q_norm [[buffer(4)]],
            \\    uint lid [[thread_position_in_threadgroup]],
            \\    uint tgid [[threadgroup_position_in_grid]]
            \\) {
            \\    threadgroup float shared_qc[256];
            \\    threadgroup float shared_cc[256];
            \\    const uint lsize = 256u;
            \\    const uint base = tgid * dim;
            \\    float qc_part = 0.0f;
            \\    float cc_part = 0.0f;
            \\    for (uint i = lid; i < dim; i += lsize) {
            \\        float qv = query[i];
            \\        float cv = candidates[base + i];
            \\        qc_part += qv * cv;
            \\        cc_part += cv * cv;
            \\    }
            \\    shared_qc[lid] = qc_part;
            \\    shared_cc[lid] = cc_part;
            \\    threadgroup_barrier(mem_flags::mem_threadgroup);
            \\    for (uint stride = lsize / 2u; stride > 0u; stride >>= 1u) {
            \\        if (lid < stride) {
            \\            shared_qc[lid] += shared_qc[lid + stride];
            \\            shared_cc[lid] += shared_cc[lid + stride];
            \\        }
            \\        threadgroup_barrier(mem_flags::mem_threadgroup);
            \\    }
            \\    if (lid == 0u) {
            \\        float cc = shared_cc[0];
            \\        if (q_norm == 0.0f || cc == 0.0f) {
            \\            out[tgid] = 0.0f;
            \\        } else {
            \\            out[tgid] = shared_qc[0] / (q_norm * sqrt(cc));
            \\        }
            \\    }
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
            \\
            \\// exp(x - max) into `result`. `max_val` is the row max, passed as a
            \\// kernel constant so every lane subtracts the same shift (numerically
            \\// stable softmax). A separate reduce_sum_kernel pass sums `result`,
            \\// then softmax_norm_kernel divides each element by that sum.
            \\kernel void softmax_kernel(
            \\    device const float* in [[buffer(0)]],
            \\    constant float& max_val [[buffer(1)]],
            \\    device float* out [[buffer(2)]],
            \\    uint id [[thread_position_in_grid]]
            \\) {
            \\    out[id] = exp(in[id] - max_val);
            \\}
            \\
            \\// out[id] = in[id] / norm (the normalization step of softmax).
            \\kernel void softmax_norm_kernel(
            \\    const device float* in [[buffer(0)]],
            \\    constant float& norm [[buffer(1)]],
            \\    device float* out [[buffer(2)]],
            \\    uint id [[thread_position_in_grid]]
            \\) {
            \\    out[id] = in[id] / norm;
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

        const batch_cosine_func_name = try createNSString(allocator, "batch_cosine_kernel") orelse return error.CreateStringFailed;
        const batch_cosine_func = msg_send_id_ret_id(library, sel_newFunctionWithName, batch_cosine_func_name) orelse return error.FunctionNotFound;

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
        self.batch_cosine_pipeline = msg_send_id_err_ret_id(device, sel_newComputePipelineState, batch_cosine_func, @ptrCast(&err));
        if (self.batch_cosine_pipeline == null) return error.CreatePipelineStateFailed;

        err = null;
        self.reduce_sum_pipeline = msg_send_id_err_ret_id(device, sel_newComputePipelineState, reduce_func, @ptrCast(&err));
        if (self.reduce_sum_pipeline == null) return error.CreatePipelineStateFailed;

        const softmax_func_name = try createNSString(allocator, "softmax_kernel") orelse return error.CreateStringFailed;
        const softmax_func = msg_send_id_ret_id(library, sel_newFunctionWithName, softmax_func_name) orelse return error.FunctionNotFound;

        const softmax_norm_func_name = try createNSString(allocator, "softmax_norm_kernel") orelse return error.CreateStringFailed;
        const softmax_norm_func = msg_send_id_ret_id(library, sel_newFunctionWithName, softmax_norm_func_name) orelse return error.FunctionNotFound;

        err = null;
        self.softmax_pipeline = msg_send_id_err_ret_id(device, sel_newComputePipelineState, softmax_func, @ptrCast(&err));
        if (self.softmax_pipeline == null) return error.CreatePipelineStateFailed;

        err = null;
        self.softmax_norm_pipeline = msg_send_id_err_ret_id(device, sel_newComputePipelineState, softmax_norm_func, @ptrCast(&err));
        if (self.softmax_norm_pipeline == null) return error.CreatePipelineStateFailed;

        // Release one-time init temporaries; the pipeline states and queue stay retained.
        const sel_release = objc.sel_registerName("release");
        const msg_send_void_ret_void = @as(MsgSendVoidRetVoid, @ptrCast(&objc.objc_msgSend));
        msg_send_void_ret_void(dot_func, sel_release);
        msg_send_void_ret_void(l2_func, sel_release);
        msg_send_void_ret_void(cosine_func, sel_release);
        msg_send_void_ret_void(batch_cosine_func, sel_release);
        msg_send_void_ret_void(reduce_func, sel_release);
        msg_send_void_ret_void(softmax_func, sel_release);
        msg_send_void_ret_void(softmax_norm_func, sel_release);
        msg_send_void_ret_void(library, sel_release);
        msg_send_void_ret_void(dot_func_name, sel_release);
        msg_send_void_ret_void(l2_func_name, sel_release);
        msg_send_void_ret_void(cosine_func_name, sel_release);
        msg_send_void_ret_void(batch_cosine_func_name, sel_release);
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

    /// Batched cosine similarity: one command buffer, one dispatch, N
    /// threadgroups (one per candidate row). `candidates_flat` must be a
    /// contiguous row-major buffer of `n * d` floats (caller flattens; see
    /// vector_ops.batchCosineSimilarity, which owns that host-side copy).
    /// `out` receives the `n` cosine similarities. Correctness-parity proven
    /// against an independent scalar reference (see vector_ops.zig tests) —
    /// no speedup claim; dispatch-count reduction vs. the per-pair
    /// `runCosineFused` loop is unmeasured.
    pub fn runBatchCosineFused(
        self: *MetalContext,
        query: []const f32,
        candidates_flat: []const f32,
        n: usize,
        d: usize,
        out: []f32,
    ) !void {
        if (comptime builtin.target.os.tag != .macos) return error.NotSupported;
        if (!self.initialized) return error.NotInitialized;
        if (self.batch_cosine_pipeline == null) return error.NotInitialized;
        if (query.len != d) return error.DimensionMismatch;
        if (candidates_flat.len != n * d) return error.DimensionMismatch;
        if (out.len != n) return error.DimensionMismatch;
        if (n == 0 or d == 0) return;

        // Query norm computed once, host-side, and passed to every threadgroup
        // as a kernel constant — avoids re-deriving it per candidate.
        var q_norm_sq: f32 = 0;
        for (query) |v| q_norm_sq += v * v;
        const q_norm = @sqrt(q_norm_sq);

        const disp = MetalDispatch.load();
        const query_bytes = d * @sizeOf(f32);
        const candidates_bytes = n * d * @sizeOf(f32);
        const out_bytes = n * @sizeOf(f32);

        const buffer_query = disp.msg_bytes(self.device, disp.sel_new_bytes, query.ptr, query_bytes, 0) orelse
            return error.BufferAllocationFailed;
        defer disp.release(buffer_query);
        const buffer_candidates = disp.msg_bytes(self.device, disp.sel_new_bytes, candidates_flat.ptr, candidates_bytes, 0) orelse
            return error.BufferAllocationFailed;
        defer disp.release(buffer_candidates);
        const buffer_out = disp.msg_len(self.device, disp.sel_new_length, out_bytes, 0) orelse
            return error.BufferAllocationFailed;
        defer disp.release(buffer_out);

        const cmd_buf = disp.msg_void_id(self.queue, disp.sel_command_buffer) orelse
            return error.CommandBufferCreationFailed;

        const encoder = disp.msg_void_id(cmd_buf, disp.sel_encoder) orelse return error.EncoderCreationFailed;
        _ = disp.msg_id_id(encoder, disp.sel_set_pipeline, self.batch_cosine_pipeline);
        disp.msg_set_buf(encoder, disp.sel_set_buffer, buffer_query, 0, 0);
        disp.msg_set_buf(encoder, disp.sel_set_buffer, buffer_candidates, 0, 1);
        disp.msg_set_buf(encoder, disp.sel_set_buffer, buffer_out, 0, 2);
        var dim_u32: u32 = @intCast(d);
        disp.msg_set_bytes(encoder, disp.sel_set_bytes, &dim_u32, @sizeOf(u32), 3);
        var q_norm_var: f32 = q_norm;
        disp.msg_set_bytes(encoder, disp.sel_set_bytes, &q_norm_var, @sizeOf(f32), 4);

        // One threadgroup per candidate row; 256 threads/group matches the
        // fixed shared-memory arrays declared in batch_cosine_kernel.
        const groups = MTLSize{ .width = n, .height = 1, .depth = 1 };
        const threads = MTLSize{ .width = 256, .height = 1, .depth = 1 };
        disp.msg_dispatch(encoder, disp.sel_dispatch_groups, groups, threads);
        disp.msg_void_void(encoder, disp.sel_end);

        disp.commitAndWait(cmd_buf);
        try disp.copyBufferToHost(buffer_out, out[0..n]);
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

    /// Softmax over `values`, written into `out` (len must equal values.len).
    /// Numerically stable: the row max is subtracted before exp (computed
    /// host-side — demo-grade, not a performance path). GPU does the exp map
    /// (softmax_kernel) and the per-element divide (softmax_norm_kernel); the
    /// partition-function sum is folded on the host since `values` is already
    /// resident there (cheap, honest for a demo kernel). Falls back to the host
    /// implementation in vector_ops.zig if Metal dispatch fails. Demo-grade;
    /// CUDA/Vulkan/ANE remain non-goals.
    pub fn runSoftmax(self: *MetalContext, values: []const f32, out: []f32) !void {
        if (comptime builtin.target.os.tag != .macos) return error.NotSupported;
        if (!self.initialized) return error.NotInitialized;
        if (self.softmax_pipeline == null or self.softmax_norm_pipeline == null) return error.NotInitialized;
        if (values.len == 0) return;
        if (out.len != values.len) return error.DimensionMismatch;

        // Row max (host-side) for numerical stability.
        var max_val: f32 = values[0];
        for (values[1..]) |v| {
            if (v > max_val) max_val = v;
        }

        // Partition-function sum on the host (values already resident here).
        var sum: f32 = 0;
        for (values) |v| sum += @exp(v - max_val);
        if (sum == 0) {
            @memset(out, 0);
            return;
        }

        const d = MetalDispatch.load();
        const byte_len = values.len * @sizeOf(f32);

        const buffer_in = d.msg_bytes(self.device, d.sel_new_bytes, values.ptr, byte_len, 0) orelse
            return error.BufferAllocationFailed;
        defer d.release(buffer_in);
        // exp(x-max) lands here; owned by runSoftmax for the whole dispatch.
        const buffer_exp = d.msg_len(self.device, d.sel_new_length, byte_len, 0) orelse
            return error.BufferAllocationFailed;
        defer d.release(buffer_exp);

        // Scalar uniform buffers for the kernel's `constant float&` params.
        const buffer_max = d.msg_bytes(self.device, d.sel_new_bytes, &max_val, @sizeOf(f32), 0) orelse
            return error.BufferAllocationFailed;
        defer d.release(buffer_max);
        const buffer_sum = d.msg_bytes(self.device, d.sel_new_bytes, &sum, @sizeOf(f32), 0) orelse
            return error.BufferAllocationFailed;
        defer d.release(buffer_sum);

        const cmd_buf = d.msg_void_id(self.queue, d.sel_command_buffer) orelse
            return error.CommandBufferCreationFailed;
        // softmax_kernel: in=values, max=uniform(max_val), out=exp(x-max)
        encodeMapKernel(d, self.softmax_pipeline, cmd_buf, buffer_in, buffer_max, buffer_exp, values.len) catch |e| {
            return e;
        };

        // softmax_norm_kernel: in=exp(x-max), norm=uniform(sum), out=softmax
        const buffer_out = d.msg_bytes(self.device, d.sel_new_bytes, out.ptr, byte_len, 0) orelse
            return error.BufferAllocationFailed;
        defer d.release(buffer_out);
        encodeMapKernel(d, self.softmax_norm_pipeline, cmd_buf, buffer_exp, buffer_sum, buffer_out, values.len) catch |e| {
            return e;
        };

        d.commitAndWait(cmd_buf);
        try d.copyBufferToHost(buffer_out, out[0..values.len]);
    }
};

pub var g_metal_context = MetalContext{};

/// Skip the calling test when Metal is unavailable (off-macOS, or macOS
/// without a usable device). Mirrors the active-backend pattern in
/// vector_ops.zig: the GPU path is exercised only where it can actually run;
/// headless/CI stays green because the test returns without asserting.
fn ensureMetalInitialized() bool {
    if (comptime builtin.target.os.tag != .macos) return false;
    if (g_metal_context.initialized) return true;
    g_metal_context.init(std.heap.page_allocator) catch return false;
    return g_metal_context.initialized;
}

test "metal reduceSum matches scalar reference" {
    if (!ensureMetalInitialized()) return;
    const values = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75, 1.5, 0.0, 4.0, -2.0, 0.125, 1.0, -3.0 };
    var expected: f32 = 0;
    for (values) |v| expected += v;
    const got = try g_metal_context.runReduceSum(&values);
    try std.testing.expectApproxEqAbs(expected, got, 1e-2);
}

test "metal map+reduce dot matches scalar reference" {
    if (!ensureMetalInitialized()) return;
    const a = [_]f32{ 1.0, 2.0, -0.5, 0.25, 4.0, -1.0, 3.0, 0.5, 1.25, -3.0 };
    const b = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75, 1.5, 0.0, 4.0, -2.0, 0.125 };
    var expected: f32 = 0;
    for (a, b) |x, y| expected += x * y;
    const got = try g_metal_context.runMapAndReduce(
        g_metal_context.dot_pipeline,
        a.len,
        &a,
        &b,
    );
    try std.testing.expectApproxEqAbs(expected, got, 1e-2);
}

test "metal fused cosine returns correct ab/aa/bb" {
    if (!ensureMetalInitialized()) return;
    const a = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75, 1.5, 0.0, 4.0, -2.0, 0.125 };
    const b = [_]f32{ 1.0, 2.0, -0.5, 0.25, 4.0, -1.0, 3.0, 0.5, 1.25, -3.0 };
    var ref_ab: f32 = 0;
    var ref_aa: f32 = 0;
    var ref_bb: f32 = 0;
    for (a, b) |x, y| {
        ref_ab += x * y;
        ref_aa += x * x;
        ref_bb += y * y;
    }
    const sums = try g_metal_context.runCosineFused(a.len, &a, &b);
    try std.testing.expectApproxEqAbs(ref_ab, sums.ab, 1e-2);
    try std.testing.expectApproxEqAbs(ref_aa, sums.aa, 1e-2);
    try std.testing.expectApproxEqAbs(ref_bb, sums.bb, 1e-2);
}

test "metal batched cosine fused matches scalar reference" {
    if (!ensureMetalInitialized()) return;
    const query = [_]f32{ 0.5, -1.0, 2.25, 3.0, -0.75 };
    const c0 = [_]f32{ 1.0, 2.0, -0.5, 0.25, 4.0 };
    const c1 = [_]f32{ 0.0, 1.0, 0.0, -1.0, 2.0 };
    const c2 = [_]f32{ 2.25, 3.0, -0.75, 1.5, 0.0 };
    const n = 3;
    const d = query.len;
    const flat = [_]f32{
        c0[0], c0[1], c0[2], c0[3], c0[4],
        c1[0], c1[1], c1[2], c1[3], c1[4],
        c2[0], c2[1], c2[2], c2[3], c2[4],
    };
    var out: [n]f32 = undefined;
    try g_metal_context.runBatchCosineFused(&query, &flat, n, d, &out);

    var expected: [n]f32 = undefined;
    const candidates = [_][]const f32{ &c0, &c1, &c2 };
    for (candidates, 0..) |cand, r| {
        var qc: f32 = 0;
        var qq: f32 = 0;
        var cc: f32 = 0;
        for (query, cand) |qv, cv| {
            qc += qv * cv;
            qq += qv * qv;
            cc += cv * cv;
        }
        expected[r] = if (qq == 0 or cc == 0) 0 else qc / (@sqrt(qq) * @sqrt(cc));
    }
    for (expected, out) |exp, got| {
        try std.testing.expectApproxEqAbs(exp, got, 1e-2);
    }
}

test {
    std.testing.refAllDecls(@This());
}
