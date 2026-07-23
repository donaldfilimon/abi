//! Metal Objective-C runtime FFI helpers (macOS-gated).
//! Package-private building blocks for metal_kernels.zig / metal_shared.zig.
const builtin = @import("builtin");
const std = @import("std");

// The objc runtime symbols only exist on macOS. Declare the externs only when
// targeting macOS so cross-compiles (Linux/Windows) don't emit undefined
// symbols at link time. Off-macOS this is an empty struct and the objc-using
// function bodies are comptime-dead (see the `comptime` early returns below).
pub const objc = if (builtin.target.os.tag == .macos) struct {
    pub extern fn objc_getClass(name: [*c]const u8) ?*anyopaque;
    pub extern fn sel_registerName(name: [*c]const u8) ?*anyopaque;
    pub extern fn objc_msgSend() void;
} else struct {};

pub const MTLCreateSystemDefaultDeviceFn = fn () callconv(.c) ?*anyopaque;
pub const MTLCreateSystemDefaultDevice: ?*const MTLCreateSystemDefaultDeviceFn = if (builtin.target.os.tag == .macos)
    &struct {
        extern fn MTLCreateSystemDefaultDevice() ?*anyopaque;
    }.MTLCreateSystemDefaultDevice
else
    null;

pub const MTLSize = extern struct {
    width: usize,
    height: usize,
    depth: usize,
};

pub const MsgSendVoidRetId = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
pub const MsgSendVoidRetVoid = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) void;
pub const MsgSendIdRetId = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
pub const MsgSendIdErrRetId = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
pub const MsgSendIdIdErrRetId = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
pub const MsgSendPtrUsizeUsizeRetId = *const fn (?*anyopaque, ?*anyopaque, ?*const anyopaque, usize, usize) callconv(.c) ?*anyopaque;
pub const MsgSendUsizeUsizeRetId = *const fn (?*anyopaque, ?*anyopaque, usize, usize) callconv(.c) ?*anyopaque;
pub const MsgSendVoidRetPtr = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque;
pub const MsgSendIdUsizeUsizeRetVoid = *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque, usize, usize) callconv(.c) void;
pub const MsgSendPtrUsizeUsizeRetVoid = *const fn (?*anyopaque, ?*anyopaque, ?*const anyopaque, usize, usize) callconv(.c) void;
pub const MsgSendMtlSizeMtlSizeRetVoid = *const fn (?*anyopaque, ?*anyopaque, MTLSize, MTLSize) callconv(.c) void;

/// Shared ObjC msgSend casts + Metal selectors used by every dispatch path.
/// Constructed once per call site via `load()` so kernels share one helper path
/// instead of duplicating cast/selector boilerplate.
pub const MetalDispatch = struct {
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

    pub fn load() MetalDispatch {
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

    pub fn release(self: MetalDispatch, obj: ?*anyopaque) void {
        self.msg_void_void(obj, self.sel_release);
    }

    pub fn commitAndWait(self: MetalDispatch, cmd_buf: ?*anyopaque) void {
        self.msg_void_void(cmd_buf, self.sel_commit);
        self.msg_void_void(cmd_buf, self.sel_wait);
    }

    pub fn readF32(self: MetalDispatch, buffer: ?*anyopaque) !f32 {
        const ptr = self.msg_contents(buffer, self.sel_contents) orelse return error.ReadBufferFailed;
        return @as([*]f32, @ptrCast(@alignCast(ptr)))[0];
    }

    pub fn copyBufferToHost(self: MetalDispatch, buffer: ?*anyopaque, dest: []f32) !void {
        const ptr = self.msg_contents(buffer, self.sel_contents) orelse return error.ReadBufferFailed;
        @memcpy(dest, @as([*]f32, @ptrCast(@alignCast(ptr)))[0..dest.len]);
    }

    /// Encode one 256-wide reduce pass into an existing command buffer.
    /// Returns the newly allocated partials buffer (caller owns) and its length.
    pub fn encodeReducePass(
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
    pub fn encodeReduceToScalar(
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

pub fn createNSString(allocator: std.mem.Allocator, str: []const u8) !?*anyopaque {
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

test {
    @import("std").testing.refAllDecls(@This());
}
