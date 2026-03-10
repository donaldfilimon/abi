//! Metal compute pipeline, kernel compilation, dispatch, and execution.
//!
//! Handles compiling Metal shading language kernels, launching them
//! synchronously or asynchronously, and synchronization primitives.

const std = @import("std");
const types = @import("../kernel_types");
const s = @import("metal_state");
const metal_types = @import("metal_types");

/// Safely cast an opaque pointer to a MetalKernel pointer with validation.
fn safeCastToKernel(ptr: ?*anyopaque) ?*s.MetalKernel {
    const p = ptr orelse return null;
    const safe_kernel: *s.SafeMetalKernel = @ptrCast(@alignCast(p));
    if (safe_kernel.magic != s.kernel_magic) {
        std.log.err("Invalid MetalKernel pointer: magic mismatch (expected 0x{x}, got 0x{x})", .{ s.kernel_magic, safe_kernel.magic });
        return null;
    }
    return &safe_kernel.inner;
}

/// Safely cast a const opaque pointer to a MetalBuffer pointer with validation.
fn safeCastToBufferConst(ptr: ?*const anyopaque) ?*const s.MetalBuffer {
    const p = ptr orelse return null;
    const safe_buffer: *const s.SafeMetalBuffer = @ptrCast(@alignCast(p));
    if (safe_buffer.magic != s.buffer_magic) {
        std.log.err("Invalid MetalBuffer pointer: magic mismatch (expected 0x{x}, got 0x{x})", .{ s.buffer_magic, safe_buffer.magic });
        return null;
    }
    return &safe_buffer.inner;
}

/// Create an NSString from a Zig string slice.
fn createNSString(str: []const u8) s.MetalError!s.ID {
    const get_class = s.objc_getClass orelse return s.MetalError.NSStringCreationFailed;
    const ns_class = s.nsstring_class orelse blk: {
        const cls = get_class("NSString");
        s.nsstring_class = cls;
        break :blk cls;
    };

    var stack_buf: [4096]u8 = undefined;
    const c_str: [*:0]const u8 = if (str.len < stack_buf.len) blk: {
        @memcpy(stack_buf[0..str.len], str);
        stack_buf[str.len] = 0;
        break :blk stack_buf[0..str.len :0];
    } else blk: {
        @memcpy(stack_buf[0 .. stack_buf.len - 1], str[0 .. stack_buf.len - 1]);
        stack_buf[stack_buf.len - 1] = 0;
        break :blk stack_buf[0 .. stack_buf.len - 1 :0];
    };

    const msg_send_str: *const fn (?s.Class, s.SEL, [*:0]const u8) callconv(.c) s.ID = @ptrCast(s.objc_msgSend);
    const result = msg_send_str(ns_class, s.sel_stringWithUTF8String, c_str);
    if (result == null) {
        return s.MetalError.NSStringCreationFailed;
    }

    return result;
}

/// Create an NSString from a null-terminated C string.
fn createNSStringFromCStr(c_str: [*:0]const u8) s.MetalError!s.ID {
    const get_class = s.objc_getClass orelse return s.MetalError.NSStringCreationFailed;
    const ns_class = s.nsstring_class orelse blk: {
        const cls = get_class("NSString");
        s.nsstring_class = cls;
        break :blk cls;
    };

    const msg_send_str: *const fn (?s.Class, s.SEL, [*:0]const u8) callconv(.c) s.ID = @ptrCast(s.objc_msgSend);
    const result = msg_send_str(ns_class, s.sel_stringWithUTF8String, c_str);
    if (result == null) {
        return s.MetalError.NSStringCreationFailed;
    }

    return result;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    if (!s.metal_initialized or s.metal_device == null) {
        return types.KernelError.CompilationFailed;
    }

    const device = s.metal_device.?;

    // Create NSString from source code
    const source_nsstring = createNSString(source.source) catch {
        std.log.err("Failed to create NSString from kernel source", .{});
        return types.KernelError.CompilationFailed;
    };
    defer {
        if (s.objc_msgSend_void) |release_fn| {
            release_fn(source_nsstring, s.sel_release);
        }
    }

    // Create library from source
    const msg_send_lib: *const fn (s.ID, s.SEL, s.ID, s.ID, *s.ID) callconv(.c) s.ID = @ptrCast(s.objc_msgSend);
    var compile_error: s.ID = null;
    const library = msg_send_lib(device, s.sel_newLibraryWithSource, source_nsstring, null, &compile_error);

    if (library == null) {
        if (compile_error != null) {
            const sel_fn = s.sel_registerName orelse {
                std.log.err("Failed to create Metal library from source (unknown error)", .{});
                return types.KernelError.CompilationFailed;
            };
            const sel_desc = sel_fn("localizedDescription");
            const msg_send = s.objc_msgSend orelse {
                std.log.err("Failed to create Metal library from source (unknown error)", .{});
                return types.KernelError.CompilationFailed;
            };
            const desc_nsstring = msg_send(compile_error, sel_desc);
            if (desc_nsstring != null) {
                const utf8_fn: *const fn (s.ID, s.SEL) callconv(.c) ?[*:0]const u8 = @ptrCast(s.objc_msgSend);
                const utf8_ptr = utf8_fn(desc_nsstring, s.sel_UTF8String);
                if (utf8_ptr) |ptr| {
                    std.log.err("Metal compilation error: {s}", .{ptr});
                }
            }
        }
        std.log.err("Failed to create Metal library from source", .{});
        return types.KernelError.CompilationFailed;
    }

    // Get the function by entry point name
    const entry_point_str: [*:0]const u8 = if (source.entry_point.len > 0) blk: {
        var buf: [256]u8 = undefined;
        const len = @min(source.entry_point.len, buf.len - 1);
        @memcpy(buf[0..len], source.entry_point[0..len]);
        buf[len] = 0;
        break :blk buf[0..len :0];
    } else "main";

    const entry_point_nsstring = createNSStringFromCStr(entry_point_str) catch {
        std.log.err("Failed to create NSString for entry point", .{});
        if (s.objc_msgSend_void) |release_fn| {
            release_fn(library, s.sel_release);
        }
        return types.KernelError.CompilationFailed;
    };
    defer {
        if (s.objc_msgSend_void) |release_fn| {
            release_fn(entry_point_nsstring, s.sel_release);
        }
    }

    const msg_send_ptr = s.objc_msgSend_ptr orelse return types.KernelError.CompilationFailed;
    const function = msg_send_ptr(library, s.sel_newFunctionWithName, entry_point_nsstring);
    if (function == null) {
        std.log.err("Failed to get Metal function '{s}' from library", .{source.entry_point});
        if (s.objc_msgSend_void) |release_fn| {
            release_fn(library, s.sel_release);
        }
        return types.KernelError.CompilationFailed;
    }

    // Create compute pipeline state
    var pipeline_error: s.ID = null;
    const msg_send_pipeline: *const fn (s.ID, s.SEL, s.ID, *s.ID) callconv(.c) s.ID = @ptrCast(s.objc_msgSend);
    const pipeline_state = msg_send_pipeline(device, s.sel_newComputePipelineStateWithFunction, function, &pipeline_error);
    if (pipeline_state == null) {
        std.log.err("Failed to create Metal compute pipeline state", .{});
        if (s.objc_msgSend_void) |release_fn| {
            release_fn(function, s.sel_release);
            release_fn(library, s.sel_release);
        }
        return types.KernelError.CompilationFailed;
    }

    const safe_kernel = allocator.create(s.SafeMetalKernel) catch {
        if (s.objc_msgSend_void) |release_fn| {
            release_fn(pipeline_state, s.sel_release);
            release_fn(function, s.sel_release);
            release_fn(library, s.sel_release);
        }
        return types.KernelError.CompilationFailed;
    };
    safe_kernel.* = .{
        .magic = s.kernel_magic,
        .inner = .{
            .pipeline_state = pipeline_state,
            .library = library,
            .function = function,
        },
    };

    std.log.debug("Metal kernel compiled successfully: entry_point={s}", .{source.entry_point});
    return safe_kernel;
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    _ = allocator;

    if (!s.metal_initialized or s.metal_command_queue == null) {
        return types.KernelError.LaunchFailed;
    }

    const msg_send = s.objc_msgSend orelse return types.KernelError.LaunchFailed;
    const msg_send_void = s.objc_msgSend_void orelse return types.KernelError.LaunchFailed;
    const msg_send_void_ptr = s.objc_msgSend_void_ptr orelse return types.KernelError.LaunchFailed;
    const msg_send_void_ptr_int_int = s.objc_msgSend_void_ptr_int_int orelse return types.KernelError.LaunchFailed;

    const kernel = safeCastToKernel(kernel_handle) orelse {
        std.log.err("launchKernel: Invalid kernel handle (null or corrupted)", .{});
        return types.KernelError.LaunchFailed;
    };

    // Validate grid and block sizes
    if (config.grid_dim[0] == 0 or config.grid_dim[1] == 0 or config.grid_dim[2] == 0) {
        std.log.err("launchKernel: Invalid grid size (zero dimension)", .{});
        return types.KernelError.LaunchFailed;
    }
    if (config.block_dim[0] == 0 or config.block_dim[1] == 0 or config.block_dim[2] == 0) {
        std.log.err("launchKernel: Invalid block size (zero dimension)", .{});
        return types.KernelError.LaunchFailed;
    }

    // Create command buffer
    const command_buffer = msg_send(s.metal_command_queue, s.sel_commandBuffer);
    if (command_buffer == null) {
        std.log.err("Failed to create Metal command buffer", .{});
        return types.KernelError.LaunchFailed;
    }

    // Create compute command encoder
    const encoder = msg_send(command_buffer, s.sel_computeCommandEncoder);
    if (encoder == null) {
        std.log.err("Failed to create Metal compute command encoder", .{});
        return types.KernelError.LaunchFailed;
    }

    // Set pipeline state
    msg_send_void_ptr(encoder, s.sel_setComputePipelineState, kernel.pipeline_state);

    // Set buffers
    for (args, 0..) |arg, i| {
        if (arg != null) {
            const buffer_wrapper = safeCastToBufferConst(arg) orelse {
                std.log.err("launchKernel: Invalid buffer argument at index {} (null or corrupted)", .{i});
                return types.KernelError.LaunchFailed;
            };
            msg_send_void_ptr_int_int(encoder, s.sel_setBuffer, buffer_wrapper.buffer, 0, @intCast(i));
        }
    }

    // Dispatch threads
    const grid_size = s.MTLSize.init(
        config.grid_dim[0] * config.block_dim[0],
        config.grid_dim[1] * config.block_dim[1],
        config.grid_dim[2] * config.block_dim[2],
    );
    const threads_per_group = s.MTLSize.init(
        config.block_dim[0],
        config.block_dim[1],
        config.block_dim[2],
    );

    const dispatch_fn: *const fn (s.ID, s.SEL, s.MTLSize, s.MTLSize) callconv(.c) void = @ptrCast(s.objc_msgSend);
    dispatch_fn(encoder, s.sel_dispatchThreads, grid_size, threads_per_group);

    // End encoding, commit, wait
    msg_send_void(encoder, s.sel_endEncoding);
    msg_send_void(command_buffer, s.sel_commit);
    msg_send_void(command_buffer, s.sel_waitUntilCompleted);

    std.log.debug("Metal kernel launched: grid=({},{},{}), block=({},{},{})", .{
        config.grid_dim[0],  config.grid_dim[1],  config.grid_dim[2],
        config.block_dim[0], config.block_dim[1], config.block_dim[2],
    });
}

/// Launch a kernel asynchronously without waiting for completion.
pub fn launchKernelAsync(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!s.ID {
    _ = allocator;

    if (!s.metal_initialized or s.metal_command_queue == null) {
        return types.KernelError.LaunchFailed;
    }

    const msg_send = s.objc_msgSend orelse return types.KernelError.LaunchFailed;
    const msg_send_void = s.objc_msgSend_void orelse return types.KernelError.LaunchFailed;
    const msg_send_void_ptr = s.objc_msgSend_void_ptr orelse return types.KernelError.LaunchFailed;
    const msg_send_void_ptr_int_int = s.objc_msgSend_void_ptr_int_int orelse return types.KernelError.LaunchFailed;

    const kernel = safeCastToKernel(kernel_handle) orelse {
        std.log.err("launchKernelAsync: Invalid kernel handle", .{});
        return types.KernelError.LaunchFailed;
    };

    const command_buffer = msg_send(s.metal_command_queue, s.sel_commandBuffer);
    if (command_buffer == null) {
        return types.KernelError.LaunchFailed;
    }

    // Retain command buffer for tracking
    if (s.objc_msgSend_void) |retain_fn| {
        retain_fn(command_buffer, s.sel_retain);
    }

    const encoder = msg_send(command_buffer, s.sel_computeCommandEncoder);
    if (encoder == null) {
        if (s.objc_msgSend_void) |release_fn| {
            release_fn(command_buffer, s.sel_release);
        }
        return types.KernelError.LaunchFailed;
    }

    msg_send_void_ptr(encoder, s.sel_setComputePipelineState, kernel.pipeline_state);

    for (args, 0..) |arg, i| {
        if (arg != null) {
            const buffer_wrapper = safeCastToBufferConst(arg) orelse continue;
            msg_send_void_ptr_int_int(encoder, s.sel_setBuffer, buffer_wrapper.buffer, 0, @intCast(i));
        }
    }

    const grid_size = s.MTLSize.init(
        config.grid_dim[0] * config.block_dim[0],
        config.grid_dim[1] * config.block_dim[1],
        config.grid_dim[2] * config.block_dim[2],
    );
    const threads_per_group = s.MTLSize.init(
        config.block_dim[0],
        config.block_dim[1],
        config.block_dim[2],
    );

    const dispatch_fn: *const fn (s.ID, s.SEL, s.MTLSize, s.MTLSize) callconv(.c) void = @ptrCast(s.objc_msgSend);
    dispatch_fn(encoder, s.sel_dispatchThreads, grid_size, threads_per_group);

    msg_send_void(encoder, s.sel_endEncoding);
    msg_send_void(command_buffer, s.sel_commit);

    // Track pending command buffer
    if (s.pending_buffers_allocator) |alloc| {
        s.pending_command_buffers.append(alloc, command_buffer) catch |err| {
            std.log.debug("Failed to track Metal command buffer: {t}", .{err});
        };
    }

    return command_buffer;
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    const kernel = safeCastToKernel(kernel_handle) orelse {
        std.log.err("destroyKernel: Invalid kernel handle (null or corrupted), skipping destruction", .{});
        return;
    };

    if (s.objc_msgSend_void) |release_fn| {
        if (kernel.pipeline_state != null) release_fn(kernel.pipeline_state, s.sel_release);
        if (kernel.function != null) release_fn(kernel.function, s.sel_release);
        if (kernel.library != null) release_fn(kernel.library, s.sel_release);
    }

    const safe_kernel: *s.SafeMetalKernel = @fieldParentPtr("inner", kernel);
    safe_kernel.magic = 0;

    allocator.destroy(safe_kernel);
}

/// Synchronize with the GPU. Blocks until all pending commands are complete.
pub fn synchronize() void {
    const msg_send_void = s.objc_msgSend_void orelse return;

    for (s.pending_command_buffers.items) |cmd_buffer| {
        if (cmd_buffer != null) {
            msg_send_void(cmd_buffer, s.sel_waitUntilCompleted);
            msg_send_void(cmd_buffer, s.sel_release);
        }
    }

    if (s.pending_buffers_allocator) |alloc| {
        s.pending_command_buffers.clearRetainingCapacity();
        _ = alloc;
    }

    std.log.debug("Metal synchronize complete", .{});
}

/// Wait for a specific command buffer to complete.
pub fn waitForCommandBuffer(cmd_buffer: s.ID) void {
    if (cmd_buffer == null) return;

    const msg_send_void = s.objc_msgSend_void orelse return;

    msg_send_void(cmd_buffer, s.sel_waitUntilCompleted);
    msg_send_void(cmd_buffer, s.sel_release);

    for (s.pending_command_buffers.items, 0..) |buf, i| {
        if (buf == cmd_buffer) {
            _ = s.pending_command_buffers.swapRemove(i);
            break;
        }
    }
}
