//! ABI Framework - WebAssembly Bindings
//!
//! Exports ABI functionality for use in WASM environments (browser/Node.js).
//! Uses the shared ABI C-compatible interface where possible.

const std = @import("std");
const abi = @import("abi");

// Use general purpose allocator for WASM environment
// Use general purpose allocator for WASM environment
var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = false }){};
const allocator = gpa.allocator();

// Global framework instance
var global_framework: ?abi.Framework = null;

/// Initialize the framework.
/// Returns 0 on success, or an error code.
export fn abi_init() i32 {
    if (global_framework != null) return -1; // Already initialized

    // Initialize with default options
    // In a real binding, we'd accept configuration from the host
    global_framework = abi.createDefaultFramework(allocator) catch {
        return -2;
    };

    return 0;
}

/// Shutdown the framework and cleanup resources.
export fn abi_shutdown() void {
    if (global_framework) |*fw| {
        fw.deinit();
        global_framework = null;
    }
}

/// Get the version string length.
export fn abi_version_len() usize {
    return abi.version().len;
}

/// Write the version string into the provided buffer.
/// Returns actual bytes written.
export fn abi_version_get(ptr: [*]u8, len: usize) usize {
    const v = abi.version();
    const copy_len = @min(len, v.len);
    @memcpy(ptr[0..copy_len], v[0..copy_len]);
    return copy_len;
}

/// Allocate memory in the WASM linear memory.
/// This is required for passing string/data from JS to WASM.
export fn abi_alloc(len: usize) ?[*]u8 {
    const slice = allocator.alloc(u8, len) catch return null;
    return slice.ptr;
}

/// Free memory previously allocated by abi_alloc.
export fn abi_free(ptr: [*]u8, len: usize) void {
    allocator.free(ptr[0..len]);
}

pub fn panic(msg: []const u8, error_return_trace: ?*std.builtin.StackTrace, ret_addr: ?usize) noreturn {
    _ = msg;
    _ = error_return_trace;
    _ = ret_addr;
    @trap();
}
