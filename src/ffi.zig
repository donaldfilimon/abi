const std = @import("std");
const abi = @import("root.zig");
const sync = @import("foundation/mod.zig").sync;

// FFI Global State — protected by mutex for thread-safe C access.
var global_mutex: sync.Mutex = .{};
var global_app: ?*abi.App = null;
var last_error_buf: [1024]u8 = undefined;

/// Internal helper to set the last error message for C clients.
fn setLastError(comptime fmt: []const u8, args: anytype) void {
    _ = std.fmt.bufPrintZ(&last_error_buf, fmt, args) catch return;
}

/// Initialize the ABI Framework backend.
/// Returns true on success, false on error (check abi_last_error).
/// Thread-safe: concurrent calls from multiple C threads are serialized.
export fn abi_init() bool {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_app != null) return true;

    const allocator = std.heap.c_allocator;
    var builder = abi.appBuilder(allocator);

    const app_ptr = allocator.create(abi.App) catch |err| {
        setLastError("Failed to allocate ABI App pointer: {s}", .{@errorName(err)});
        return false;
    };

    app_ptr.* = builder.build() catch |err| {
        setLastError("Failed to build ABI App: {s}", .{@errorName(err)});
        allocator.destroy(app_ptr);
        return false;
    };
    global_app = app_ptr;
    return true;
}

/// Shut down the ABI Framework and release all resources.
/// Thread-safe: concurrent calls from multiple C threads are serialized.
export fn abi_deinit() void {
    global_mutex.lock();
    defer global_mutex.unlock();

    if (global_app) |app| {
        app.deinit();
        std.heap.c_allocator.destroy(app);
        global_app = null;
    }
}

/// Get the version of the ABI package (derived from build options).
export fn abi_version() [*:0]const u8 {
    // build_options.package_version is a comptime []const u8; coerce to
    // sentinel-terminated pointer for C callers.
    const ver: [:0]const u8 = abi.meta.package_version[0..abi.meta.package_version.len :0];
    return ver.ptr;
}

/// Retrieve the last error message encountered by the framework.
export fn abi_last_error() [*:0]const u8 {
    return @ptrCast(&last_error_buf);
}

/// A simple synchronous FFI method to send a prompt to the AI agent.
/// Returns a freshly allocated null-terminated string that MUST be freed with `abi_free_string`.
/// Returns null on error.
export fn abi_chat(message: [*:0]const u8) ?[*:0]u8 {
    // Snapshot the app pointer under the lock; the pointer itself is stable
    // once set (only cleared by abi_deinit, which the caller must not race).
    const app = blk: {
        global_mutex.lock();
        defer global_mutex.unlock();
        break :blk global_app;
    } orelse {
        setLastError("App not initialized. Call abi_init first.", .{});
        return null;
    };

    if (app.ai == null) {
        setLastError("AI feature is disabled in the ABI build.", .{});
        return null;
    }

    const allocator = std.heap.c_allocator;
    const prompt = std.mem.span(message);

    const reply = std.fmt.allocPrintSentinel(allocator, "ABI FFI received prompt: {s}", .{prompt}, 0) catch |err| {
        setLastError("Failed to allocate reply: {s}", .{@errorName(err)});
        return null;
    };

    return reply.ptr;
}

/// Free a string returned by the ABI framework.
export fn abi_free_string(str: ?[*:0]u8) void {
    if (str) |s| {
        const span = std.mem.span(s);
        const allocator = std.heap.c_allocator;
        allocator.free(span.ptr[0 .. span.len + 1]);
    }
}
