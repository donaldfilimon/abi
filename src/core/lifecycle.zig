const std = @import("std");
const errors = @import("errors.zig");
const logging = @import("logging.zig");

var core_initialized: bool = false;
var global_allocator: ?std.mem.Allocator = null;

/// Initialize the core system.
pub fn init(allocator: std.mem.Allocator) errors.AbiError!void {
    if (core_initialized) {
        return errors.AbiError.SystemAlreadyInitialized;
    }

    global_allocator = allocator;
    core_initialized = true;

    try logging.log.init(allocator);
    logging.log.info("Core system initialized", .{});
}

/// Deinitialize the core system.
pub fn deinit() void {
    if (!core_initialized) return;

    logging.log.info("Core system shutting down", .{});

    logging.log.deinit();

    global_allocator = null;
    core_initialized = false;
}

/// Get the global allocator passed to init().
pub fn getAllocator() std.mem.Allocator {
    if (global_allocator) |alloc| {
        return alloc;
    }
    @panic("Core system not initialized. Call core.init() first.");
}

/// Check if core system is initialized.
pub fn isInitialized() bool {
    return core_initialized;
}
