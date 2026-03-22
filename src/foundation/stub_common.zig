//! Common Stub Utilities
//!
//! Provides shared functionality for stub implementations when modules are disabled.

const std = @import("std");

/// Common error types for disabled modules
pub const CommonError = error{
    ModuleDisabled,
    FeatureNotAvailable,
    InvalidOperation,
};

/// Generic stub return function
pub fn stubReturn(comptime T: type, comptime err: anyerror) T {
    if (T == void) {
        return;
    } else if (@typeInfo(T) == .ErrorUnion) {
        return err;
    } else {
        @compileError("Unsupported return type for stub: " ++ @typeName(T));
    }
}

/// Stub implementation helper for functions that return void
pub fn stubVoid() void {
    // No-op
}

/// Stub implementation helper for functions that return error
pub fn stubError(comptime err: anyerror) anyerror {
    return err;
}

/// Common stub configuration
pub const StubConfig = struct {
    module_name: []const u8,
    description: []const u8,
};

/// Generate a standard stub error message
pub fn errorMessage(config: StubConfig, allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s} is disabled at compile time", .{config.module_name});
}

/// Helper for creating consistent stub structures
pub fn createStub(comptime config: StubConfig) type {
    return struct {
        pub const Error = CommonError;

        pub fn init(_: std.mem.Allocator) Error!@This() {
            return Error.ModuleDisabled;
        }

        pub fn deinit(_: *@This()) void {
            // No-op
        }

        pub fn isAvailable() bool {
            return false;
        }

        pub fn getDescription(allocator: std.mem.Allocator) ![]u8 {
            return errorMessage(config, allocator);
        }
    };
}

test {
    std.testing.refAllDecls(@This());
}
