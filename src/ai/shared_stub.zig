//! Shared AI Stub Utilities
//!
//! Provides common patterns for AI stubs to reduce code duplication.

const std = @import("std");

/// Returns a "Disabled" error for any service.
pub fn disabledError(comptime service_name: []const u8) type {
    _ = service_name;
    return anyerror;
}

pub fn wrapDisabled(comptime err_val: anyerror) type {
    return struct {
        pub fn init(_: std.mem.Allocator, _: anytype) anyerror!*@This() {
            return err_val;
        }
        pub fn deinit(_: *@This()) void {}
    };
}

/// Generic implementation of isEnabled which always returns false.
pub fn isEnabled() bool {
    return false;
}
