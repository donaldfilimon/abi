//! Native Documents Parser Module
//!
//! Provides zero-dependency parsers for complex file formats like
//! HTML, DOM trees, and PDF binaries.

pub const html = @import("html.zig");
pub const pdf = @import("pdf.zig");
pub const types = @import("types.zig");

const std = @import("std");

pub const DocumentsError = types.DocumentsError;
pub const Error = types.Error;

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{ .allocator = allocator, .initialized = true };
    }

    pub fn deinit(self: *Context) void {
        self.initialized = false;
    }
};

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return true; // Module is available when feature flag is enabled
}

test {
    std.testing.refAllDecls(@This());
}

test "Context init and deinit" {
    var ctx = Context.init(std.testing.allocator);
    try std.testing.expect(ctx.initialized);
    ctx.deinit();
    try std.testing.expect(!ctx.initialized);
}

test "isEnabled returns true" {
    try std.testing.expect(isEnabled());
}

test "sub-modules accessible" {
    _ = html;
    _ = pdf;
}
