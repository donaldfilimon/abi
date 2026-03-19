//! Native Documents Parser Module
//!
//! Provides zero-dependency parsers for complex file formats like
//! HTML, DOM trees, and PDF binaries.

pub const html = @import("html.zig");
pub const pdf = @import("pdf.zig");

const std = @import("std");

pub const DocumentsError = error{
    ParseFailed,
    UnsupportedFormat,
    InvalidInput,
    OutOfMemory,
};

pub const Error = DocumentsError;

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
    return false;
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
