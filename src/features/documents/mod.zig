//! Native Documents Parser Module
//!
//! Provides zero-dependency parsers for complex file formats like
//! HTML, DOM trees, and PDF binaries.

pub const html = @import("html.zig");
pub const pdf = @import("pdf.zig");

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
