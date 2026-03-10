//! Native Documents Parser Module
//!
//! Provides zero-dependency parsers for complex file formats like
//! HTML, DOM trees, and PDF binaries.

pub const html = @import("html");
pub const pdf = @import("pdf");

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
