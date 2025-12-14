//! Transformer module facade to align with new module layout.
//! Currently provides lightweight init/deinit stubs; the legacy implementation
//! remains in `../transformer.zig` for future integration.

const std = @import("std");

pub fn init(_: std.mem.Allocator) !void {}

pub fn deinit() void {}
