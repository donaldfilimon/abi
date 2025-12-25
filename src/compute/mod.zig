const std = @import("std");

pub const runtime = @import("runtime/mod.zig");
pub const memory = @import("memory/mod.zig");
pub const concurrency = @import("concurrency/mod.zig");

pub fn init(_: std.mem.Allocator) !void {}

pub fn deinit() void {}
