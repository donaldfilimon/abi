//! Optimization module stub.
//! Provides placeholders to keep the build working; concrete optimizers live
//! in `optimizers/` and can be wired in here later.

const std = @import("std");

pub fn init(_: std.mem.Allocator) !void {}

pub fn deinit() void {}
