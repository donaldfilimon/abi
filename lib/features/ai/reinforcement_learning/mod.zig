//! Reinforcement learning facade that re-exports the legacy implementation.
//! Placeholder surface to keep module structure consistent while the
//! full RL implementation is refactored.

const std = @import("std");

pub fn init(_: std.mem.Allocator) !void {}

pub fn deinit() void {}
