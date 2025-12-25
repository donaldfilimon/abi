const std = @import("std");

pub const agent = @import("agent.zig");
pub const model_registry = @import("model_registry.zig");
pub const training = @import("training/mod.zig");
pub const federated = @import("federated/mod.zig");
pub const transformer = @import("transformer/mod.zig");

pub const Agent = agent.Agent;

pub fn init(_: std.mem.Allocator) !void {}

pub fn deinit() void {}
