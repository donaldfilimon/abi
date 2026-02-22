const std = @import("std");

pub const CommandContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
};
