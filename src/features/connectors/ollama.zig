const std = @import("std");
const connectors = @import("mod.zig");

pub const Config = struct {
    host: []u8,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.host);
        self.* = undefined;
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const host = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OLLAMA_HOST",
        "OLLAMA_HOST",
    })) orelse try allocator.dupe(u8, "http://127.0.0.1:11434");

    return .{ .host = host };
}
