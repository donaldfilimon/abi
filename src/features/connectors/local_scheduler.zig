const std = @import("std");
const connectors = @import("mod.zig");

pub const Config = struct {
    url: []u8,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.url);
        self.* = undefined;
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const url = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_LOCAL_SCHEDULER_URL",
        "LOCAL_SCHEDULER_URL",
    })) orelse try allocator.dupe(u8, "http://127.0.0.1:9090");

    return .{ .url = url };
}
