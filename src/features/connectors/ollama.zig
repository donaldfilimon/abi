const std = @import("std");

const connectors = @import("mod.zig");

pub const Config = struct {
    host: []u8,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.host);
        self.* = undefined;
    }

    pub fn endpoint(self: *const Config, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
        const prefix = if (path.len > 0 and path[0] == '/') "" else "/";
        return std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ self.host, prefix, path });
    }

    pub fn generateUrl(self: *const Config, allocator: std.mem.Allocator) ![]u8 {
        return self.endpoint(allocator, "/api/generate");
    }

    pub fn chatUrl(self: *const Config, allocator: std.mem.Allocator) ![]u8 {
        return self.endpoint(allocator, "/api/chat");
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const host = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OLLAMA_HOST",
        "OLLAMA_HOST",
    })) orelse try allocator.dupe(u8, "http://127.0.0.1:11434");
    return .{ .host = host };
}

test "ollama endpoint join" {
    var config = Config{ .host = try std.testing.allocator.dupe(u8, "http://localhost:11434") };
    defer config.deinit(std.testing.allocator);

    const url = try config.generateUrl(std.testing.allocator);
    defer std.testing.allocator.free(url);
    try std.testing.expectEqualStrings("http://localhost:11434/api/generate", url);
}
