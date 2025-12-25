const std = @import("std");
const connectors = @import("mod.zig");

pub const OpenAIError = error{
    MissingApiKey,
};

pub const Config = struct {
    api_key: []u8,
    base_url: []u8,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.api_key);
        allocator.free(self.base_url);
        self.* = undefined;
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const api_key = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    })) orelse return OpenAIError.MissingApiKey;

    const base_url = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENAI_BASE_URL",
        "OPENAI_BASE_URL",
    })) orelse try allocator.dupe(u8, "https://api.openai.com/v1");

    return .{
        .api_key = api_key,
        .base_url = base_url,
    };
}
