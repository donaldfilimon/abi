const std = @import("std");
const connectors = @import("mod.zig");

pub const HuggingFaceError = error{
    MissingApiToken,
};

pub const Config = struct {
    api_token: []u8,
    base_url: []u8,

    pub fn authHeader(self: *const Config, allocator: std.mem.Allocator) !connectors.AuthHeader {
        return connectors.buildBearerHeader(allocator, self.api_token);
    }

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.api_token);
        allocator.free(self.base_url);
        self.* = undefined;
    }
};

pub fn loadFromEnv(allocator: std.mem.Allocator) !Config {
    const api_token = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_HF_API_TOKEN",
        "HF_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    })) orelse return HuggingFaceError.MissingApiToken;

    const base_url = (try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_HF_BASE_URL",
    })) orelse try allocator.dupe(u8, "https://api-inference.huggingface.co");

    return .{
        .api_token = api_token,
        .base_url = base_url,
    };
}
