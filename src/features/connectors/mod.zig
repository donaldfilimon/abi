//! Connector configuration loaders and auth helpers.
const std = @import("std");

pub const openai = @import("openai.zig");
pub const huggingface = @import("huggingface.zig");
pub const ollama = @import("ollama.zig");
pub const local_scheduler = @import("local_scheduler.zig");

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return initialized;
}

pub fn getEnvOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    // Create an environment map from the system environment
    var env_map = std.process.Environ.createMap(.{ .block = {} }, allocator) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    defer env_map.deinit();

    // Look up the variable and return a copy if found
    if (env_map.get(name)) |value| {
        return try allocator.dupe(u8, value);
    }
    return null;
}

pub fn getFirstEnvOwned(allocator: std.mem.Allocator, names: []const []const u8) !?[]u8 {
    for (names) |name| {
        if (try getEnvOwned(allocator, name)) |value| {
            return value;
        }
    }
    return null;
}

pub const AuthHeader = struct {
    value: []u8,

    pub fn header(self: *const AuthHeader) std.http.Header {
        return .{ .name = "authorization", .value = self.value };
    }

    pub fn deinit(self: *AuthHeader, allocator: std.mem.Allocator) void {
        allocator.free(self.value);
        self.* = undefined;
    }
};

pub fn buildBearerHeader(allocator: std.mem.Allocator, token: []const u8) !AuthHeader {
    const value = try std.fmt.allocPrint(allocator, "Bearer {s}", .{token});
    return .{ .value = value };
}

pub fn loadOpenAI(allocator: std.mem.Allocator) !openai.Config {
    return openai.loadFromEnv(allocator);
}

pub fn tryLoadOpenAI(allocator: std.mem.Allocator) !?openai.Config {
    return openai.loadFromEnv(allocator) catch |err| switch (err) {
        openai.OpenAIError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadHuggingFace(allocator: std.mem.Allocator) !huggingface.Config {
    return huggingface.loadFromEnv(allocator);
}

pub fn tryLoadHuggingFace(allocator: std.mem.Allocator) !?huggingface.Config {
    return huggingface.loadFromEnv(allocator) catch |err| switch (err) {
        huggingface.HuggingFaceError.MissingApiToken => null,
        else => return err,
    };
}

pub fn loadOllama(allocator: std.mem.Allocator) !ollama.Config {
    return ollama.loadFromEnv(allocator);
}

pub fn loadLocalScheduler(allocator: std.mem.Allocator) !local_scheduler.Config {
    return local_scheduler.loadFromEnv(allocator);
}

test "connectors init toggles state" {
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}
