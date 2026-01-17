//! Connector configuration loaders and auth helpers.
const std = @import("std");

pub const openai = @import("openai.zig");
pub const huggingface = @import("huggingface.zig");
pub const ollama = @import("ollama.zig");
pub const local_scheduler = @import("local_scheduler.zig");
pub const discord = @import("discord/mod.zig");
pub const anthropic = @import("anthropic.zig");
pub const mistral = @import("mistral.zig");
pub const cohere = @import("cohere.zig");

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

const builtin = @import("builtin");

// libc import for environment access - required for Zig 0.16
const c = @cImport(@cInclude("stdlib.h"));

pub fn getEnvOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    // Zig 0.16: Environment access via libc getenv (build links libc)
    const name_z = allocator.dupeZ(u8, name) catch return error.OutOfMemory;
    defer allocator.free(name_z);

    const value_ptr = c.getenv(name_z.ptr);
    if (value_ptr) |ptr| {
        const value = std.mem.span(ptr);
        return allocator.dupe(u8, value) catch return error.OutOfMemory;
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

pub fn loadDiscord(allocator: std.mem.Allocator) !discord.Config {
    return discord.loadFromEnv(allocator);
}

pub fn tryLoadDiscord(allocator: std.mem.Allocator) !?discord.Config {
    return discord.loadFromEnv(allocator) catch |err| switch (err) {
        discord.DiscordError.MissingBotToken => null,
        else => return err,
    };
}

pub fn loadAnthropic(allocator: std.mem.Allocator) !anthropic.Config {
    return anthropic.loadFromEnv(allocator);
}

pub fn tryLoadAnthropic(allocator: std.mem.Allocator) !?anthropic.Config {
    return anthropic.loadFromEnv(allocator) catch |err| switch (err) {
        anthropic.AnthropicError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadMistral(allocator: std.mem.Allocator) !mistral.Config {
    return mistral.loadFromEnv(allocator);
}

pub fn tryLoadMistral(allocator: std.mem.Allocator) !?mistral.Config {
    return mistral.loadFromEnv(allocator) catch |err| switch (err) {
        mistral.MistralError.MissingApiKey => null,
        else => return err,
    };
}

pub fn loadCohere(allocator: std.mem.Allocator) !cohere.Config {
    return cohere.loadFromEnv(allocator);
}

pub fn tryLoadCohere(allocator: std.mem.Allocator) !?cohere.Config {
    return cohere.loadFromEnv(allocator) catch |err| switch (err) {
        cohere.CohereError.MissingApiKey => null,
        else => return err,
    };
}

test "connectors init toggles state" {
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}
