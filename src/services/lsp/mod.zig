//! LSP (ZLS) service module.

const std = @import("std");
const client = @import("client.zig");
const config_mod = @import("../../core/config/mod.zig");

pub const Config = config_mod.LspConfig;
pub const Client = client.Client;
pub const Response = client.Response;
pub const types = @import("types.zig");
pub const jsonrpc = @import("jsonrpc.zig");

pub const resolveWorkspaceRoot = client.resolveWorkspaceRoot;
pub const resolvePath = client.resolvePath;
pub const pathToUri = client.pathToUri;

pub const EnvConfig = struct {
    allocator: std.mem.Allocator,
    config: Config,
    owned: std.ArrayListUnmanaged([]const u8),

    pub fn deinit(self: *EnvConfig) void {
        for (self.owned.items) |s| {
            @memset(@constCast(@volatileCast(s)), 0);
            self.allocator.free(s);
        }
        self.owned.deinit(self.allocator);
    }

    fn own(self: *EnvConfig, value: []const u8) ![]const u8 {
        const copy = try self.allocator.dupe(u8, value);
        try self.owned.append(self.allocator, copy);
        return copy;
    }
};

pub fn loadConfigFromEnv(allocator: std.mem.Allocator) !EnvConfig {
    var env = EnvConfig{
        .allocator = allocator,
        .config = Config.defaults(),
        .owned = .empty,
    };
    errdefer env.deinit();

    if (getEnv("ABI_LSP_ZLS_PATH")) |path| {
        env.config.zls_path = try env.own(path);
    }
    if (getEnv("ABI_LSP_ZIG_EXE_PATH")) |path| {
        env.config.zig_exe_path = try env.own(path);
    }
    if (getEnv("ABI_LSP_WORKSPACE_ROOT")) |path| {
        env.config.workspace_root = try env.own(path);
    }
    if (getEnv("ABI_LSP_LOG_LEVEL")) |level| {
        env.config.log_level = try env.own(level);
    }
    if (getEnv("ABI_LSP_ENABLE_SNIPPETS")) |flag| {
        if (parseBool(flag)) |value| {
            env.config.enable_snippets = value;
        }
    }

    return env;
}

fn parseBool(s: []const u8) ?bool {
    if (std.ascii.eqlIgnoreCase(s, "true") or
        std.ascii.eqlIgnoreCase(s, "yes") or
        std.ascii.eqlIgnoreCase(s, "1"))
    {
        return true;
    }
    if (std.ascii.eqlIgnoreCase(s, "false") or
        std.ascii.eqlIgnoreCase(s, "no") or
        std.ascii.eqlIgnoreCase(s, "0"))
    {
        return false;
    }
    return null;
}

fn getEnv(name: []const u8) ?[]const u8 {
    var key_buf: [256]u8 = undefined;
    const key_len = @min(name.len, 255);
    @memcpy(key_buf[0..key_len], name[0..key_len]);
    key_buf[key_len] = 0;
    const key_z: [*:0]const u8 = @ptrCast(&key_buf);
    const ptr = std.c.getenv(key_z) orelse return null;
    return std.mem.span(ptr);
}

test {
    std.testing.refAllDecls(@This());
}
