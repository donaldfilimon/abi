//! LSP (ZLS) service stub.
//!
//! Mirrors the full API of mod.zig, returning error.FeatureDisabled for all operations.

const std = @import("std");

pub const types = @import("types.zig");
pub const jsonrpc = @import("jsonrpc.zig");

/// LSP configuration.
pub const Config = struct {
    zls_path: []const u8 = "zls",
    zig_exe_path: ?[]const u8 = null,
    workspace_root: ?[]const u8 = null,
    log_level: []const u8 = "info",
    enable_snippets: bool = true,

    pub fn defaults() Config {
        return .{};
    }
};

pub const Response = struct {
    json: []u8,
    is_error: bool,
};

pub const Client = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: anytype, _: Config) !Client {
        _ = allocator;
        return error.FeatureDisabled;
    }

    pub fn deinit(self: *Client) void {
        _ = self;
    }

    pub fn workspaceRoot(self: *const Client) []const u8 {
        _ = self;
        return ".";
    }

    pub fn didOpen(self: *Client, _: types.TextDocumentItem) !void {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn hover(self: *Client, _: []const u8, _: types.Position) !Response {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn completion(self: *Client, _: []const u8, _: types.Position) !Response {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn definition(self: *Client, _: []const u8, _: types.Position) !Response {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn references(self: *Client, _: []const u8, _: types.Position, _: bool) !Response {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn rename(self: *Client, _: []const u8, _: types.Position, _: []const u8) !Response {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn formatting(self: *Client, _: []const u8, _: types.FormattingOptions) !Response {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn diagnostics(self: *Client, _: []const u8) !Response {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn requestRaw(self: *Client, _: []const u8, _: ?[]const u8) !Response {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn notifyRaw(self: *Client, _: []const u8, _: ?[]const u8) !void {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn waitForNotification(self: *Client, _: []const u8, _: usize) !?[]u8 {
        _ = self;
        return error.FeatureDisabled;
    }
};

pub const resolveWorkspaceRoot = struct {
    pub fn f(_: std.mem.Allocator, _: anytype, _: ?[]const u8) ![]u8 {
        return error.FeatureDisabled;
    }
}.f;

pub const resolvePath = struct {
    pub fn f(_: std.mem.Allocator, _: anytype, _: ?[]const u8, _: []const u8) ![]u8 {
        return error.FeatureDisabled;
    }
}.f;

pub const pathToUri = struct {
    pub fn f(_: std.mem.Allocator, _: []const u8) ![]u8 {
        return error.FeatureDisabled;
    }
}.f;

pub const EnvConfig = struct {
    allocator: std.mem.Allocator,
    config: Config,
    owned: std.ArrayListUnmanaged([]const u8),

    pub fn deinit(self: *EnvConfig) void {
        self.owned.deinit(self.allocator);
    }
};

pub fn loadConfigFromEnv(allocator: std.mem.Allocator) !EnvConfig {
    return EnvConfig{
        .allocator = allocator,
        .config = Config.defaults(),
        .owned = .empty,
    };
}

pub fn isEnabled() bool {
    return false;
}

pub const Context = struct {
    pub fn isEnabled() bool {
        return false;
    }
};

test {
    std.testing.refAllDecls(@This());
}
