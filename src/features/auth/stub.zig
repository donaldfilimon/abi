//! Auth Stub Module
//!
//! API-compatible no-op implementations when auth is disabled.

const std = @import("std");
const core_config = @import("../../core/config/auth.zig");

pub const AuthConfig = core_config.AuthConfig;

pub const AuthError = error{
    FeatureDisabled,
    InvalidCredentials,
    TokenExpired,
    Unauthorized,
    OutOfMemory,
};

pub const Token = struct {
    raw: []const u8 = "",
    claims: Claims = .{},

    pub const Claims = struct {
        sub: []const u8 = "",
        exp: u64 = 0,
        iat: u64 = 0,
    };
};

pub const Session = struct {
    id: []const u8 = "",
    user_id: []const u8 = "",
    created_at: u64 = 0,
    expires_at: u64 = 0,
};

pub const Permission = enum { read, write, admin };

pub const Context = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: AuthConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

pub fn init(_: std.mem.Allocator, _: AuthConfig) AuthError!void {
    return error.FeatureDisabled;
}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}

pub fn createToken(_: std.mem.Allocator, _: []const u8) AuthError!Token {
    return error.FeatureDisabled;
}

pub fn verifyToken(_: []const u8) AuthError!Token {
    return error.FeatureDisabled;
}

pub fn createSession(_: std.mem.Allocator, _: []const u8) AuthError!Session {
    return error.FeatureDisabled;
}

pub fn checkPermission(_: []const u8, _: Permission) AuthError!bool {
    return error.FeatureDisabled;
}
