//! Auth Stub Module
//!
//! API-compatible no-op implementations when auth is disabled.
//! Sub-module re-exports match mod.zig surface for parity.

const std = @import("std");
const core_config = @import("../../core/config/platform.zig");
const stub_context = @import("../../core/stub_context.zig");

pub const AuthConfig = core_config.AuthConfig;

// ============================================================================
// Security Sub-modules (re-exported — same as mod.zig)
// ============================================================================
// Note: Security sub-modules are always compiled (they live in services/shared).
// We re-export the same files here so that code using `abi.features.auth.jwt` compiles
// regardless of whether auth is enabled or disabled at build time. The feature
// gate only affects the Context lifecycle and high-level auth functions.

pub const api_keys = @import("shared_services").security.api_keys;
pub const audit = @import("shared_services").security.audit;
pub const certificates = @import("shared_services").security.certificates;
pub const cors = @import("shared_services").security.cors;
pub const encryption = @import("shared_services").security.encryption;
pub const headers = @import("shared_services").security.headers;
pub const ip_filter = @import("shared_services").security.ip_filter;
pub const jwt = @import("shared_services").security.jwt;
pub const mtls = @import("shared_services").security.mtls;
pub const password = @import("shared_services").security.password;
pub const rate_limit = @import("shared_services").security.rate_limit;
pub const rbac = @import("shared_services").security.rbac;
pub const secrets = @import("shared_services").security.secrets;
pub const session = @import("shared_services").security.session;
pub const tls = @import("shared_services").security.tls;
pub const validation = @import("shared_services").security.validation;

// ============================================================================
// Auth-level Types
// ============================================================================

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

// ============================================================================
// Feature Context (stub)
// ============================================================================

pub const Context = stub_context.StubContext(AuthConfig);

// ============================================================================
// Module-level API (stub — returns FeatureDisabled)
// ============================================================================

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

pub fn createToken(
    _: std.mem.Allocator,
    _: []const u8,
) AuthError!Token {
    return error.FeatureDisabled;
}

pub fn verifyToken(_: []const u8) AuthError!Token {
    return error.FeatureDisabled;
}

pub fn createSession(
    _: std.mem.Allocator,
    _: []const u8,
) AuthError!Session {
    return error.FeatureDisabled;
}

pub fn checkPermission(_: []const u8, _: Permission) AuthError!bool {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
