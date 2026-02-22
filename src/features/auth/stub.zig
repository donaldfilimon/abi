//! Auth Stub Module
//!
//! API-compatible no-op implementations when auth is disabled.
//! Sub-module re-exports match mod.zig surface for parity.

const std = @import("std");
const core_config = @import("../../core/config/auth.zig");
const stub_context = @import("../../core/stub_context.zig");

pub const AuthConfig = core_config.AuthConfig;

// ============================================================================
// Security Sub-modules (re-exported — same as mod.zig)
// ============================================================================
// Note: Security sub-modules are always compiled (they live in services/shared).
// We re-export the same files here so that code using `abi.auth.jwt` compiles
// regardless of whether auth is enabled or disabled at build time. The feature
// gate only affects the Context lifecycle and high-level auth functions.

pub const api_keys = @import("../../services/shared/security/api_keys.zig");
pub const audit = @import("../../services/shared/security/audit.zig");
pub const certificates = @import("../../services/shared/security/certificates.zig");
pub const cors = @import("../../services/shared/security/cors.zig");
pub const encryption = @import("../../services/shared/security/encryption.zig");
pub const headers = @import("../../services/shared/security/headers.zig");
pub const ip_filter = @import("../../services/shared/security/ip_filter.zig");
pub const jwt = @import("../../services/shared/security/jwt.zig");
pub const mtls = @import("../../services/shared/security/mtls.zig");
pub const password = @import("../../services/shared/security/password.zig");
pub const rate_limit = @import("../../services/shared/security/rate_limit.zig");
pub const rbac = @import("../../services/shared/security/rbac.zig");
pub const secrets = @import("../../services/shared/security/secrets.zig");
pub const session = @import("../../services/shared/security/session.zig");
pub const tls = @import("../../services/shared/security/tls.zig");
pub const validation = @import("../../services/shared/security/validation.zig");

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
