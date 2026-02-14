//! Auth Module
//!
//! Authentication and security infrastructure for the ABI framework.
//! Re-exports the full security infrastructure from `services/shared/security/`.
//!
//! When the `auth` feature is enabled, all security sub-modules are available:
//! - `abi.auth.jwt` — JSON Web Tokens (HMAC-SHA256/384/512)
//! - `abi.auth.api_keys` — API key management with secure hashing
//! - `abi.auth.rbac` — Role-based access control
//! - `abi.auth.session` — Session management
//! - `abi.auth.password` — Secure password hashing (Argon2id, PBKDF2, scrypt)
//! - `abi.auth.cors` — Cross-Origin Resource Sharing
//! - `abi.auth.rate_limit` — Token bucket, sliding window, leaky bucket
//! - `abi.auth.encryption` — AES-256-GCM, ChaCha20-Poly1305
//! - `abi.auth.tls` / `abi.auth.mtls` — Transport security
//! - `abi.auth.certificates` — X.509 certificate management
//! - `abi.auth.secrets` — Encrypted credential storage
//! - `abi.auth.audit` — Tamper-evident security event logging
//! - `abi.auth.validation` — Input sanitization
//! - `abi.auth.ip_filter` — IP allow/deny lists
//! - `abi.auth.headers` — Security headers middleware

const std = @import("std");
const core_config = @import("../../core/config/auth.zig");

pub const AuthConfig = core_config.AuthConfig;

// ============================================================================
// Security Sub-modules (re-exported from services/shared/security/)
// ============================================================================

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
// Feature Context
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: AuthConfig,

    pub fn init(allocator: std.mem.Allocator, config: AuthConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{ .allocator = allocator, .config = config };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Module-level API
// ============================================================================

pub fn init(_: std.mem.Allocator, _: AuthConfig) AuthError!void {}
pub fn deinit() void {}
pub fn isEnabled() bool {
    return true;
}
pub fn isInitialized() bool {
    return true;
}

pub fn createToken(
    allocator: std.mem.Allocator,
    _: []const u8,
) AuthError!Token {
    _ = allocator;
    return .{};
}

pub fn verifyToken(_: []const u8) AuthError!Token {
    return .{};
}

pub fn createSession(
    allocator: std.mem.Allocator,
    _: []const u8,
) AuthError!Session {
    _ = allocator;
    return .{};
}

pub fn checkPermission(_: []const u8, _: Permission) AuthError!bool {
    return true;
}
