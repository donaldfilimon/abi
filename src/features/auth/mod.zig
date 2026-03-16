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
const core_config = @import("../../core/config/platform.zig");

pub const AuthConfig = core_config.AuthConfig;

// ============================================================================
// Security Sub-modules (re-exported from services/shared/security/)
// ============================================================================

const shared = @import("../../services/shared/mod.zig");
pub const api_keys = shared.security.api_keys;
pub const audit = @import("../../services/shared/mod.zig").security.audit;
pub const certificates = @import("../../services/shared/mod.zig").security.certificates;
pub const cors = @import("../../services/shared/mod.zig").security.cors;
pub const encryption = @import("../../services/shared/mod.zig").security.encryption;
pub const headers = @import("../../services/shared/mod.zig").security.headers;
pub const ip_filter = @import("../../services/shared/mod.zig").security.ip_filter;
pub const jwt = @import("../../services/shared/mod.zig").security.jwt;
pub const mtls = @import("../../services/shared/mod.zig").security.mtls;
pub const password = @import("../../services/shared/mod.zig").security.password;
pub const rate_limit = @import("../../services/shared/mod.zig").security.rate_limit;
pub const rbac = @import("../../services/shared/mod.zig").security.rbac;
pub const secrets = @import("../../services/shared/mod.zig").security.secrets;
pub const session = @import("../../services/shared/mod.zig").security.session;
pub const tls = @import("../../services/shared/mod.zig").security.tls;
pub const validation = @import("../../services/shared/mod.zig").security.validation;

// ============================================================================
// Auth-level Types (from types.zig)
// ============================================================================

const auth_types = @import("types.zig");
pub const AuthError = auth_types.AuthError;
pub const Error = AuthError;
pub const Token = auth_types.Token;
pub const Session = auth_types.Session;
pub const Permission = auth_types.Permission;

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
//
// These convenience functions delegate to the shared security sub-modules
// (jwt, session, rbac). They create ephemeral manager instances per call,
// which is fine for low-frequency operations. For high-throughput use,
// callers should instantiate the sub-module managers directly.
// ============================================================================

/// Default secret used when AuthConfig.jwt_secret is null.
/// Only suitable for development/testing — production code MUST provide
/// a real secret via AuthConfig.
const default_jwt_secret = "abi-auth-default-dev-secret-key!";

/// Module-level state.  `init()` stores the configured secret so that
/// `createToken()` / `verifyToken()` use it instead of the hardcoded default.
var active_jwt_secret: []const u8 = default_jwt_secret;
var initialized: bool = false;

/// Initialise the auth module with a caller-provided config.
/// If `config.jwt_secret` is set, it will be used for all subsequent token
/// operations.  Otherwise the default dev secret is used and a warning is
/// printed to stderr.
pub fn init(_: std.mem.Allocator, config: AuthConfig) AuthError!void {
    if (@hasField(AuthConfig, "jwt_secret")) {
        if (@field(config, "jwt_secret")) |secret| {
            if (secret.len > 0) {
                active_jwt_secret = secret;
                initialized = true;
                return;
            }
        }
    }
    // Fallback to default — warn loudly.
    std.log.warn("auth: using default dev JWT secret — NOT suitable for production", .{});
    active_jwt_secret = default_jwt_secret;
    initialized = true;
}

pub fn deinit() void {
    active_jwt_secret = default_jwt_secret;
    initialized = false;
}

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return initialized;
}

/// Create a signed JWT token for the given user_id.
/// Delegates to `jwt.JwtManager.createToken` from `services/shared/security/jwt.zig`.
pub fn createToken(
    allocator: std.mem.Allocator,
    user_id: []const u8,
) AuthError!Token {
    var manager = jwt.JwtManager.init(allocator, active_jwt_secret, .{
        .required_claims = &.{},
    });
    defer manager.deinit();

    const raw = manager.createToken(.{
        .sub = user_id,
    }) catch return error.OutOfMemory;

    return Token{
        .raw = raw,
        .claims = .{
            .sub = user_id,
        },
    };
}

/// Verify a JWT token string and return parsed token info.
/// Delegates to `jwt.JwtManager.verifyToken` from `services/shared/security/jwt.zig`.
/// Uses the default dev secret; production callers should use `jwt.JwtManager`
/// directly with their own secret.
///
/// Note: The returned Token's `.claims.sub` is heap-allocated via page_allocator
/// when non-empty and should be freed by the caller if needed. The `.raw` field
/// points to the input `token_str` (caller-owned, not duped).
pub fn verifyToken(token_str: []const u8) AuthError!Token {
    // Use the page allocator since we have no allocator parameter.
    // This is acceptable for a convenience wrapper; high-throughput code
    // should use jwt.JwtManager directly with a proper allocator.
    const allocator = std.heap.page_allocator;

    var manager = jwt.JwtManager.init(allocator, active_jwt_secret, .{
        .required_claims = &.{},
    });
    defer manager.deinit();

    var jwt_token = manager.verifyToken(token_str) catch |err| {
        return switch (err) {
            error.TokenExpired => error.TokenExpired,
            error.InvalidSignature => error.InvalidCredentials,
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidCredentials,
        };
    };

    // Extract values before deinit frees the jwt_token's strings.
    const sub_val: []const u8 = if (jwt_token.claims.sub) |s|
        allocator.dupe(u8, s) catch {
            jwt_token.deinit(allocator);
            return error.OutOfMemory;
        }
    else
        "";
    const exp_val: u64 = if (jwt_token.claims.exp) |e| @intCast(@as(u64, @bitCast(e))) else 0;
    const iat_val: u64 = if (jwt_token.claims.iat) |i| @intCast(@as(u64, @bitCast(i))) else 0;

    jwt_token.deinit(allocator);

    return Token{
        .raw = token_str,
        .claims = .{
            .sub = sub_val,
            .exp = exp_val,
            .iat = iat_val,
        },
    };
}

/// Create a new session for the given user_id.
/// Delegates to `session.SessionManager.create` from
/// `services/shared/security/session.zig`.
///
/// The returned `Session.id` and `Session.user_id` are heap-allocated via the
/// provided allocator; callers should free them when done (or use an arena).
pub fn createSession(
    allocator: std.mem.Allocator,
    user_id: []const u8,
) AuthError!Session {
    // Use page_allocator for the ephemeral SessionManager's internal
    // allocations so they don't interfere with the caller's allocator.
    const internal = std.heap.page_allocator;
    var manager = session.SessionManager.init(internal, .{});

    const sess = manager.create(.{
        .user_id = user_id,
    }) catch {
        manager.deinit();
        return error.OutOfMemory;
    };

    // Copy all values out before deinit frees the session.
    const created: i64 = sess.created_at;
    const expires: i64 = sess.expires_at;
    const id_slice: []const u8 = sess.id;
    const uid_slice: []const u8 = sess.user_id orelse "";

    // Dupe strings with caller's allocator so they outlive the manager.
    const id_copy = allocator.dupe(u8, id_slice) catch {
        manager.deinit();
        return error.OutOfMemory;
    };
    const uid_copy: []const u8 = if (uid_slice.len > 0)
        allocator.dupe(u8, uid_slice) catch {
            allocator.free(id_copy);
            manager.deinit();
            return error.OutOfMemory;
        }
    else
        "";

    manager.deinit();

    return Session{
        .id = id_copy,
        .user_id = uid_copy,
        .created_at = @intCast(@as(u64, @bitCast(created))),
        .expires_at = @intCast(@as(u64, @bitCast(expires))),
    };
}

/// Check if a user has a given permission.
/// Delegates to `rbac.RbacManager.hasPermission` from
/// `services/shared/security/rbac.zig`.
///
/// Maps the auth-level `Permission` enum to the RBAC module's `Permission`.
/// Creates an ephemeral RbacManager with default roles. Without explicit
/// role assignment the user will have no permissions, so this returns false
/// by default — callers needing real RBAC should use `rbac.RbacManager`
/// directly and assign roles.
pub fn checkPermission(user_id: []const u8, permission: Permission) AuthError!bool {
    const allocator = std.heap.page_allocator;

    var manager = rbac.RbacManager.init(allocator, .{}) catch return error.OutOfMemory;
    defer manager.deinit();

    // Map auth Permission to rbac Permission
    const rbac_perm: rbac.Permission = switch (permission) {
        .read => .read,
        .write => .write,
        .admin => .admin,
    };

    return manager.hasPermission(user_id, rbac_perm) catch return error.OutOfMemory;
}

// Test discovery — standalone test file avoids pulling in security sub-modules
// that have pre-existing Zig 0.16 compile issues
test {
    _ = @import("auth_test.zig");
}

test {
    std.testing.refAllDecls(@This());
}
