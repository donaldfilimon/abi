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
// We re-export the same files here so that code using `abi.auth.jwt` compiles
// regardless of whether auth is enabled or disabled at build time. The feature
// gate only affects the Context lifecycle and high-level auth functions.

pub const api_keys = @import("../../services/shared/mod.zig").security.api_keys;
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
// Feature Context (stub)
// ============================================================================

const feature = stub_context.StubFeature(AuthConfig, AuthError);
pub const Context = feature.Context;

// ============================================================================
// Module-level API (stub — returns FeatureDisabled)
// ============================================================================

pub const init = feature.init;
pub const deinit = feature.deinit;
pub const isEnabled = feature.isEnabled;
pub const isInitialized = feature.isInitialized;

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
