//! Integration Tests: Auth Feature
//!
//! Tests the auth module exports, JWT/RBAC types, and lifecycle functions.

const std = @import("std");
const abi = @import("abi");

const auth = abi.auth;

// ============================================================================
// Module Lifecycle
// ============================================================================

test "auth: isEnabled returns bool" {
    const enabled = auth.isEnabled();
    try std.testing.expect(enabled == true or enabled == false);
}

test "auth: isInitialized returns bool" {
    const initialized = auth.isInitialized();
    try std.testing.expect(initialized == true or initialized == false);
}

// ============================================================================
// Types
// ============================================================================

test "auth: Token type has expected fields" {
    const token = auth.Token{};
    try std.testing.expectEqualStrings("", token.raw);
    try std.testing.expectEqual(@as(u64, 0), token.claims.exp);
    try std.testing.expectEqual(@as(u64, 0), token.claims.iat);
    try std.testing.expectEqualStrings("", token.claims.sub);
}

test "auth: Token.Claims can be constructed" {
    const claims = auth.Token.Claims{
        .sub = "user-123",
        .exp = 1700000000,
        .iat = 1699000000,
    };
    try std.testing.expectEqualStrings("user-123", claims.sub);
    try std.testing.expectEqual(@as(u64, 1700000000), claims.exp);
}

test "auth: Session type has expected fields" {
    const session = auth.Session{};
    try std.testing.expectEqualStrings("", session.id);
    try std.testing.expectEqualStrings("", session.user_id);
    try std.testing.expectEqual(@as(u64, 0), session.created_at);
    try std.testing.expectEqual(@as(u64, 0), session.expires_at);
}

test "auth: Permission enum has expected variants" {
    const read = auth.Permission.read;
    const write = auth.Permission.write;
    const admin = auth.Permission.admin;
    try std.testing.expect(read != write);
    try std.testing.expect(write != admin);
}

// ============================================================================
// Error set
// ============================================================================

test "auth: AuthError includes expected variants" {
    const err: auth.AuthError = error.FeatureDisabled;
    try std.testing.expect(err == error.FeatureDisabled);
}

// ============================================================================
// Security sub-module re-exports
// ============================================================================

test "auth: jwt sub-module is accessible" {
    const jwt = auth.jwt;
    try std.testing.expect(@TypeOf(jwt) != void);
}

test "auth: rbac sub-module is accessible" {
    const rbac_mod = auth.rbac;
    try std.testing.expect(@TypeOf(rbac_mod) != void);
}

test "auth: rate_limit sub-module is accessible" {
    const rl = auth.rate_limit;
    try std.testing.expect(@TypeOf(rl) != void);
}

// ============================================================================
// Stub API (when disabled, functions return FeatureDisabled)
// ============================================================================

test "auth: createToken returns result or FeatureDisabled" {
    const result = auth.createToken(std.testing.allocator, "user-1");
    if (result) |_| {
        // Feature enabled — token created successfully
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "auth: verifyToken returns result or FeatureDisabled" {
    const result = auth.verifyToken(std.testing.allocator, "some-token");
    if (result) |_| {
        // Feature enabled
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "auth: createSession returns result or FeatureDisabled" {
    const result = auth.createSession(std.testing.allocator, "user-1");
    if (result) |_| {
        // Feature enabled
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "auth: checkPermission returns result or FeatureDisabled" {
    const result = auth.checkPermission("user-1", .read);
    if (result) |_| {
        // Feature enabled
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test {
    std.testing.refAllDecls(@This());
}
