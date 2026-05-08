//! Focused auth unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const auth = @import("features/auth/mod.zig");

test {
    std.testing.refAllDecls(auth);
}

// ── Module lifecycle tests ─────────────────────────────────────────────

test "auth isEnabled returns true" {
    try std.testing.expect(auth.isEnabled());
}

test "auth init deinit cycle resets initialized state" {
    const allocator = std.testing.allocator;
    try auth.init(allocator, auth.AuthConfig.defaults());
    try std.testing.expect(auth.isInitialized());
    auth.deinit();
    try std.testing.expect(!auth.isInitialized());
}

test "auth double init is idempotent" {
    const allocator = std.testing.allocator;
    try auth.init(allocator, auth.AuthConfig.defaults());
    defer auth.deinit();
    // Second init should be a no-op (already initialized)
    try auth.init(allocator, auth.AuthConfig.defaults());
    try std.testing.expect(auth.isInitialized());
}

// ── Token tests ────────────────────────────────────────────────────────

test "auth createToken returns valid JWT structure" {
    const allocator = std.testing.allocator;
    const token = try auth.createToken(allocator, "test_user");
    defer allocator.free(token.raw);

    // JWT must have exactly 3 dot-separated segments
    var count: u32 = 0;
    var it = std.mem.splitScalar(u8, token.raw, '.');
    while (it.next()) |seg| {
        try std.testing.expect(seg.len > 0);
        count += 1;
    }
    try std.testing.expectEqual(@as(u32, 3), count);
    try std.testing.expectEqualStrings("test_user", token.claims.sub);
}

test "auth verifyToken round-trip preserves subject claim" {
    const allocator = std.testing.allocator;
    const token = try auth.createToken(allocator, "verify_user");
    defer allocator.free(token.raw);

    const verified = try auth.verifyToken(allocator, token.raw);
    defer if (verified.claims.sub.len > 0) allocator.free(verified.claims.sub);

    try std.testing.expectEqualStrings("verify_user", verified.claims.sub);
    try std.testing.expect(verified.claims.exp > 0);
}

test "auth verifyToken rejects garbage input" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidCredentials, auth.verifyToken(allocator, ""));
    try std.testing.expectError(error.InvalidCredentials, auth.verifyToken(allocator, "not.a.jwt"));
    try std.testing.expectError(error.InvalidCredentials, auth.verifyToken(allocator, "x"));
}

// ── Session tests ──────────────────────────────────────────────────────

test "auth createSession returns valid session" {
    const allocator = std.testing.allocator;
    const sess = try auth.createSession(allocator, "session_user");
    defer allocator.free(sess.id);
    defer if (sess.user_id.len > 0) allocator.free(sess.user_id);

    try std.testing.expect(sess.id.len > 0);
    try std.testing.expectEqualStrings("session_user", sess.user_id);
    try std.testing.expect(sess.expires_at >= sess.created_at);
    try std.testing.expect(sess.expires_at > 0);
}

test "auth createSession produces unique IDs" {
    const allocator = std.testing.allocator;
    const s1 = try auth.createSession(allocator, "user_a");
    defer allocator.free(s1.id);
    defer if (s1.user_id.len > 0) allocator.free(s1.user_id);

    const s2 = try auth.createSession(allocator, "user_b");
    defer allocator.free(s2.id);
    defer if (s2.user_id.len > 0) allocator.free(s2.user_id);

    try std.testing.expect(!std.mem.eql(u8, s1.id, s2.id));
}

// ── Permission tests ───────────────────────────────────────────────────

test "auth checkPermission returns false without role assignment" {
    // Ephemeral RbacManager with no roles always returns false
    try std.testing.expect(!(try auth.checkPermission("nobody", .read)));
    try std.testing.expect(!(try auth.checkPermission("nobody", .write)));
    try std.testing.expect(!(try auth.checkPermission("nobody", .admin)));
}

// ── Type tests ─────────────────────────────────────────────────────────

test "auth Token default values" {
    const token = auth.Token{};
    try std.testing.expectEqualStrings("", token.raw);
    try std.testing.expectEqualStrings("", token.claims.sub);
    try std.testing.expectEqual(@as(u64, 0), token.claims.exp);
    try std.testing.expectEqual(@as(u64, 0), token.claims.iat);
}

test "auth Session default values" {
    const sess = auth.Session{};
    try std.testing.expectEqualStrings("", sess.id);
    try std.testing.expectEqualStrings("", sess.user_id);
    try std.testing.expectEqual(@as(u64, 0), sess.created_at);
    try std.testing.expectEqual(@as(u64, 0), sess.expires_at);
}

test "auth Permission enum variants" {
    const perms = [_]auth.Permission{ .read, .write, .admin };
    try std.testing.expectEqual(@as(usize, 3), perms.len);
}
