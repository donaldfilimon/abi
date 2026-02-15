//! Auth module standalone tests.
//! Separated from mod.zig to avoid pulling in security sub-modules that have
//! pre-existing Zig 0.16 compile issues (rbac, secrets, session, validation).

const std = @import("std");
const auth = @import("mod.zig");

test "auth context init and deinit" {
    const allocator = std.testing.allocator;
    const ctx = try auth.Context.init(allocator, auth.AuthConfig.defaults());
    defer ctx.deinit();
    try std.testing.expect(@intFromPtr(ctx) != 0);
}

test "auth module enabled and initialized" {
    try std.testing.expect(auth.isEnabled());
    try std.testing.expect(auth.isInitialized());
}

test "auth token creation" {
    const token = try auth.createToken(std.testing.allocator, "user123");
    try std.testing.expectEqualStrings("", token.raw);
}

test "auth permission check" {
    const result = try auth.checkPermission("user123", .read);
    try std.testing.expect(result);
}

test "auth session creation" {
    const session_inst = try auth.createSession(std.testing.allocator, "user456");
    try std.testing.expectEqualStrings("", session_inst.id);
}

test "auth error type variants" {
    const err: auth.AuthError = error.InvalidCredentials;
    try std.testing.expect(err == error.InvalidCredentials);
    const err2: auth.AuthError = error.TokenExpired;
    try std.testing.expect(err2 == error.TokenExpired);
}

test "auth type definitions" {
    const token = auth.Token{};
    try std.testing.expectEqualStrings("", token.raw);
    try std.testing.expectEqual(@as(u64, 0), token.claims.exp);

    const sess = auth.Session{};
    try std.testing.expectEqualStrings("", sess.id);
    try std.testing.expectEqual(@as(u64, 0), sess.expires_at);
}

test "auth permission enum" {
    const perm: auth.Permission = .read;
    try std.testing.expect(perm == .read);
    const perm2: auth.Permission = .write;
    try std.testing.expect(perm2 != .admin);
}
