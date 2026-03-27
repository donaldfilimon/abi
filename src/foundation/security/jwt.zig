//! JWT (JSON Web Token) authentication support.
//!
//! This module provides:
//! - JWT creation and signing (HS256, HS384, HS512, RS256)
//! - JWT verification and validation
//! - Claims management (standard and custom)
//! - Token refresh and rotation
//! - Blacklist/revocation support
//! - JWK (JSON Web Key) support

const std = @import("std");

const types = @import("jwt/types.zig");
const manager_mod = @import("jwt/manager.zig");
const standalone_mod = @import("jwt/standalone.zig");

pub const Algorithm = types.Algorithm;
pub const Claims = types.Claims;
pub const Token = types.Token;
pub const JwtConfig = types.JwtConfig;
pub const JwtError = types.JwtError;
pub const JwtHeader = types.JwtHeader;
pub const JwtPayload = types.JwtPayload;
pub const JwtToken = types.JwtToken;

pub const supportedAlgorithms = types.supportedAlgorithms;
pub const getRsaUnavailableReason = types.getRsaUnavailableReason;
pub const validateExpiry = types.validateExpiry;

pub const JwtManager = manager_mod.JwtManager;

pub const decode = standalone_mod.decode;
pub const verify = standalone_mod.verify;
pub const isExpired = standalone_mod.isExpired;
pub const base64UrlEncode = standalone_mod.base64UrlEncode;
pub const base64UrlDecode = standalone_mod.base64UrlDecode;
pub const extractBearerToken = standalone_mod.extractBearerToken;
pub const generateSecretKey = standalone_mod.generateSecretKey;

test "jwt create and verify" {
    const allocator = std.testing.allocator;

    var manager = JwtManager.init(allocator, "test-secret-key-32-bytes-long!!", .{});
    defer manager.deinit();

    const token_str = try manager.createToken(.{
        .sub = "user123",
        .exp = types.wallClockSeconds() + 3600,
    });
    defer allocator.free(token_str);

    var parts = std.mem.splitScalar(u8, token_str, '.');
    _ = parts.next();
    _ = parts.next();
    _ = parts.next();
    try std.testing.expect(parts.next() == null);

    var token = try manager.verifyToken(token_str);
    defer token.deinit(allocator);

    try std.testing.expect(token.verified);
    try std.testing.expectEqualStrings("user123", token.claims.sub.?);
}

test "jwt expiration" {
    const allocator = std.testing.allocator;

    var manager = JwtManager.init(allocator, "test-secret-key-32-bytes-long!!", .{
        .clock_skew = 0,
    });
    defer manager.deinit();

    const token_str = try manager.createToken(.{
        .sub = "user123",
        .exp = types.wallClockSeconds() - 100,
    });
    defer allocator.free(token_str);

    const result = manager.verifyToken(token_str);
    try std.testing.expectError(error.TokenExpired, result);
}

test "jwt blacklist" {
    const allocator = std.testing.allocator;

    var manager = JwtManager.init(allocator, "test-secret-key-32-bytes-long!!", .{
        .enable_blacklist = true,
    });
    defer manager.deinit();

    const token_str = try manager.createToken(.{
        .sub = "user123",
    });
    defer allocator.free(token_str);

    var token = try manager.verifyToken(token_str);

    try manager.revokeToken(&token);
    token.deinit(allocator);

    const result = manager.verifyToken(token_str);
    try std.testing.expectError(error.TokenRevoked, result);
}

test "extract bearer token" {
    const token = extractBearerToken("Bearer eyJhbGciOiJIUzI1NiJ9.test.sig");
    try std.testing.expect(token != null);
    try std.testing.expect(std.mem.startsWith(u8, token.?, "eyJ"));

    const no_bearer = extractBearerToken("Basic dXNlcjpwYXNz");
    try std.testing.expect(no_bearer == null);
}

test "supportedAlgorithms returns HMAC variants" {
    const algs = supportedAlgorithms();
    try std.testing.expectEqual(@as(usize, 3), algs.len);
    try std.testing.expectEqual(Algorithm.hs256, algs[0]);
    try std.testing.expectEqual(Algorithm.hs384, algs[1]);
    try std.testing.expectEqual(Algorithm.hs512, algs[2]);
}

test "getRsaUnavailableReason is non-empty" {
    const reason = getRsaUnavailableReason();
    try std.testing.expect(reason.len > 0);
}

test "validateExpiry with Claims struct" {
    const future: i64 = 9999999999;
    const past: i64 = 1;
    const now: i64 = 1000000;

    const valid_claims = Claims{ .exp = future };
    try std.testing.expect(validateExpiry(valid_claims, now));

    const expired_claims = Claims{ .exp = past };
    try std.testing.expect(!validateExpiry(expired_claims, now));

    const no_exp = Claims{};
    try std.testing.expect(validateExpiry(no_exp, now));
}

test "standalone decode splits and parses JWT" {
    const allocator = std.testing.allocator;

    const token_str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." ++
        "eyJzdWIiOiIxMjM0NTY3ODkwIiwiaXNzIjoidGVzdCIsImV4cCI6OTk5OTk5OTk5OSwiaWF0IjoxNTE2MjM5MDIyfQ." ++
        "JD4jJSonpR3FCs-xnBECNx3MLLqVDMBJbvlHJGR9z2Y";

    var token = try decode(allocator, token_str);
    defer token.deinit(allocator);

    try std.testing.expectEqual(Algorithm.hs256, token.header.alg);
    try std.testing.expectEqualStrings("1234567890", token.claims.sub.?);
    try std.testing.expectEqualStrings("test", token.claims.iss.?);
    try std.testing.expectEqual(@as(i64, 9999999999), token.claims.exp.?);
    try std.testing.expectEqual(@as(i64, 1516239022), token.claims.iat.?);
    try std.testing.expect(!token.verified);
}

test "standalone decode rejects malformed tokens" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(error.InvalidToken, decode(allocator, "aaa.bbb"));
    try std.testing.expectError(error.InvalidToken, decode(allocator, "aaa.bbb.ccc.ddd"));
}

test "standalone verify with known HMAC-SHA256 test vector" {
    const allocator = std.testing.allocator;

    const secret = "test-secret-key-for-standalone!!";
    var manager = JwtManager.init(allocator, secret, .{
        .clock_skew = 0,
    });
    defer manager.deinit();

    const token_str = try manager.createToken(.{
        .sub = "alice",
        .exp = types.wallClockSeconds() + 3600,
    });
    defer allocator.free(token_str);

    var token = try verify(allocator, token_str, secret);
    defer token.deinit(allocator);

    try std.testing.expect(token.verified);
    try std.testing.expectEqualStrings("alice", token.claims.sub.?);
}

test "standalone verify rejects wrong secret" {
    const allocator = std.testing.allocator;

    const secret = "correct-secret-key-32-bytes!!!!!";
    var manager = JwtManager.init(allocator, secret, .{});
    defer manager.deinit();

    const token_str = try manager.createToken(.{
        .sub = "bob",
        .exp = types.wallClockSeconds() + 3600,
    });
    defer allocator.free(token_str);

    const result = verify(allocator, token_str, "wrong-secret-key-32-bytes-long!!");
    try std.testing.expectError(error.InvalidSignature, result);
}

test "isExpired checks exp claim" {
    const valid = Token{
        .raw = "",
        .header = .{ .alg = .hs256 },
        .claims = .{ .exp = 9999999999 },
        .signature = "",
        .verified = true,
    };
    try std.testing.expect(!isExpired(valid));

    const expired = Token{
        .raw = "",
        .header = .{ .alg = .hs256 },
        .claims = .{ .exp = 1 },
        .signature = "",
        .verified = true,
    };
    try std.testing.expect(isExpired(expired));

    const no_exp = Token{
        .raw = "",
        .header = .{ .alg = .hs256 },
        .claims = .{},
        .signature = "",
        .verified = true,
    };
    try std.testing.expect(!isExpired(no_exp));
}

test "base64UrlEncode and base64UrlDecode roundtrip" {
    const allocator = std.testing.allocator;

    const original = "Hello, JWT world! Special chars: +/=";
    const encoded = try base64UrlEncode(allocator, original);
    defer allocator.free(encoded);

    for (encoded) |c| {
        try std.testing.expect(c != '+');
        try std.testing.expect(c != '/');
        try std.testing.expect(c != '=');
    }

    const decoded = try base64UrlDecode(allocator, encoded);
    defer allocator.free(decoded);

    try std.testing.expectEqualStrings(original, decoded);
}

test "base64UrlDecode rejects invalid input" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidBase64, base64UrlDecode(allocator, "!!!invalid!!!"));
}

test "standalone decode with custom claims" {
    const allocator = std.testing.allocator;

    const secret = "custom-claims-test-secret-key!!!";
    var manager = JwtManager.init(allocator, secret, .{});
    defer manager.deinit();

    var custom = std.StringArrayHashMapUnmanaged([]const u8).empty;
    try custom.put(allocator, try allocator.dupe(u8, "role"), try allocator.dupe(u8, "admin"));
    defer {
        var it = custom.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        custom.deinit(allocator);
    }

    const token_str = try manager.createToken(.{
        .sub = "charlie",
        .exp = types.wallClockSeconds() + 3600,
        .custom = custom,
    });
    // Free the custom map entries after createToken has serialised them
    {
        var it = custom.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        custom.deinit(allocator);
    }
    defer allocator.free(token_str);

    var token = try decode(allocator, token_str);
    defer token.deinit(allocator);

    try std.testing.expectEqualStrings("charlie", token.claims.sub.?);
    try std.testing.expect(token.claims.custom != null);
    const role = token.claims.custom.?.get("role");
    try std.testing.expect(role != null);
    try std.testing.expectEqualStrings("admin", role.?);
}

test {
    std.testing.refAllDecls(@This());
}
