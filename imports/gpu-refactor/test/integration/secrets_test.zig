//! Integration Tests: Secrets module public surface

const std = @import("std");
const abi = @import("abi");

const security = abi.foundation.security;

test "secrets: public types are re-exported" {
    _ = security.SecretsManager;
    _ = security.SecretsConfig;
    _ = security.SecretValue;
    _ = security.SecretMetadata;
    _ = security.SecretType;
    _ = security.SecureString;
    _ = security.SecretsError;
}

test "secrets: SecureString round trip" {
    var secret = try security.SecureString.init(std.testing.allocator, "top-secret");
    defer secret.deinit();

    try std.testing.expectEqualStrings("top-secret", secret.slice());
}

test "secrets: memory provider round trip via public surface" {
    const key: [32]u8 = [_]u8{7} ** 32;

    var manager = try security.SecretsManager.init(std.testing.allocator, .{
        .provider = .memory,
        .master_key = key,
    });
    defer manager.deinit();

    try manager.set("integration_secret", "hello");

    const value = try manager.get("integration_secret");
    defer std.testing.allocator.free(value);

    try std.testing.expectEqualStrings("hello", value);
}

test {
    std.testing.refAllDecls(@This());
}
