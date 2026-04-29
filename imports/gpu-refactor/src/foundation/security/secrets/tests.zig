const std = @import("std");
const secrets = @import("../secrets.zig");
const validation = @import("validation.zig");
const csprng = @import("../csprng.zig");

test "secrets manager initialization" {
    const allocator = std.testing.allocator;

    var key: [32]u8 = undefined;
    try csprng.fillRandom(&key);

    var manager = try secrets.SecretsManager.init(allocator, .{
        .provider = .memory,
        .master_key = key,
    });
    defer manager.deinit();

    try std.testing.expectEqual(@as(u64, 0), manager.getStats().secrets_loaded);
}

test "secure string wiping" {
    const allocator = std.testing.allocator;

    var secret = try secrets.SecureString.init(allocator, "sensitive-data");
    try std.testing.expectEqualStrings("sensitive-data", secret.slice());

    const ptr = secret.data.ptr;
    secret.deinit();
    _ = ptr;
}

test "pattern matching" {
    try std.testing.expect(validation.matchesPattern("API_KEY", "*"));
    try std.testing.expect(validation.matchesPattern("API_KEY", "API_*"));
    try std.testing.expect(validation.matchesPattern("API_KEY", "*_KEY"));
    try std.testing.expect(validation.matchesPattern("API_KEY", "API_KEY"));
    try std.testing.expect(!validation.matchesPattern("API_KEY", "SECRET_*"));
}

test "secret encryption round trip" {
    const allocator = std.testing.allocator;

    var key: [32]u8 = undefined;
    try csprng.fillRandom(&key);

    var manager = try secrets.SecretsManager.init(allocator, .{
        .provider = .memory,
        .master_key = key,
        .cache_secrets = true,
    });
    defer manager.deinit();

    try manager.set("test_secret", "my-secret-value");

    const retrieved = try manager.get("test_secret");
    defer allocator.free(retrieved);

    try std.testing.expectEqualStrings("my-secret-value", retrieved);
}

test {
    std.testing.refAllDecls(@This());
}
