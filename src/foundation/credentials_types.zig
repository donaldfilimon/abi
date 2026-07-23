//! Credential struct and in-memory secret hygiene helpers.
//! Package-private building blocks for credentials.zig (hub).
const std = @import("std");

pub const Credentials = struct {
    openai_api_key: ?[]const u8 = null,
    anthropic_api_key: ?[]const u8 = null,
    discord_token: ?[]const u8 = null,
    grok_api_key: ?[]const u8 = null,
    twilio_account_sid: ?[]const u8 = null,
    twilio_auth_token: ?[]const u8 = null,

    pub fn deinit(self: *Credentials, allocator: std.mem.Allocator) void {
        wipeAndFree(allocator, &self.openai_api_key);
        wipeAndFree(allocator, &self.anthropic_api_key);
        wipeAndFree(allocator, &self.discord_token);
        wipeAndFree(allocator, &self.grok_api_key);
        wipeAndFree(allocator, &self.twilio_account_sid);
        wipeAndFree(allocator, &self.twilio_auth_token);
    }
};

/// Securely zero an owned secret slice then free it. Heap hygiene on all
/// platforms; Windows ACL owner-only is applied at write time separately.
/// When the keychain backend is active (`ABI_CREDENTIALS_BACKEND=keychain`,
/// macOS only), keychain-stored secrets are cleared via `keychainDelete` in
/// `clearKeychainCredentials`/`saveCredentialsToKeychain`, not by this
/// in-process heap wipe — this only wipes the caller's in-memory copy.
pub fn wipeAndFree(allocator: std.mem.Allocator, field: *?[]const u8) void {
    if (field.*) |k| {
        const mutable: []u8 = @constCast(k);
        std.crypto.secureZero(u8, mutable);
        allocator.free(mutable);
        field.* = null;
    }
}

/// Best-effort wipe of a borrowed mutable buffer (stdin, JSON scratch).
pub fn secureWipe(buf: []u8) void {
    if (buf.len == 0) return;
    std.crypto.secureZero(u8, buf);
}

pub fn replaceOwnedString(allocator: std.mem.Allocator, field: *?[]const u8, value: []const u8) !void {
    const replacement = try allocator.dupe(u8, value);
    wipeAndFree(allocator, field);
    field.* = replacement;
}

test {
    std.testing.refAllDecls(@This());
}

test "replaceOwnedString preserves old value on allocation failure" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const allocator = std.testing.allocator;
    var field: ?[]const u8 = try allocator.dupe(u8, "old-value");
    defer wipeAndFree(allocator, &field);

    try std.testing.expectError(error.OutOfMemory, replaceOwnedString(failing.allocator(), &field, "new-value"));
    try std.testing.expectEqualStrings("old-value", field orelse return error.MissingOldValue);
}

test "Credentials deinit clears secret field; secureZero clears in place" {
    const allocator = std.testing.allocator;
    var creds = Credentials{};
    try replaceOwnedString(allocator, &creds.openai_api_key, "sk-secret-test-key");
    creds.deinit(allocator);
    try std.testing.expect(creds.openai_api_key == null);

    const buf = try allocator.dupe(u8, "still-secret");
    defer allocator.free(buf);
    std.crypto.secureZero(u8, buf);
    for (buf) |b| try std.testing.expectEqual(@as(u8, 0), b);
}
