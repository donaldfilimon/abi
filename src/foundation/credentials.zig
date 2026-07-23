//! Credential storage hub: file backend (default) + opt-in macOS keychain.
//!
//! Leaf modules:
//! - `credentials_types.zig` — struct + in-memory secret hygiene
//! - `credentials_file.zig` — path resolution + JSON file I/O + ACL hardening
//! - `credentials_keychain.zig` — `ABI_CREDENTIALS_BACKEND=keychain` glue
//!
//! Public surface is re-exported here so `foundation.credentials` and direct
//! `@import(".../credentials.zig")` callers stay unchanged.
const std = @import("std");
const builtin = @import("builtin");
const env = @import("env.zig");
const utils = @import("utils.zig");
const types = @import("credentials_types.zig");
const file = @import("credentials_file.zig");
const keychain_backend = @import("credentials_keychain.zig");
const temp_path = @import("temp_path.zig");

pub const Credentials = types.Credentials;
pub const secureWipe = types.secureWipe;
pub const replaceOwnedString = types.replaceOwnedString;

pub const getCredentialsPath = file.getCredentialsPath;

pub const credentialsBackendIsKeychain = keychain_backend.credentialsBackendIsKeychain;

pub fn loadCredentials(allocator: std.mem.Allocator) !Credentials {
    // Keychain I/O is macOS-only. Off-macOS, env `=keychain` is disclosed by
    // `abi auth status` but load falls back to the file store — Windows/
    // Linux Secret Service remain Proposed (never silent empty success).
    if (credentialsBackendIsKeychain() and comptime builtin.os.tag == .macos) {
        return try keychain_backend.loadCredentialsFromKeychain(allocator);
    }

    const path = try getCredentialsPath(allocator);
    defer allocator.free(path);

    return try file.loadCredentialsFromPath(allocator, path);
}

pub fn saveCredentials(allocator: std.mem.Allocator, creds: Credentials) !void {
    if (credentialsBackendIsKeychain() and comptime builtin.os.tag == .macos) {
        return try keychain_backend.saveCredentialsToKeychain(creds);
    }

    const path = try getCredentialsPath(allocator);
    defer allocator.free(path);

    const dir = utils.pathDirname(path);
    try file.ensureCredentialsDir(dir);

    try file.saveCredentialsToPath(allocator, path, creds);
}

/// Clear keychain-held secrets when the keychain backend is active on macOS.
/// Off-macOS this is a no-op success: there is no linked keychain backend to
/// clear (Windows/Linux remain Proposed).
pub fn clearKeychainCredentials() !void {
    if (comptime builtin.os.tag != .macos) return;
    if (!credentialsBackendIsKeychain()) return;
    try keychain_backend.clearKeychainCredentials();
}

test {
    _ = types;
    _ = file;
    _ = keychain_backend;
    std.testing.refAllDecls(@This());
}

test "keychain env on non-macOS falls back to file load/save without KeychainUnsupported" {
    // Claim-honest gate: env may request keychain, but only macOS links SecItem.
    if (comptime builtin.os.tag == .macos) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const test_path = try temp_path.tempFilePath(allocator, "abi_credentials_keychain_fallback", "json");
    defer allocator.free(test_path);
    defer {
        var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{});
        defer threaded.deinit();
        std.Io.Dir.deleteFileAbsolute(threaded.io(), test_path) catch {};
    }

    var environ = std.process.Environ.Map.init(allocator);
    defer environ.deinit();
    try environ.put("ABI_CREDENTIALS_BACKEND", "keychain");
    try environ.put("ABI_CREDENTIALS_PATH", test_path);
    env.install(&environ);
    defer env.resetForTesting();

    try std.testing.expect(credentialsBackendIsKeychain());

    var creds = Credentials{
        .openai_api_key = try allocator.dupe(u8, "sk-file-fallback"),
    };
    defer creds.deinit(allocator);
    try saveCredentials(allocator, creds);

    var loaded = try loadCredentials(allocator);
    defer loaded.deinit(allocator);
    try std.testing.expectEqualStrings("sk-file-fallback", loaded.openai_api_key orelse return error.MissingKey);
    try clearKeychainCredentials(); // no-op success off-macOS
}

test "keychain backend save/load/clear round trip through the public API (opt-in, hits real keychain)" {
    // Exercises loadCredentials/saveCredentials/clearKeychainCredentials
    // end-to-end against the real macOS login keychain, including the "a
    // cleared field must delete the stale keychain entry, not just skip
    // writing it" requirement. Same real-keychain / trust-prompt caution as
    // keychain.zig's own round-trip test, so it needs the same opt-in.
    //
    // `zig build test` never calls `env.install()`, so `env.get` always
    // reports "unset" for the *real* process environment inside a test
    // binary. This outer gate therefore reads libc `getenv` directly (a
    // narrow, test-only, macOS-only exception — see keychain.zig for the
    // full rationale). The inner `ABI_CREDENTIALS_BACKEND=keychain` value is
    // intentionally simulated via `env.install`, same as the pure-logic
    // backend-selection tests above.
    if (comptime builtin.target.os.tag != .macos) return error.SkipZigTest;
    if (std.c.getenv("ABI_KEYCHAIN_TEST") == null) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var environ = std.process.Environ.Map.init(allocator);
    defer environ.deinit();
    try environ.put("ABI_CREDENTIALS_BACKEND", "keychain");
    env.install(&environ);
    defer env.resetForTesting();

    defer clearKeychainCredentials() catch |err| {
        std.log.warn("keychain backend test cleanup failed: {s}", .{@errorName(err)});
    };

    var creds = Credentials{
        .openai_api_key = try allocator.dupe(u8, "sk-keychain-test"),
        .discord_token = try allocator.dupe(u8, "discord-keychain-test"),
    };
    defer creds.deinit(allocator);
    try saveCredentials(allocator, creds);

    var loaded = try loadCredentials(allocator);
    defer loaded.deinit(allocator);
    try std.testing.expectEqualStrings("sk-keychain-test", loaded.openai_api_key orelse return error.MissingOpenAiKey);
    try std.testing.expectEqualStrings("discord-keychain-test", loaded.discord_token orelse return error.MissingDiscordToken);
    try std.testing.expect(loaded.anthropic_api_key == null);

    // Save again with discord_token absent: the stale keychain entry must be
    // deleted, not merely left un-overwritten.
    var updated = Credentials{
        .openai_api_key = try allocator.dupe(u8, "sk-keychain-test"),
    };
    defer updated.deinit(allocator);
    try saveCredentials(allocator, updated);

    var reloaded = try loadCredentials(allocator);
    defer reloaded.deinit(allocator);
    try std.testing.expectEqualStrings("sk-keychain-test", reloaded.openai_api_key orelse return error.MissingOpenAiKeyAfterUpdate);
    try std.testing.expect(reloaded.discord_token == null);

    try clearKeychainCredentials();
    var cleared = try loadCredentials(allocator);
    defer cleared.deinit(allocator);
    try std.testing.expect(cleared.openai_api_key == null);
    try std.testing.expect(cleared.discord_token == null);
}
