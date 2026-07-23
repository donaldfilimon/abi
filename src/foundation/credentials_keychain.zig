//! Opt-in macOS login-keychain credential backend glue.
//! Active only when `ABI_CREDENTIALS_BACKEND=keychain`. OS at-rest protection
//! via Security.framework SecItem — not hardware-backed (no Secure Enclave /
//! biometric), not audited, not headless-CI-safe.
//! Package-private building blocks for credentials.zig (hub).
const std = @import("std");
const env = @import("env.zig");
const keychain = @import("keychain.zig");
const types = @import("credentials_types.zig");

pub const Credentials = types.Credentials;

/// Service name under which all credential fields are stored in the OS
/// keychain when the keychain backend is active. Each field's `account` is
/// its struct field name (e.g. "openai_api_key").
const keychain_service = "abi-credentials";

/// True when `ABI_CREDENTIALS_BACKEND=keychain` is set. Any other value
/// (including unset) keeps the default file-based backend — no silent
/// migration of existing `~/.abi/credentials.json` users.
fn useKeychainBackend() bool {
    const v = env.get("ABI_CREDENTIALS_BACKEND") orelse return false;
    return std.mem.eql(u8, v, "keychain");
}

/// Whether the keychain backend is currently selected. Exposed so callers
/// like `auth logout` can clear keychain-stored secrets in addition to (or
/// instead of) the plaintext credential file.
pub fn credentialsBackendIsKeychain() bool {
    return useKeychainBackend();
}

pub fn loadCredentialsFromKeychain(allocator: std.mem.Allocator) !Credentials {
    var creds = Credentials{};
    errdefer creds.deinit(allocator);

    creds.openai_api_key = try keychain.keychainLoad(allocator, keychain_service, "openai_api_key");
    creds.anthropic_api_key = try keychain.keychainLoad(allocator, keychain_service, "anthropic_api_key");
    creds.discord_token = try keychain.keychainLoad(allocator, keychain_service, "discord_token");
    creds.grok_api_key = try keychain.keychainLoad(allocator, keychain_service, "grok_api_key");
    creds.twilio_account_sid = try keychain.keychainLoad(allocator, keychain_service, "twilio_account_sid");
    creds.twilio_auth_token = try keychain.keychainLoad(allocator, keychain_service, "twilio_auth_token");
    return creds;
}

/// Mirror `creds` into the OS keychain: present fields are stored/overwritten,
/// absent (null) fields are actively deleted so a cleared field does not
/// leave a stale secret behind. No dual-write: when this backend is active,
/// secrets never also land in plaintext `credentials.json`.
pub fn saveCredentialsToKeychain(creds: Credentials) !void {
    try saveKeychainField("openai_api_key", creds.openai_api_key);
    try saveKeychainField("anthropic_api_key", creds.anthropic_api_key);
    try saveKeychainField("discord_token", creds.discord_token);
    try saveKeychainField("grok_api_key", creds.grok_api_key);
    try saveKeychainField("twilio_account_sid", creds.twilio_account_sid);
    try saveKeychainField("twilio_auth_token", creds.twilio_auth_token);
}

fn saveKeychainField(account: []const u8, value: ?[]const u8) !void {
    if (value) |v| {
        try keychain.keychainStore(keychain_service, account, v);
    } else {
        try keychain.keychainDelete(keychain_service, account);
    }
}

/// Delete every keychain-stored credential field, regardless of which are
/// currently present. Used by `auth logout` when the keychain backend is
/// active, so logout clears keychain secrets the same way it clears the
/// plaintext file.
pub fn clearKeychainCredentials() !void {
    try saveCredentialsToKeychain(Credentials{});
}

test {
    std.testing.refAllDecls(@This());
}

test "credentialsBackendIsKeychain reads ABI_CREDENTIALS_BACKEND, default stays file-based" {
    // Pure env-parsing logic; does not touch the real OS keychain.
    try std.testing.expect(!credentialsBackendIsKeychain());

    var environ = std.process.Environ.Map.init(std.testing.allocator);
    defer environ.deinit();
    try environ.put("ABI_CREDENTIALS_BACKEND", "keychain");
    env.install(&environ);
    defer env.resetForTesting();

    try std.testing.expect(credentialsBackendIsKeychain());
}

test "credentialsBackendIsKeychain rejects unrecognized backend values" {
    var environ = std.process.Environ.Map.init(std.testing.allocator);
    defer environ.deinit();
    try environ.put("ABI_CREDENTIALS_BACKEND", "vault");
    env.install(&environ);
    defer env.resetForTesting();

    // No silent migration to an unknown backend: anything but exactly
    // "keychain" keeps the default file-based path.
    try std.testing.expect(!credentialsBackendIsKeychain());
}
