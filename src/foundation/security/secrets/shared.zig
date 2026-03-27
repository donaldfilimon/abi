const std = @import("std");
const time = @import("../../time.zig");
const crypto = std.crypto;

pub const SecretsError = error{
    NotImplemented,
    SecretNotFound,
    SecretTooShort,
    SecretTooLong,
    InvalidSecretFormat,
    ForbiddenCharacter,
    InvalidBase64,
    DecryptionFailed,
    ReadOnlyProvider,
    RequiredSecretMissing,
    OutOfMemory,
    FileWriteFailed,
    MasterKeyRequired,
    VaultUrlNotConfigured,
    VaultTokenNotConfigured,
    SecretsFileNotConfigured,
    VaultRequestFailed,
    VaultConnectionFailed,
};

pub const ProviderType = enum {
    environment,
    file,
    memory,
    vault,
};

pub const SecretType = enum {
    generic,
    api_key,
    password,
    certificate,
    private_key,
    database_credential,
    oauth_token,
    encryption_key,
    signing_key,
};

pub const SecretMetadata = struct {
    created_at: i64,
    last_accessed_at: ?i64 = null,
    expires_at: ?i64 = null,
    rotation_due_at: ?i64 = null,
    access_count: u64 = 0,
    version: u32 = 1,
    secret_type: SecretType = .generic,
    tags: []const []const u8 = &.{},
};

pub const SecretValue = struct {
    allocator: std.mem.Allocator,
    encrypted_value: []const u8,
    nonce: [12]u8,
    tag: [16]u8,
    metadata: SecretMetadata,

    pub fn decrypt(self: *SecretValue, key: [32]u8) SecretsError![]u8 {
        const plaintext = try self.allocator.alloc(u8, self.encrypted_value.len);
        errdefer self.allocator.free(plaintext);

        const aead = crypto.aead.aes_gcm.Aes256Gcm;
        aead.decrypt(
            plaintext,
            self.encrypted_value,
            self.tag,
            &.{},
            self.nonce,
            key,
        ) catch return error.DecryptionFailed;

        self.metadata.last_accessed_at = time.unixSeconds();
        self.metadata.access_count += 1;

        return plaintext;
    }

    pub fn deinit(self: *SecretValue) void {
        crypto.secureZero(u8, @constCast(self.encrypted_value));
        self.allocator.free(self.encrypted_value);
        crypto.secureZero(u8, &self.nonce);
        crypto.secureZero(u8, &self.tag);
    }
};

pub const VaultProviderType = enum {
    hashicorp,
    aws_secrets_manager,
    azure_key_vault,
};

pub const ValidationRule = struct {
    name_pattern: []const u8,
    min_length: ?usize = null,
    max_length: ?usize = null,
    required_prefix: ?[]const u8 = null,
    forbidden_chars: ?[]const u8 = null,
    must_be_base64: bool = false,
};

pub const SecretsConfig = struct {
    provider: ProviderType = .environment,
    master_key: ?[32]u8 = null,
    secrets_file: ?[]const u8 = null,
    vault_url: ?[]const u8 = null,
    vault_token: ?[]const u8 = null,
    vault_provider: VaultProviderType = .hashicorp,
    env_prefix: []const u8 = "ABI_",
    audit_logging: bool = true,
    default_rotation_days: u32 = 90,
    cache_secrets: bool = true,
    cache_ttl: i64 = 300,
    required_secrets: []const []const u8 = &.{},
    validation_rules: []const ValidationRule = &.{},
    require_master_key: bool = false,
};

pub const SecureString = struct {
    allocator: std.mem.Allocator,
    data: []u8,

    pub fn init(allocator: std.mem.Allocator, value: []const u8) !SecureString {
        const data = try allocator.alloc(u8, value.len);
        @memcpy(data, value);
        return .{
            .allocator = allocator,
            .data = data,
        };
    }

    pub fn deinit(self: *SecureString) void {
        crypto.secureZero(u8, self.data);
        self.allocator.free(self.data);
    }

    pub fn slice(self: SecureString) []const u8 {
        return self.data;
    }
};

test {
    std.testing.refAllDecls(@This());
}
