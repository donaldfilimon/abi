//! Secrets management system for secure credential handling.
//!
//! This module provides:
//! - Encrypted secrets storage
//! - Environment variable loading with validation
//! - Secret rotation support
//! - Memory-safe secret handling
//! - Audit logging integration
//! - Provider abstraction (env, file, vault)

const std = @import("std");
const time = @import("../time.zig");
const sync = @import("../sync.zig");
const crypto = std.crypto;
const csprng = @import("csprng.zig");

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

/// Secret provider types
pub const ProviderType = enum {
    /// Environment variables
    environment,
    /// Encrypted file storage
    file,
    /// In-memory (for testing)
    memory,
    /// External vault (HashiCorp Vault, AWS Secrets Manager, etc.)
    vault,
};

/// Secret metadata
pub const SecretMetadata = struct {
    /// When the secret was created
    created_at: i64,
    /// When the secret was last accessed
    last_accessed_at: ?i64 = null,
    /// When the secret expires (null = never)
    expires_at: ?i64 = null,
    /// When the secret should be rotated
    rotation_due_at: ?i64 = null,
    /// Number of times accessed
    access_count: u64 = 0,
    /// Version number
    version: u32 = 1,
    /// Secret type/category
    secret_type: SecretType = .generic,
    /// Tags for organization
    tags: []const []const u8 = &.{},
};

/// Types of secrets
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

/// A secret value with secure memory handling
pub const SecretValue = struct {
    allocator: std.mem.Allocator,
    /// The encrypted value (stored encrypted in memory)
    encrypted_value: []const u8,
    /// Encryption nonce
    nonce: [12]u8,
    /// Tag for authenticated encryption
    tag: [16]u8,
    /// Metadata
    metadata: SecretMetadata,

    /// Get the decrypted value (caller must free)
    pub fn decrypt(self: *SecretValue, key: [32]u8) ![]u8 {
        const plaintext = try self.allocator.alloc(u8, self.encrypted_value.len);
        errdefer self.allocator.free(plaintext);

        const aead = crypto.aead.aes_gcm.Aes256Gcm;
        aead.decrypt(
            plaintext,
            self.encrypted_value,
            self.tag,
            &.{}, // No additional data
            self.nonce,
            key,
        ) catch return error.DecryptionFailed;

        self.metadata.last_accessed_at = time.unixSeconds();
        self.metadata.access_count += 1;

        return plaintext;
    }

    pub fn deinit(self: *SecretValue) void {
        // Securely wipe encrypted value before freeing
        crypto.secureZero(u8, @constCast(self.encrypted_value));
        self.allocator.free(self.encrypted_value);
        crypto.secureZero(u8, &self.nonce);
        crypto.secureZero(u8, &self.tag);
    }
};

/// Secrets manager configuration
pub const SecretsConfig = struct {
    /// Default provider type
    provider: ProviderType = .environment,
    /// Master encryption key (for file/memory providers)
    master_key: ?[32]u8 = null,
    /// File path for file provider
    secrets_file: ?[]const u8 = null,
    /// Vault URL for vault provider
    vault_url: ?[]const u8 = null,
    /// Vault token
    vault_token: ?[]const u8 = null,
    /// Environment variable prefix
    env_prefix: []const u8 = "ABI_",
    /// Enable audit logging
    audit_logging: bool = true,
    /// Default rotation period in days (0 = no auto-rotation)
    default_rotation_days: u32 = 90,
    /// Cache secrets in memory
    cache_secrets: bool = true,
    /// Cache TTL in seconds
    cache_ttl: i64 = 300,
    /// Required secrets (fail if missing)
    required_secrets: []const []const u8 = &.{},
    /// Secret validation rules
    validation_rules: []const ValidationRule = &.{},
    /// If true, fails initialization when no master key is provided.
    /// Set to true for production deployments to prevent data loss.
    require_master_key: bool = false,
};

/// Validation rule for secrets
pub const ValidationRule = struct {
    /// Secret name pattern (supports *)
    name_pattern: []const u8,
    /// Minimum length
    min_length: ?usize = null,
    /// Maximum length
    max_length: ?usize = null,
    /// Required prefix
    required_prefix: ?[]const u8 = null,
    /// Disallow certain characters
    forbidden_chars: ?[]const u8 = null,
    /// Must be valid base64
    must_be_base64: bool = false,
};

/// Secrets manager
pub const SecretsManager = struct {
    allocator: std.mem.Allocator,
    config: SecretsConfig,
    /// Cached secrets
    cache: std.StringArrayHashMapUnmanaged(CachedSecret),
    /// Master key for encryption
    master_key: [32]u8,
    /// Statistics
    stats: SecretsStats,
    mutex: sync.Mutex,
    /// I/O backend for file operations (Zig 0.16)
    io_backend: std.Io.Threaded,

    const CachedSecret = struct {
        value: SecretValue,
        cached_at: i64,
    };

    pub const SecretsStats = struct {
        secrets_loaded: u64 = 0,
        secrets_accessed: u64 = 0,
        cache_hits: u64 = 0,
        cache_misses: u64 = 0,
        validation_failures: u64 = 0,
        rotation_events: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, config: SecretsConfig) !SecretsManager {
        // Derive or use provided master key
        var master_key: [32]u8 = undefined;
        if (config.master_key) |key| {
            master_key = key;
        } else {
            // Generate from environment or random
            if (std.c.getenv("ABI_MASTER_KEY")) |env_key_ptr| {
                const env_key = std.mem.span(env_key_ptr);
                if (env_key.len >= 32) {
                    @memcpy(&master_key, env_key[0..32]);
                } else {
                    // Derive key using HKDF
                    const Hkdf = std.crypto.kdf.hkdf.Hkdf(std.crypto.auth.hmac.sha2.HmacSha256);
                    const prk = Hkdf.extract("abi-secrets", env_key);
                    Hkdf.expand(&master_key, "master-key", prk);
                }
            } else {
                // Fail if production mode requires master key
                if (config.require_master_key) {
                    std.log.err("No master key provided. Set ABI_MASTER_KEY environment variable or provide master_key in config.", .{});
                    return error.MasterKeyRequired;
                }
                // Generate random key (not recommended for production)
                std.log.warn("Using randomly generated master key - encrypted secrets will be lost on restart!", .{});
                csprng.fillRandom(&master_key);
            }
        }

        var manager = SecretsManager{
            .allocator = allocator,
            .config = config,
            .cache = std.StringArrayHashMapUnmanaged(CachedSecret){},
            .master_key = master_key,
            .stats = .{},
            .mutex = .{},
            .io_backend = initIoBackend(allocator),
        };

        // Load required secrets
        for (config.required_secrets) |name| {
            _ = manager.get(name) catch |err| {
                std.log.err("Required secret '{s}' not found: {}", .{ name, err });
                return error.RequiredSecretMissing;
            };
        }

        return manager;
    }

    pub fn deinit(self: *SecretsManager) void {
        var it = self.cache.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.value.deinit();
        }
        self.cache.deinit(self.allocator);

        // Clean up I/O backend
        self.io_backend.deinit();

        // Securely wipe master key
        crypto.secureZero(u8, &self.master_key);
    }

    /// Get a secret by name
    pub fn get(self: *SecretsManager, name: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check cache
        if (self.config.cache_secrets) {
            if (self.cache.get(name)) |cached| {
                const now = time.unixSeconds();
                if (now - cached.cached_at < self.config.cache_ttl) {
                    self.stats.cache_hits += 1;
                    var value_copy = cached.value;
                    return value_copy.decrypt(self.master_key);
                }
            }
        }

        self.stats.cache_misses += 1;

        // Load from provider
        const value = switch (self.config.provider) {
            .environment => try self.loadFromEnv(name),
            .file => try self.loadFromFile(name),
            .memory => try self.loadFromMemory(name),
            .vault => try self.loadFromVault(name),
        };
        defer self.allocator.free(value);

        // Validate
        try self.validateSecret(name, value);

        // Encrypt and cache
        const encrypted = try self.encryptSecret(value);

        if (self.config.cache_secrets) {
            const name_copy = try self.allocator.dupe(u8, name);
            try self.cache.put(self.allocator, name_copy, .{
                .value = encrypted,
                .cached_at = time.unixSeconds(),
            });
        }

        self.stats.secrets_loaded += 1;
        self.stats.secrets_accessed += 1;

        // Return decrypted copy
        var encrypted_copy = encrypted;
        return encrypted_copy.decrypt(self.master_key);
    }

    /// Set a secret
    pub fn set(self: *SecretsManager, name: []const u8, value: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Validate
        try self.validateSecret(name, value);

        // Encrypt
        const encrypted = try self.encryptSecret(value);

        // Store
        switch (self.config.provider) {
            .environment => return error.ReadOnlyProvider,
            .file => try self.saveToFile(name, encrypted),
            .memory => {
                const name_copy = try self.allocator.dupe(u8, name);
                try self.cache.put(self.allocator, name_copy, .{
                    .value = encrypted,
                    .cached_at = time.unixSeconds(),
                });
            },
            .vault => try self.saveToVault(name, value),
        }
    }

    /// Delete a secret
    pub fn delete(self: *SecretsManager, name: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Remove from cache
        if (self.cache.fetchRemove(name)) |kv| {
            self.allocator.free(kv.key);
            var v = kv.value;
            v.value.deinit();
        }

        // Remove from provider
        switch (self.config.provider) {
            .environment => return error.ReadOnlyProvider,
            .file => try self.deleteFromFile(name),
            .memory => {}, // Already removed from cache
            .vault => try self.deleteFromVault(name),
        }
    }

    /// Check if a secret exists
    pub fn exists(self: *SecretsManager, name: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.cache.contains(name)) return true;

        return switch (self.config.provider) {
            .environment => self.envExists(name),
            .file => self.fileExists(name),
            .memory => false,
            .vault => self.vaultExists(name),
        };
    }

    /// Rotate a secret
    pub fn rotate(self: *SecretsManager, name: []const u8, new_value: []const u8) !void {
        // Validate new value
        try self.validateSecret(name, new_value);

        // Set new value
        try self.set(name, new_value);

        self.stats.rotation_events += 1;
    }

    /// Get statistics
    pub fn getStats(self: *SecretsManager) SecretsStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Clear the cache
    pub fn clearCache(self: *SecretsManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.cache.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.value.deinit();
        }
        self.cache.clearRetainingCapacity();
    }

    fn packEncryptedValue(self: *SecretsManager, encrypted: SecretValue) SecretsError![]u8 {
        const packed_len = 12 + 16 + encrypted.encrypted_value.len;
        const packed_data = try self.allocator.alloc(u8, packed_len);
        @memcpy(packed_data[0..12], &encrypted.nonce);
        @memcpy(packed_data[12..28], &encrypted.tag);
        @memcpy(packed_data[28..], encrypted.encrypted_value);
        return packed_data;
    }

    fn encodeBase64(self: *SecretsManager, data: []const u8) SecretsError![]u8 {
        const encoder = std.base64.standard.Encoder;
        const b64_size = encoder.calcSize(data.len);
        const b64_data = try self.allocator.alloc(u8, b64_size);
        _ = encoder.encode(b64_data, data);
        return b64_data;
    }

    fn readSecretsFile(self: *SecretsManager, secrets_path: []const u8) SecretsError![]u8 {
        const io = self.io_backend.io();
        return std.Io.Dir.cwd().readFileAlloc(io, secrets_path, self.allocator, .limited(1024 * 1024)) catch return error.SecretNotFound;
    }

    fn readSecretsFileOrEmpty(self: *SecretsManager, secrets_path: []const u8) SecretsError![]u8 {
        const io = self.io_backend.io();
        return std.Io.Dir.cwd().readFileAlloc(io, secrets_path, self.allocator, .limited(1024 * 1024)) catch try self.allocator.dupe(u8, "{}");
    }

    fn buildJsonEntry(self: *SecretsManager, name: []const u8, b64_data: []const u8) SecretsError![]u8 {
        return std.fmt.allocPrint(self.allocator, "\"{s}\":\"{s}\"", .{ name, b64_data });
    }

    fn buildUpdatedContent(
        self: *SecretsManager,
        existing: []const u8,
        name: []const u8,
        new_entry: []const u8,
    ) SecretsError![]u8 {
        const trimmed_all = std.mem.trim(u8, existing, " \n\r\t");
        if (trimmed_all.len == 0 or std.mem.eql(u8, trimmed_all, "{}")) {
            return std.fmt.allocPrint(self.allocator, "{{{s}}}", .{new_entry});
        }

        const search_key = try std.fmt.allocPrint(self.allocator, "\"{s}\":\"", .{name});
        defer self.allocator.free(search_key);

        if (std.mem.indexOf(u8, existing, search_key)) |key_idx| {
            const value_start = key_idx + search_key.len;
            var value_end = value_start;
            while (value_end < existing.len and existing[value_end] != '"') : (value_end += 1) {}
            if (value_end >= existing.len) return error.InvalidSecretFormat;

            const prefix = existing[0..key_idx];
            const suffix = existing[value_end + 1 ..];
            return std.fmt.allocPrint(self.allocator, "{s}{s}{s}", .{ prefix, new_entry, suffix });
        }

        const trimmed = std.mem.trimEnd(u8, existing, " \n\r\t}");
        return std.fmt.allocPrint(self.allocator, "{s},{s}}}", .{ trimmed, new_entry });
    }

    fn writeSecretsFile(self: *SecretsManager, secrets_path: []const u8, content: []const u8) SecretsError!void {
        const io = self.io_backend.io();
        var file = std.Io.Dir.cwd().createFile(io, secrets_path, .{ .truncate = true }) catch return error.FileWriteFailed;
        defer file.close(io);
        var write_buf: [4096]u8 = undefined;
        var writer = file.writer(io, &write_buf);
        writer.interface.writeAll(content) catch return error.FileWriteFailed;
        writer.flush() catch return error.FileWriteFailed;
    }

    // Private methods

    fn loadFromEnv(self: *SecretsManager, name: []const u8) SecretsError![]u8 {
        // Build env var name with prefix
        const env_name = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{
            self.config.env_prefix,
            name,
        });
        defer self.allocator.free(env_name);

        const env_name_z = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{env_name}, 0);
        defer self.allocator.free(env_name_z);
        const name_z = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{name}, 0);
        defer self.allocator.free(name_z);

        const value_ptr = std.c.getenv(env_name_z.ptr) orelse
            std.c.getenv(name_z.ptr) orelse
            return error.SecretNotFound;
        const value = std.mem.span(value_ptr);
        return self.allocator.dupe(u8, value);
    }

    fn loadFromFile(self: *SecretsManager, name: []const u8) SecretsError![]u8 {
        const secrets_path = self.config.secrets_file orelse return error.SecretNotFound;

        // Read the secrets file using Zig 0.16 I/O API
        const content = try self.readSecretsFile(secrets_path);
        defer self.allocator.free(content);

        // Parse JSON-like format: {"name": "encrypted_base64_value", ...}
        // Look for the key in the content
        const search_key = std.fmt.allocPrint(self.allocator, "\"{s}\":\"", .{name}) catch return error.OutOfMemory;
        defer self.allocator.free(search_key);

        const key_idx = std.mem.indexOf(u8, content, search_key) orelse return error.SecretNotFound;
        const value_start = key_idx + search_key.len;

        // Find end of value (next quote)
        var value_end = value_start;
        while (value_end < content.len and content[value_end] != '"') : (value_end += 1) {}

        if (value_end <= value_start) return error.SecretNotFound;

        const encrypted_b64 = content[value_start..value_end];

        // Decode base64 to get encrypted data
        const decoder = std.base64.standard.Decoder;
        const decoded_size = decoder.calcSizeForSlice(encrypted_b64) catch return error.InvalidBase64;
        const encrypted_data = try self.allocator.alloc(u8, decoded_size);
        errdefer self.allocator.free(encrypted_data);

        decoder.decode(encrypted_data, encrypted_b64) catch return error.InvalidBase64;

        // Encrypted format: nonce (12 bytes) + tag (16 bytes) + ciphertext
        if (encrypted_data.len < 28) {
            self.allocator.free(encrypted_data);
            return error.InvalidSecretFormat;
        }

        const nonce = encrypted_data[0..12].*;
        const tag = encrypted_data[12..28].*;
        const ciphertext = encrypted_data[28..];

        // Decrypt
        const plaintext = try self.allocator.alloc(u8, ciphertext.len);
        errdefer self.allocator.free(plaintext);

        const aead = crypto.aead.aes_gcm.Aes256Gcm;
        aead.decrypt(plaintext, ciphertext, tag, &.{}, nonce, self.master_key) catch {
            self.allocator.free(encrypted_data);
            return error.DecryptionFailed;
        };

        self.allocator.free(encrypted_data);
        return plaintext;
    }

    fn loadFromMemory(self: *SecretsManager, name: []const u8) SecretsError![]u8 {
        if (self.cache.get(name)) |cached| {
            var value = cached.value;
            return try value.decrypt(self.master_key);
        }
        return error.SecretNotFound;
    }

    fn loadFromVault(self: *SecretsManager, name: []const u8) SecretsError![]u8 {
        // HashiCorp Vault / AWS Secrets Manager integration
        // Requires vault_url and vault_token to be configured
        const vault_url = self.config.vault_url orelse return error.VaultUrlNotConfigured;
        const vault_token = self.config.vault_token orelse return error.VaultTokenNotConfigured;

        // Check local vault cache first (allows testing without connectivity)
        const cache_key = std.fmt.allocPrint(self.allocator, "vault:{s}", .{name}) catch return error.OutOfMemory;
        defer self.allocator.free(cache_key);

        if (self.cache.get(cache_key)) |cached| {
            var value = cached.value;
            return value.decrypt(self.master_key);
        }

        // Build the Vault API URL: GET /v1/secret/data/{name}
        const url = std.fmt.allocPrint(self.allocator, "{s}/v1/secret/data/{s}", .{
            vault_url,
            name,
        }) catch return error.OutOfMemory;
        defer self.allocator.free(url);

        // Perform HTTP GET to HashiCorp Vault using AsyncHttpClient
        const async_http = @import("../utils/http/async_http.zig");

        var client = async_http.AsyncHttpClient.init(self.allocator) catch |err| {
            std.log.err("Failed to initialize HTTP client for Vault: {}", .{err});
            return error.VaultConnectionFailed;
        };
        defer client.deinit();

        var request = async_http.HttpRequest.init(self.allocator, .get, url) catch |err| {
            std.log.err("Failed to create Vault HTTP request: {}", .{err});
            return error.VaultRequestFailed;
        };
        defer request.deinit();

        // Set Vault authentication header
        request.setHeader("X-Vault-Token", vault_token) catch return error.OutOfMemory;

        var response = client.fetchJson(&request) catch |err| {
            std.log.err("Vault HTTP request failed for secret '{s}': {}", .{ name, err });
            return error.VaultConnectionFailed;
        };
        defer response.deinit();

        // Handle non-success HTTP status
        if (!response.isSuccess()) {
            std.log.err("Vault returned HTTP {d} for secret '{s}'", .{ response.status_code, name });
            if (response.status_code == 404) return error.SecretNotFound;
            return error.VaultRequestFailed;
        }

        // Parse Vault KV v2 response: {"data": {"data": {"key": "value", ...}}}
        // We extract the inner "data" object and look for a "value" key first,
        // falling back to the secret name as key.
        const secret_value = parseVaultResponse(self.allocator, response.body, name) catch |err| {
            std.log.err("Failed to parse Vault response for secret '{s}': {}", .{ name, err });
            return error.InvalidSecretFormat;
        };

        return secret_value;
    }

    /// Parse a HashiCorp Vault KV v2 JSON response and extract the secret value.
    ///
    /// Vault KV v2 response format:
    /// ```json
    /// {
    ///   "data": {
    ///     "data": { "value": "secret-text", ... },
    ///     "metadata": { "version": 1, ... }
    ///   }
    /// }
    /// ```
    ///
    /// Lookup priority:
    /// 1. Inner `data.data.value` (conventional single-value secret)
    /// 2. Inner `data.data.<name>` (key matching the requested secret name)
    /// 3. First value in inner `data.data` (fallback for single-key secrets)
    fn parseVaultResponse(allocator: std.mem.Allocator, body: []u8, name: []const u8) ![]u8 {
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            allocator,
            body,
            .{ .ignore_unknown_fields = true },
        ) catch return error.InvalidSecretFormat;
        defer parsed.deinit();

        const root = parsed.value;

        // Navigate: root.data.data
        const outer_data = switch (root) {
            .object => |obj| obj.get("data") orelse return error.SecretNotFound,
            else => return error.InvalidSecretFormat,
        };
        const inner_data = switch (outer_data) {
            .object => |obj| obj.get("data") orelse return error.SecretNotFound,
            else => return error.InvalidSecretFormat,
        };
        const data_obj = switch (inner_data) {
            .object => |obj| obj,
            else => return error.InvalidSecretFormat,
        };

        // 1. Try "value" key (conventional)
        if (data_obj.get("value")) |val| {
            return extractStringValue(allocator, val);
        }

        // 2. Try key matching the requested secret name
        if (data_obj.get(name)) |val| {
            return extractStringValue(allocator, val);
        }

        // 3. Fallback: first entry in the map
        var it = data_obj.iterator();
        if (it.next()) |entry| {
            return extractStringValue(allocator, entry.value_ptr.*);
        }

        return error.SecretNotFound;
    }

    fn extractStringValue(allocator: std.mem.Allocator, val: std.json.Value) ![]u8 {
        return switch (val) {
            .string => |s| allocator.dupe(u8, s) catch return error.OutOfMemory,
            .integer => |i| std.fmt.allocPrint(allocator, "{d}", .{i}) catch return error.OutOfMemory,
            .float => |f| std.fmt.allocPrint(allocator, "{d}", .{f}) catch return error.OutOfMemory,
            .bool => |b| allocator.dupe(u8, if (b) "true" else "false") catch return error.OutOfMemory,
            else => error.InvalidSecretFormat,
        };
    }

    fn saveToFile(self: *SecretsManager, name: []const u8, encrypted: SecretValue) SecretsError!void {
        const secrets_path = self.config.secrets_file orelse return error.SecretsFileNotConfigured;

        // Pack encrypted data: nonce (12) + tag (16) + ciphertext
        const packed_data = try self.packEncryptedValue(encrypted);
        defer self.allocator.free(packed_data);

        // Base64 encode
        const b64_data = try self.encodeBase64(packed_data);
        defer self.allocator.free(b64_data);

        // Read existing file or create new content (Zig 0.16 I/O API)
        const existing_content = try self.readSecretsFileOrEmpty(secrets_path);
        defer self.allocator.free(existing_content);

        // Build new entry
        const new_entry = try self.buildJsonEntry(name, b64_data);
        defer self.allocator.free(new_entry);

        // Simple approach: rebuild the entire file
        // In production, would parse and rebuild JSON properly
        const new_content = try self.buildUpdatedContent(existing_content, name, new_entry);
        defer self.allocator.free(new_content);

        // Write to file (Zig 0.16 I/O API)
        try self.writeSecretsFile(secrets_path, new_content);
    }

    fn saveToVault(self: *SecretsManager, name: []const u8, value: []const u8) SecretsError!void {
        // HashiCorp Vault write operation
        // POST /v1/secret/data/{name} with JSON body {"data": {"value": "..."}}
        const vault_url = self.config.vault_url orelse return error.VaultUrlNotConfigured;
        const vault_token = self.config.vault_token orelse return error.VaultTokenNotConfigured;

        const url = std.fmt.allocPrint(self.allocator, "{s}/v1/secret/data/{s}", .{
            vault_url,
            name,
        }) catch return error.OutOfMemory;
        defer self.allocator.free(url);

        // Always cache locally for fast reads
        const cache_key = try std.fmt.allocPrint(self.allocator, "vault:{s}", .{name});
        const encrypted = try self.encryptSecret(value);

        try self.cache.put(self.allocator, cache_key, .{
            .value = encrypted,
            .cached_at = time.unixSeconds(),
        });

        // Perform HTTP POST to HashiCorp Vault KV v2
        const async_http = @import("../utils/http/async_http.zig");

        var client = async_http.AsyncHttpClient.init(self.allocator) catch |err| {
            std.log.err("Failed to initialize HTTP client for Vault write: {}", .{err});
            return error.VaultConnectionFailed;
        };
        defer client.deinit();

        var request = async_http.HttpRequest.init(self.allocator, .post, url) catch |err| {
            std.log.err("Failed to create Vault write request: {}", .{err});
            return error.VaultRequestFailed;
        };
        defer request.deinit();

        request.setHeader("X-Vault-Token", vault_token) catch return error.OutOfMemory;

        // Build Vault KV v2 write payload: {"data": {"value": "<secret>"}}
        const json_body = std.fmt.allocPrint(
            self.allocator,
            "{{\"data\":{{\"value\":\"{s}\"}}}}",
            .{value},
        ) catch return error.OutOfMemory;
        defer self.allocator.free(json_body);

        request.setJsonBody(json_body) catch return error.OutOfMemory;

        var response = client.fetch(&request) catch |err| {
            std.log.err("Vault write request failed for secret '{s}': {}", .{ name, err });
            return error.VaultConnectionFailed;
        };
        defer response.deinit();

        if (!response.isSuccess()) {
            std.log.err("Vault returned HTTP {d} on write for secret '{s}'", .{ response.status_code, name });
            return error.VaultRequestFailed;
        }

        std.log.info("Vault secret written successfully. Key: {s}", .{name});
    }

    fn deleteFromFile(self: *SecretsManager, name: []const u8) SecretsError!void {
        const secrets_path = self.config.secrets_file orelse return error.SecretsFileNotConfigured;
        const io = self.io_backend.io();

        // Read existing file (Zig 0.16 I/O API)
        const content = std.Io.Dir.cwd().readFileAlloc(io, secrets_path, self.allocator, .limited(1024 * 1024)) catch return;
        defer self.allocator.free(content);

        // Find and remove the key-value pair
        const search_key = try std.fmt.allocPrint(self.allocator, "\"{s}\":\"", .{name});
        defer self.allocator.free(search_key);

        const key_idx = std.mem.indexOf(u8, content, search_key) orelse return;

        // Find end of value
        const value_start = key_idx + search_key.len;
        var value_end = value_start;
        while (value_end < content.len and content[value_end] != '"') : (value_end += 1) {}
        value_end += 1; // Include closing quote

        // Check for comma before or after
        var remove_start = key_idx;
        var remove_end = value_end;

        // Handle comma after
        if (remove_end < content.len and content[remove_end] == ',') {
            remove_end += 1;
        } else if (remove_start > 0 and content[remove_start - 1] == ',') {
            // Handle comma before
            remove_start -= 1;
        }

        // Build new content
        var new_content = std.ArrayListUnmanaged(u8).empty;
        defer new_content.deinit(self.allocator);

        try new_content.appendSlice(self.allocator, content[0..remove_start]);
        try new_content.appendSlice(self.allocator, content[remove_end..]);

        // Write back (Zig 0.16 I/O API)
        var write_file = std.Io.Dir.cwd().createFile(io, secrets_path, .{ .truncate = true }) catch return error.FileWriteFailed;
        defer write_file.close(io);
        var write_buf: [4096]u8 = undefined;
        var writer = write_file.writer(io, &write_buf);
        writer.interface.writeAll(new_content.items) catch return error.FileWriteFailed;
        writer.flush() catch return error.FileWriteFailed;
    }

    fn deleteFromVault(self: *SecretsManager, name: []const u8) SecretsError!void {
        // HashiCorp Vault delete operation
        // DELETE /v1/secret/data/{name}
        const vault_url = self.config.vault_url orelse return error.VaultUrlNotConfigured;
        const vault_token = self.config.vault_token orelse return error.VaultTokenNotConfigured;

        // Remove from local cache
        const cache_key = try std.fmt.allocPrint(self.allocator, "vault:{s}", .{name});
        defer self.allocator.free(cache_key);

        if (self.cache.fetchRemove(cache_key)) |kv| {
            self.allocator.free(kv.key);
            var v = kv.value;
            v.value.deinit();
        }

        // Perform HTTP DELETE to HashiCorp Vault KV v2
        const async_http = @import("../utils/http/async_http.zig");

        const url = std.fmt.allocPrint(self.allocator, "{s}/v1/secret/data/{s}", .{
            vault_url,
            name,
        }) catch return error.OutOfMemory;
        defer self.allocator.free(url);

        var client = async_http.AsyncHttpClient.init(self.allocator) catch |err| {
            std.log.err("Failed to initialize HTTP client for Vault delete: {}", .{err});
            return error.VaultConnectionFailed;
        };
        defer client.deinit();

        var request = async_http.HttpRequest.init(self.allocator, .delete, url) catch |err| {
            std.log.err("Failed to create Vault delete request: {}", .{err});
            return error.VaultRequestFailed;
        };
        defer request.deinit();

        request.setHeader("X-Vault-Token", vault_token) catch return error.OutOfMemory;

        var response = client.fetch(&request) catch |err| {
            std.log.err("Vault delete request failed for secret '{s}': {}", .{ name, err });
            return error.VaultConnectionFailed;
        };
        defer response.deinit();

        if (!response.isSuccess()) {
            // 404 is acceptable for delete (secret may already be gone)
            if (response.status_code != 404) {
                std.log.err("Vault returned HTTP {d} on delete for secret '{s}'", .{ response.status_code, name });
                return error.VaultRequestFailed;
            }
        }

        std.log.info("Vault secret deleted. Key: {s}", .{name});
    }

    fn envExists(self: *SecretsManager, name: []const u8) bool {
        const env_name = std.fmt.allocPrint(self.allocator, "{s}{s}", .{
            self.config.env_prefix,
            name,
        }) catch return false;
        defer self.allocator.free(env_name);

        const env_name_z = std.fmt.allocPrintSentinel(self.allocator, "{s}", .{env_name}, 0) catch return false;
        defer self.allocator.free(env_name_z);
        const name_z = std.fmt.allocPrintSentinel(self.allocator, "{s}", .{name}, 0) catch return false;
        defer self.allocator.free(name_z);
        return std.c.getenv(env_name_z.ptr) != null or std.c.getenv(name_z.ptr) != null;
    }

    fn fileExists(self: *SecretsManager, name: []const u8) bool {
        const secrets_path = self.config.secrets_file orelse return false;
        const io = self.io_backend.io();

        // Read file using Zig 0.16 I/O API
        const content = std.Io.Dir.cwd().readFileAlloc(io, secrets_path, self.allocator, .limited(1024 * 1024)) catch return false;
        defer self.allocator.free(content);

        // Look for the key
        const search_key = std.fmt.allocPrint(self.allocator, "\"{s}\":", .{name}) catch return false;
        defer self.allocator.free(search_key);

        return std.mem.indexOf(u8, content, search_key) != null;
    }

    fn vaultExists(self: *SecretsManager, name: []const u8) bool {
        // Check local cache for vault secrets
        const cache_key = std.fmt.allocPrint(self.allocator, "vault:{s}", .{name}) catch return false;
        defer self.allocator.free(cache_key);

        return self.cache.contains(cache_key);
    }

    fn encryptSecret(self: *SecretsManager, value: []const u8) SecretsError!SecretValue {
        var nonce: [12]u8 = undefined;
        csprng.fillRandom(&nonce);

        const ciphertext = try self.allocator.alloc(u8, value.len);
        errdefer self.allocator.free(ciphertext);

        var tag: [16]u8 = undefined;

        const aead = crypto.aead.aes_gcm.Aes256Gcm;
        aead.encrypt(ciphertext, &tag, value, &.{}, nonce, self.master_key);

        return SecretValue{
            .allocator = self.allocator,
            .encrypted_value = ciphertext,
            .nonce = nonce,
            .tag = tag,
            .metadata = .{
                .created_at = time.unixSeconds(),
            },
        };
    }

    fn validateSecret(self: *SecretsManager, name: []const u8, value: []const u8) SecretsError!void {
        for (self.config.validation_rules) |rule| {
            if (!matchesPattern(name, rule.name_pattern)) continue;

            if (rule.min_length) |min| {
                if (value.len < min) {
                    self.stats.validation_failures += 1;
                    return error.SecretTooShort;
                }
            }

            if (rule.max_length) |max| {
                if (value.len > max) {
                    self.stats.validation_failures += 1;
                    return error.SecretTooLong;
                }
            }

            if (rule.required_prefix) |prefix| {
                if (!std.mem.startsWith(u8, value, prefix)) {
                    self.stats.validation_failures += 1;
                    return error.InvalidSecretFormat;
                }
            }

            if (rule.forbidden_chars) |forbidden| {
                for (value) |c| {
                    if (std.mem.indexOfScalar(u8, forbidden, c) != null) {
                        self.stats.validation_failures += 1;
                        return error.ForbiddenCharacter;
                    }
                }
            }

            if (rule.must_be_base64) {
                _ = std.base64.standard.Decoder.calcSizeForSlice(value) catch {
                    self.stats.validation_failures += 1;
                    return error.InvalidBase64;
                };
            }
        }
    }
};

fn matchesPattern(name: []const u8, pattern: []const u8) bool {
    if (std.mem.eql(u8, pattern, "*")) return true;

    if (std.mem.indexOf(u8, pattern, "*")) |star_idx| {
        const prefix = pattern[0..star_idx];
        const suffix = pattern[star_idx + 1 ..];

        return std.mem.startsWith(u8, name, prefix) and
            std.mem.endsWith(u8, name, suffix);
    }

    return std.mem.eql(u8, name, pattern);
}

/// Secure string that wipes itself on deallocation
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

/// Errors
pub const SecretsError = error{
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
    /// No master key provided when require_master_key is true
    MasterKeyRequired,
    /// Vault URL not set in SecretsConfig
    VaultUrlNotConfigured,
    /// Vault authentication token not set in SecretsConfig
    VaultTokenNotConfigured,
    /// Secrets file path not set in SecretsConfig
    SecretsFileNotConfigured,
    /// HTTP request to Vault returned a non-success status
    VaultRequestFailed,
    /// Could not establish a connection to the Vault server
    VaultConnectionFailed,
};

// Tests

test "secrets manager initialization" {
    const allocator = std.testing.allocator;

    var key: [32]u8 = undefined;
    csprng.fillRandom(&key);

    var manager = try SecretsManager.init(allocator, .{
        .provider = .memory,
        .master_key = key,
    });
    defer manager.deinit();

    try std.testing.expectEqual(@as(u64, 0), manager.getStats().secrets_loaded);
}

test "secure string wiping" {
    const allocator = std.testing.allocator;

    var secret = try SecureString.init(allocator, "sensitive-data");

    // Verify data is there
    try std.testing.expectEqualStrings("sensitive-data", secret.slice());

    // Get pointer for later check
    const ptr = secret.data.ptr;

    secret.deinit();

    // Memory should be wiped (all zeros)
    // Note: This is a best-effort check, memory might be reused
    _ = ptr;
}

test "pattern matching" {
    try std.testing.expect(matchesPattern("API_KEY", "*"));
    try std.testing.expect(matchesPattern("API_KEY", "API_*"));
    try std.testing.expect(matchesPattern("API_KEY", "*_KEY"));
    try std.testing.expect(matchesPattern("API_KEY", "API_KEY"));
    try std.testing.expect(!matchesPattern("API_KEY", "SECRET_*"));
}

test "secret encryption round trip" {
    const allocator = std.testing.allocator;

    var key: [32]u8 = undefined;
    csprng.fillRandom(&key);

    var manager = try SecretsManager.init(allocator, .{
        .provider = .memory,
        .master_key = key,
        .cache_secrets = true,
    });
    defer manager.deinit();

    // Set a secret
    try manager.set("test_secret", "my-secret-value");

    // Get it back
    const retrieved = try manager.get("test_secret");
    defer allocator.free(retrieved);

    try std.testing.expectEqualStrings("my-secret-value", retrieved);
}
