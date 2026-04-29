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
const os = @import("../os.zig");
const sync = @import("../sync.zig");
const time = @import("../time.zig");
const crypto = std.crypto;
const csprng = @import("csprng.zig");
const providers_mod = @import("secrets/providers.zig");
const shared = @import("secrets/shared.zig");
const validation = @import("secrets/validation.zig");

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{
        .environ = if (comptime !os.no_os) std.process.Environ.empty else .{},
    });
}

pub const ProviderType = shared.ProviderType;
pub const SecretMetadata = shared.SecretMetadata;
pub const SecretType = shared.SecretType;
pub const SecretValue = shared.SecretValue;
pub const VaultProviderType = shared.VaultProviderType;
pub const SecretsConfig = shared.SecretsConfig;
pub const ValidationRule = shared.ValidationRule;
pub const SecureString = shared.SecureString;
pub const SecretsError = shared.SecretsError;

pub const SecretsManager = struct {
    allocator: std.mem.Allocator,
    config: SecretsConfig,
    cache: std.StringArrayHashMapUnmanaged(CachedSecret),
    master_key: [32]u8,
    stats: SecretsStats,
    mutex: sync.Mutex,
    io_backend: std.Io.Threaded,

    const CachedSecret = struct {
        value: SecretValue,
        cached_at: i64,
    };

    const provider_impl = providers_mod.Providers(@This());

    pub const SecretsStats = struct {
        secrets_loaded: u64 = 0,
        secrets_accessed: u64 = 0,
        cache_hits: u64 = 0,
        cache_misses: u64 = 0,
        validation_failures: u64 = 0,
        rotation_events: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, config: SecretsConfig) !SecretsManager {
        var master_key: [32]u8 = undefined;
        if (config.master_key) |key| {
            master_key = key;
        } else {
            if (std.c.getenv("ABI_MASTER_KEY")) |env_key_ptr| {
                const env_key = std.mem.span(env_key_ptr);
                if (env_key.len >= 32) {
                    @memcpy(&master_key, env_key[0..32]);
                } else {
                    const Hkdf = std.crypto.kdf.hkdf.Hkdf(std.crypto.auth.hmac.sha2.HmacSha256);
                    const prk = Hkdf.extract("abi-secrets", env_key);
                    Hkdf.expand(&master_key, "master-key", prk);
                }
            } else {
                if (config.require_master_key) {
                    std.log.err("No master key provided. Set ABI_MASTER_KEY environment variable or provide master_key in config.", .{});
                    return error.MasterKeyRequired;
                }
                std.log.warn("Using randomly generated master key - encrypted secrets will be lost on restart!", .{});
                try csprng.fillRandom(&master_key);
            }
        }

        var manager = SecretsManager{
            .allocator = allocator,
            .config = config,
            .cache = std.StringArrayHashMapUnmanaged(CachedSecret).empty,
            .master_key = master_key,
            .stats = .{},
            .mutex = .{},
            .io_backend = initIoBackend(allocator),
        };

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

        self.io_backend.deinit();
        crypto.secureZero(u8, &self.master_key);
    }

    pub fn get(self: *SecretsManager, name: []const u8) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

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

        const value = switch (self.config.provider) {
            .environment => try provider_impl.loadFromEnv(self, name),
            .file => try provider_impl.loadFromFile(self, name),
            .memory => try provider_impl.loadFromMemory(self, name),
            .vault => try provider_impl.loadFromVault(self, name),
        };
        defer self.allocator.free(value);

        try self.validateSecret(name, value);

        var encrypted = try self.encryptSecret(value);

        if (self.config.cache_secrets) {
            const name_copy = try self.allocator.dupe(u8, name);
            try self.cache.put(self.allocator, name_copy, .{
                .value = encrypted,
                .cached_at = time.unixSeconds(),
            });
        }

        self.stats.secrets_loaded += 1;
        self.stats.secrets_accessed += 1;

        const decrypted = try encrypted.decrypt(self.master_key);
        if (!self.config.cache_secrets) {
            encrypted.deinit();
        }

        return decrypted;
    }

    pub fn set(self: *SecretsManager, name: []const u8, value: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.validateSecret(name, value);

        const encrypted = try self.encryptSecret(value);

        switch (self.config.provider) {
            .environment => return error.ReadOnlyProvider,
            .file => try provider_impl.saveToFile(self, name, encrypted),
            .memory => {
                const name_copy = try self.allocator.dupe(u8, name);
                try self.cache.put(self.allocator, name_copy, .{
                    .value = encrypted,
                    .cached_at = time.unixSeconds(),
                });
            },
            .vault => try provider_impl.saveToVault(self, name, value),
        }
    }

    pub fn delete(self: *SecretsManager, name: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.cache.fetchRemove(name)) |kv| {
            self.allocator.free(kv.key);
            var v = kv.value;
            v.value.deinit();
        }

        switch (self.config.provider) {
            .environment => return error.ReadOnlyProvider,
            .file => try provider_impl.deleteFromFile(self, name),
            .memory => {},
            .vault => try provider_impl.deleteFromVault(self, name),
        }
    }

    pub fn exists(self: *SecretsManager, name: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.cache.contains(name)) return true;

        return switch (self.config.provider) {
            .environment => provider_impl.envExists(self, name),
            .file => provider_impl.fileExists(self, name),
            .memory => false,
            .vault => provider_impl.vaultExists(self, name),
        };
    }

    pub fn rotate(self: *SecretsManager, name: []const u8, new_value: []const u8) !void {
        try self.validateSecret(name, new_value);
        try self.set(name, new_value);
        self.stats.rotation_events += 1;
    }

    pub fn getStats(self: *SecretsManager) SecretsStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

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

    fn encryptSecret(self: *SecretsManager, value: []const u8) SecretsError!SecretValue {
        var nonce: [12]u8 = undefined;
        csprng.fillRandom(&nonce) catch return error.DecryptionFailed;

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
        try validation.validateSecret(
            self.config.validation_rules,
            &self.stats.validation_failures,
            name,
            value,
        );
    }
};

test {
    std.testing.refAllDecls(@This());
}
