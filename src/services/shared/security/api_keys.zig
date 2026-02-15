//! API key management for authentication and authorization.
//!
//! Security features:
//! - Salted hashing with configurable KDF iterations
//! - Timing-safe comparison to prevent timing attacks
//! - Key rotation and expiration support
//! - Secure memory wiping for sensitive data
const std = @import("std");
const time = @import("../time.zig");
const csprng = @import("csprng.zig");

/// Constant-time comparison for variable-length slices (timing-attack resistant).
fn constantTimeEqlSlice(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, b) |x, y| {
        diff |= x ^ y;
    }
    return diff == 0;
}

/// Salt length in bytes for key hashing
pub const SALT_LENGTH: usize = 16;

/// Default number of hash iterations for key derivation
pub const DEFAULT_HASH_ITERATIONS: u32 = 100_000;

pub const ApiKeyConfig = struct {
    key_length: usize = 32,
    prefix: []const u8 = "abi_",
    hash_algorithm: HashAlgorithm = .blake3,
    storage_path: []const u8 = "api_keys.json",
    enable_rotation: bool = true,
    rotation_period_days: u64 = 90,
    max_keys_per_user: usize = 10,
    /// Number of iterations for key derivation (higher = more secure but slower)
    hash_iterations: u32 = DEFAULT_HASH_ITERATIONS,
};

pub const HashAlgorithm = enum {
    sha256,
    sha512,
    blake3,
};

/// Securely wipe memory to prevent sensitive data leakage.
/// Uses volatile writes to prevent compiler optimization.
fn secureWipe(data: anytype) void {
    const T = @TypeOf(data);
    const info = @typeInfo(T);

    if (info == .pointer) {
        const slice = switch (info.pointer.size) {
            .slice => data,
            .one => if (@typeInfo(info.pointer.child) == .array)
                @as([]u8, data)
            else
                @as(*[1]u8, @ptrCast(data))[0..1],
            else => return,
        };

        // Use std.crypto.secureZero for guaranteed wiping
        std.crypto.secureZero(u8, @constCast(slice));
    }
}

pub const ApiKey = struct {
    id: []const u8,
    key_hash: []const u8,
    /// Random salt used for this key's hash
    salt: [SALT_LENGTH]u8,
    key_prefix: []const u8,
    user_id: []const u8,
    created_at: i64,
    expires_at: ?i64,
    last_used_at: ?i64,
    is_active: bool,
    scopes: []const []const u8,
    metadata: std.StringArrayHashMapUnmanaged([]const u8),

    pub fn deinit(self: *ApiKey, allocator: std.mem.Allocator) void {
        // Securely wipe sensitive data before freeing
        secureWipe(self.key_hash);
        secureWipe(&self.salt);
        allocator.free(self.id);
        allocator.free(self.key_hash);
        allocator.free(self.key_prefix);
        allocator.free(self.user_id);
        for (self.scopes) |scope| {
            allocator.free(scope);
        }
        allocator.free(self.scopes);
        self.metadata.deinit(allocator);
    }
};

pub const ApiKeyManager = struct {
    allocator: std.mem.Allocator,
    config: ApiKeyConfig,
    keys: std.StringArrayHashMapUnmanaged(*ApiKey),
    key_id_counter: std.atomic.Value(u64),

    pub fn init(allocator: std.mem.Allocator, config: ApiKeyConfig) ApiKeyManager {
        return .{
            .allocator = allocator,
            .config = config,
            .keys = std.StringArrayHashMapUnmanaged(*ApiKey).empty,
            .key_id_counter = std.atomic.Value(u64).init(0),
        };
    }

    pub fn deinit(self: *ApiKeyManager) void {
        for (self.keys.values()) |api_key| {
            // ApiKey.deinit frees .id which is also the map key
            api_key.deinit(self.allocator);
            self.allocator.destroy(api_key);
        }
        self.keys.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn generateKey(
        self: *ApiKeyManager,
        user_id: []const u8,
        scopes: []const []const u8,
    ) !GeneratedKey {
        const key_id = try self.generateKeyId();
        const key_plain = try self.generateKeyPlain();

        // Generate random salt for this key
        var salt: [SALT_LENGTH]u8 = undefined;
        csprng.fillRandom(&salt);

        const key_hash = try self.hashKeyWithSalt(key_plain, &salt);
        const key_prefix = key_plain[0..@min(8, key_plain.len)];

        const api_key = try self.allocator.create(ApiKey);
        errdefer self.allocator.destroy(api_key);

        api_key.* = ApiKey{
            .id = key_id,
            .key_hash = key_hash,
            .salt = salt,
            .key_prefix = try self.allocator.dupe(u8, key_prefix),
            .user_id = try self.allocator.dupe(u8, user_id),
            .created_at = time.unixSeconds(),
            .expires_at = if (self.config.rotation_period_days > 0)
                time.unixSeconds() + @as(i64, @intCast(self.config.rotation_period_days * 86400))
            else
                null,
            .last_used_at = null,
            .is_active = true,
            .scopes = try self.duplicateScopes(scopes),
            .metadata = std.StringArrayHashMapUnmanaged([]const u8).empty,
        };

        try self.keys.put(self.allocator, api_key.id, api_key);

        return .{
            .key_id = api_key.id,
            .key_plain = key_plain,
            .key_prefix = api_key.key_prefix,
        };
    }

    pub fn validateKey(self: *ApiKeyManager, key_plain: []const u8) !?*ApiKey {
        for (self.keys.values()) |key| {
            // Hash the provided key with this key's salt
            const computed_hash = self.hashKeyWithSalt(key_plain, &key.salt) catch continue;
            defer self.allocator.free(computed_hash);

            // Timing-safe comparison to prevent timing attacks
            if (computed_hash.len == key.key_hash.len and
                constantTimeEqlSlice(computed_hash, key.key_hash))
            {
                if (!key.is_active) return null;
                if (key.expires_at) |exp| {
                    if (time.unixSeconds() > exp) return null;
                }
                key.last_used_at = time.unixSeconds();
                return key;
            }
        }

        return null;
    }

    pub fn revokeKey(self: *ApiKeyManager, key_id: []const u8) bool {
        if (self.keys.fetchOrderedRemove(key_id)) |entry| {
            entry.value.deinit(self.allocator);
            self.allocator.destroy(entry.value);
            return true;
        }
        return false;
    }

    pub fn rotateKey(self: *ApiKeyManager, key_id: []const u8, user_id: []const u8) !?GeneratedKey {
        const old_key = self.keys.get(key_id);
        if (old_key == null or !std.mem.eql(u8, old_key.?.user_id, user_id)) {
            return null;
        }

        _ = self.revokeKey(key_id);
        return try self.generateKey(user_id, old_key.?.scopes);
    }

    pub fn getKey(self: *ApiKeyManager, key_id: []const u8) ?*const ApiKey {
        return self.keys.get(key_id);
    }

    pub fn getKeysForUser(self: *ApiKeyManager, user_id: []const u8) []*ApiKey {
        var result = std.ArrayListUnmanaged(*ApiKey).empty;
        for (self.keys.values()) |key| {
            if (std.mem.eql(u8, key.user_id, user_id)) {
                result.appendAssumeCapacity(key);
            }
        }
        return result.items;
    }

    pub fn hasScope(self: *ApiKey, scope: []const u8) bool {
        for (self.scopes) |s| {
            if (std.mem.eql(u8, s, scope)) return true;
        }
        return false;
    }

    fn generateKeyId(self: *ApiKeyManager) ![]const u8 {
        const id = self.key_id_counter.fetchAdd(1, .monotonic);
        return std.fmt.allocPrint(self.allocator, "{s}{d}", .{ self.config.prefix, id });
    }

    fn generateKeyPlain(self: *ApiKeyManager) ![]const u8 {
        const key_bytes = try self.allocator.alloc(u8, self.config.key_length);
        defer self.allocator.free(key_bytes);
        csprng.fillRandom(key_bytes);
        return self.encodeKey(key_bytes);
    }

    /// Hash a key with the given salt using iterative hashing for key stretching.
    /// This provides resistance against brute-force attacks.
    fn hashKeyWithSalt(self: *ApiKeyManager, key: []const u8, salt: []const u8) ![]const u8 {
        var hash: [32]u8 = undefined;

        // Initial hash: salt || key
        switch (self.config.hash_algorithm) {
            .sha256 => {
                var hasher = std.crypto.hash.sha2.Sha256.init(.{});
                hasher.update(salt);
                hasher.update(key);
                hasher.final(&hash);

                // Iterative hashing for key stretching
                var i: u32 = 1;
                while (i < self.config.hash_iterations) : (i += 1) {
                    hasher = std.crypto.hash.sha2.Sha256.init(.{});
                    hasher.update(&hash);
                    hasher.update(salt);
                    hasher.final(&hash);
                }
            },
            .sha512 => {
                var hasher = std.crypto.hash.sha2.Sha512.init(.{});
                hasher.update(salt);
                hasher.update(key);
                var out: [64]u8 = undefined;
                hasher.final(&out);
                @memcpy(hash[0..32], out[0..32]);

                // Iterative hashing for key stretching
                var i: u32 = 1;
                while (i < self.config.hash_iterations) : (i += 1) {
                    hasher = std.crypto.hash.sha2.Sha512.init(.{});
                    hasher.update(&hash);
                    hasher.update(salt);
                    hasher.final(&out);
                    @memcpy(hash[0..32], out[0..32]);
                }
            },
            .blake3 => {
                var hasher = std.crypto.hash.Blake3.init(.{});
                hasher.update(salt);
                hasher.update(key);
                hasher.final(&hash);

                // Iterative hashing for key stretching
                var i: u32 = 1;
                while (i < self.config.hash_iterations) : (i += 1) {
                    hasher = std.crypto.hash.Blake3.init(.{});
                    hasher.update(&hash);
                    hasher.update(salt);
                    hasher.final(&hash);
                }
            },
        }
        return self.allocator.dupe(u8, &hash);
    }

    fn encodeKey(self: *ApiKeyManager, key_bytes: []const u8) ![]const u8 {
        const encoder = std.base64.standard.Encoder;
        const encoded_len = encoder.calcSize(key_bytes.len);
        const encoded = try self.allocator.alloc(u8, encoded_len);
        _ = encoder.encode(encoded, key_bytes);
        return encoded;
    }

    fn duplicateScopes(self: *ApiKeyManager, scopes: []const []const u8) ![]const []const u8 {
        const duped = try self.allocator.alloc([]const u8, scopes.len);
        for (scopes, 0..) |scope, i| {
            duped[i] = try self.allocator.dupe(u8, scope);
        }
        return duped;
    }
};

pub const GeneratedKey = struct {
    key_id: []const u8,
    key_plain: []const u8,
    key_prefix: []const u8,
};

pub const ApiKeyError = error{
    KeyNotFound,
    KeyExpired,
    KeyRevoked,
    InvalidKey,
    MaxKeysExceeded,
};

test "api key generation" {
    const allocator = std.testing.allocator;
    var manager = ApiKeyManager.init(allocator, .{ .hash_iterations = 10 });
    defer manager.deinit();

    const scopes = &.{ "read", "write" };
    const generated = try manager.generateKey("user1", scopes);
    defer allocator.free(generated.key_plain);

    try std.testing.expect(std.mem.startsWith(u8, generated.key_id, "abi_"));
    try std.testing.expect(generated.key_plain.len > 0);
    try std.testing.expectEqualStrings(generated.key_prefix, generated.key_plain[0..8]);
}

test "api key validation" {
    const allocator = std.testing.allocator;
    var manager = ApiKeyManager.init(allocator, .{ .hash_iterations = 10 });
    defer manager.deinit();

    const generated = try manager.generateKey("user1", &.{"read"});
    defer allocator.free(generated.key_plain);
    const validated = try manager.validateKey(generated.key_plain);

    try std.testing.expect(validated != null);
    try std.testing.expectEqualStrings("user1", validated.?.user_id);
}

test "api key revocation" {
    const allocator = std.testing.allocator;
    var manager = ApiKeyManager.init(allocator, .{ .hash_iterations = 10 });
    defer manager.deinit();

    const generated = try manager.generateKey("user1", &.{"read"});
    defer allocator.free(generated.key_plain);
    const revoked = manager.revokeKey(generated.key_id);
    try std.testing.expect(revoked);

    const validated = try manager.validateKey(generated.key_plain);
    try std.testing.expectEqual(null, validated);
}
