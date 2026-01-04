//! API key management for authentication and authorization.
const std = @import("std");

pub const ApiKeyConfig = struct {
    key_length: usize = 32,
    prefix: []const u8 = "abi_",
    hash_algorithm: HashAlgorithm = .sha256,
    storage_path: []const u8 = "api_keys.json",
    enable_rotation: bool = true,
    rotation_period_days: u64 = 90,
    max_keys_per_user: usize = 10,
};

pub const HashAlgorithm = enum {
    sha256,
    sha512,
    blake3,
};

pub const ApiKey = struct {
    id: []const u8,
    key_hash: []const u8,
    key_prefix: []const u8,
    user_id: []const u8,
    created_at: i64,
    expires_at: ?i64,
    last_used_at: ?i64,
    is_active: bool,
    scopes: []const []const u8,
    metadata: std.StringArrayHashMapUnmanaged([]const u8),
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
        var it = self.keys.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit(self.allocator);
            self.allocator.destroy(entry.value_ptr);
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
        const key_hash = try self.hashKey(key_plain);
        const key_prefix = key_plain[0..@min(8, key_plain.len)];

        const api_key = try self.allocator.create(ApiKey);
        errdefer self.allocator.destroy(api_key);

        api_key.* = ApiKey{
            .id = try self.allocator.dupe(u8, key_id),
            .key_hash = key_hash,
            .key_prefix = try self.allocator.dupe(u8, key_prefix),
            .user_id = try self.allocator.dupe(u8, user_id),
            .created_at = std.time.timestamp(),
            .expires_at = if (self.config.rotation_period_days > 0)
                std.time.timestamp() + @as(i64, @intCast(self.config.rotation_period_days * 86400))
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
        const key_hash = self.hashKey(key_plain) catch null;

        for (self.keys.values()) |key| {
            if (std.crypto.utils.timingSafeEql(key.key_hash, key_hash)) {
                if (!key.is_active) return null;
                if (key.expires_at) |exp| {
                    if (std.time.timestamp() > exp) return null;
                }
                key.last_used_at = std.time.timestamp();
                return key;
            }
        }

        return null;
    }

    pub fn revokeKey(self: *ApiKeyManager, key_id: []const u8) bool {
        if (self.keys.remove(key_id)) |entry| {
            entry.value_ptr.*.deinit(self.allocator);
            self.allocator.destroy(entry.value_ptr);
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
        std.crypto.random.bytes(key_bytes);
        return self.encodeKey(key_bytes);
    }

    fn hashKey(self: *ApiKeyManager, key: []const u8) ![]const u8 {
        var hash: [32]u8 = undefined;
        switch (self.config.hash_algorithm) {
            .sha256 => {
                var hasher = std.crypto.hash.sha2.Sha256.init(.{});
                hasher.update(key);
                hasher.final(&hash);
            },
            .sha512 => {
                var hasher = std.crypto.hash.sha2.Sha512.init(.{});
                hasher.update(key);
                var out: [64]u8 = undefined;
                hasher.final(&out);
                @memcpy(hash[0..32], out[0..32]);
            },
            .blake3 => {
                var hasher = std.crypto.hash.blake3.Blake3.init(.{});
                hasher.update(key);
                hasher.final(&hash);
            },
        }
        return self.allocator.dupe(u8, &hash);
    }

    fn encodeKey(self: *ApiKeyManager, key_bytes: []const u8) ![]const u8 {
        const encoder = std.base64.standard.Encoder;
        const encoded_len = encoder.calcSize(key_bytes.len);
        const encoded = try self.allocator.alloc(u8, encoded_len);
        encoder.encode(encoded, key_bytes);
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
    var manager = ApiKeyManager.init(allocator, .{});
    defer manager.deinit();

    const scopes = &.{ "read", "write" };
    const generated = try manager.generateKey("user1", scopes);

    try std.testing.expect(std.mem.startsWith(u8, generated.key_id, "abi_"));
    try std.testing.expect(generated.key_plain.len > 0);
    try std.testing.expectEqualStrings(generated.key_prefix, generated.key_plain[0..8]);
}

test "api key validation" {
    const allocator = std.testing.allocator;
    var manager = ApiKeyManager.init(allocator, .{});
    defer manager.deinit();

    const generated = try manager.generateKey("user1", &.{"read"});
    const validated = try manager.validateKey(generated.key_plain);

    try std.testing.expect(validated != null);
    try std.testing.expectEqualStrings("user1", validated.?.user_id);
}

test "api key revocation" {
    const allocator = std.testing.allocator;
    var manager = ApiKeyManager.init(allocator, .{});
    defer manager.deinit();

    const generated = try manager.generateKey("user1", &.{"read"});
    const revoked = manager.revokeKey(generated.key_id);
    try std.testing.expect(revoked);

    const validated = try manager.validateKey(generated.key_plain);
    try std.testing.expectEqual(null, validated);
}
