//! API Key Authentication
//!
//! Manages API keys using SHA-256 hashing. Keys are stored as hashes
//! so that the raw key material is never persisted in memory after creation.

const std = @import("std");
const time = @import("../services/shared/time.zig");
const Allocator = std.mem.Allocator;

pub const ApiKey = struct {
    name: []const u8,
    hash: [32]u8,
    created_at: i64,
    last_used: i64,
    request_count: u64,
};

pub const Auth = struct {
    const Self = @This();

    allocator: Allocator,
    keys: std.StringHashMapUnmanaged(ApiKey),
    enabled: bool,

    pub fn init(allocator: Allocator, enabled: bool) Self {
        return .{
            .allocator = allocator,
            .keys = .empty,
            .enabled = enabled,
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.keys.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.name);
        }
        self.keys.deinit(self.allocator);
    }

    /// Create a new API key. Returns the raw key (64 hex chars).
    /// The raw key is only available at creation time.
    pub fn createKey(self: *Self, name: []const u8) ![64]u8 {
        // Generate 32 random bytes → 64 hex chars.
        var raw_bytes: [32]u8 = undefined;
        std.c.arc4random_buf(&raw_bytes, raw_bytes.len);

        var key_hex: [64]u8 = undefined;
        _ = std.fmt.bufPrint(&key_hex, "{}", .{std.fmt.fmtSliceHexLower(&raw_bytes)}) catch unreachable;

        // Hash the key for storage.
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(&key_hex, &hash, .{});

        const name_owned = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_owned);

        try self.keys.put(self.allocator, name_owned, .{
            .name = name_owned,
            .hash = hash,
            .created_at = time.unixSeconds(),
            .last_used = 0,
            .request_count = 0,
        });

        return key_hex;
    }

    /// Validate an API key. Returns the key name if valid.
    pub fn validate(self: *Self, raw_key: []const u8) ?[]const u8 {
        if (!self.enabled) return "anonymous";

        // Hash the provided key.
        if (raw_key.len != 64) return null;
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(raw_key[0..64], &hash, .{});

        // Search for matching hash.
        var it = self.keys.iterator();
        while (it.next()) |entry| {
            if (std.mem.eql(u8, &entry.value_ptr.hash, &hash)) {
                entry.value_ptr.last_used = time.unixSeconds();
                entry.value_ptr.request_count += 1;
                return entry.value_ptr.name;
            }
        }
        return null;
    }

    /// Revoke a key by name.
    pub fn revokeKey(self: *Self, name: []const u8) bool {
        if (self.keys.fetchRemove(name)) |kv| {
            self.allocator.free(kv.value.name);
            return true;
        }
        return false;
    }

    pub fn keyCount(self: *const Self) usize {
        return self.keys.count();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "auth create and validate key" {
    const allocator = std.testing.allocator;

    var auth = Auth.init(allocator, true);
    defer auth.deinit();

    const raw_key = try auth.createKey("test-app");
    try std.testing.expectEqual(@as(usize, 1), auth.keyCount());

    // Validate with correct key.
    const name = auth.validate(&raw_key);
    try std.testing.expect(name != null);
    try std.testing.expectEqualStrings("test-app", name.?);

    // Validate with wrong key.
    var bad_key: [64]u8 = undefined;
    @memset(&bad_key, 'x');
    try std.testing.expect(auth.validate(&bad_key) == null);
}

test "auth disabled allows all" {
    const allocator = std.testing.allocator;

    var auth = Auth.init(allocator, false);
    defer auth.deinit();

    const name = auth.validate("anything");
    try std.testing.expect(name != null);
    try std.testing.expectEqualStrings("anonymous", name.?);
}

test "auth revoke key" {
    const allocator = std.testing.allocator;

    var auth = Auth.init(allocator, true);
    defer auth.deinit();

    _ = try auth.createKey("temp-key");
    try std.testing.expectEqual(@as(usize, 1), auth.keyCount());

    const revoked = auth.revokeKey("temp-key");
    try std.testing.expect(revoked);
    try std.testing.expectEqual(@as(usize, 0), auth.keyCount());
}
