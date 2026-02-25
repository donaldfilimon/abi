//! Shared Blackboard for Inter-Agent State
//!
//! A concurrent key-value store that agents read and write during workflow
//! execution. Enables the classic blackboard pattern where agents post partial
//! results and other agents react to changes.
//!
//! Features:
//! - **Thread-safe**: Protected by mutex for parallel agent execution
//! - **Versioned entries**: Monotonic version counter lets agents detect changes
//! - **Typed sections**: Namespace keys by agent or workflow step
//! - **History**: Optional change log for debugging and auditing
//! - **Snapshots**: Capture consistent point-in-time state

const std = @import("std");
const sync = @import("../../../services/shared/sync.zig");
const time = @import("../../../services/shared/time.zig");

// ============================================================================
// Types
// ============================================================================

/// A single entry on the blackboard.
pub const Entry = struct {
    /// The key (namespaced, e.g. "reviewer:findings").
    key: []const u8,
    /// The value (opaque bytes, typically UTF-8 text or JSON).
    value: []const u8,
    /// Who wrote this entry (agent ID or "system").
    author: []const u8,
    /// Monotonically increasing version for this key.
    version: u64,
    /// Timestamp when this entry was written (monotonic nanoseconds).
    timestamp_ns: u64,
};

/// A record of a change for the history log.
pub const ChangeRecord = struct {
    key: []const u8,
    old_version: u64,
    new_version: u64,
    author: []const u8,
    timestamp_ns: u64,
};

/// A point-in-time snapshot of the entire blackboard.
pub const Snapshot = struct {
    entries: []const Entry,
    taken_at_ns: u64,
    version: u64,

    pub fn deinit(self: Snapshot, allocator: std.mem.Allocator) void {
        for (self.entries) |entry| {
            allocator.free(entry.key);
            allocator.free(entry.value);
            allocator.free(entry.author);
        }
        allocator.free(self.entries);
    }
};

// ============================================================================
// Blackboard
// ============================================================================

/// Thread-safe shared blackboard for multi-agent coordination.
pub const Blackboard = struct {
    allocator: std.mem.Allocator,
    mutex: sync.Mutex,
    /// Current entries indexed by key.
    entries: std.StringHashMapUnmanaged(StoredEntry),
    /// Global version counter (incremented on every write).
    global_version: u64,
    /// Change history (bounded ring buffer).
    history: std.ArrayListUnmanaged(ChangeRecord),
    /// Maximum history entries to retain.
    max_history: usize,

    const StoredEntry = struct {
        value: []u8,
        author: []u8,
        version: u64,
        timestamp_ns: u64,
    };

    pub fn init(allocator: std.mem.Allocator, max_history: usize) Blackboard {
        return .{
            .allocator = allocator,
            .mutex = sync.Mutex{},
            .entries = .{},
            .global_version = 0,
            .history = .empty,
            .max_history = max_history,
        };
    }

    pub fn deinit(self: *Blackboard) void {
        var iter = self.entries.iterator();
        while (iter.next()) |kv| {
            self.allocator.free(kv.value_ptr.value);
            self.allocator.free(kv.value_ptr.author);
            self.allocator.free(kv.key_ptr.*);
        }
        self.entries.deinit(self.allocator);

        for (self.history.items) |rec| {
            self.allocator.free(rec.key);
            self.allocator.free(rec.author);
        }
        self.history.deinit(self.allocator);
    }

    /// Write a key-value pair to the blackboard.
    pub fn put(self: *Blackboard, key: []const u8, value: []const u8, author: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.global_version += 1;
        const timestamp = time.timestampNs();

        if (self.entries.getPtr(key)) |existing| {
            const old_version = existing.version;
            try self.recordChange(key, old_version, self.global_version, author, timestamp);

            self.allocator.free(existing.value);
            self.allocator.free(existing.author);
            existing.value = try self.allocator.dupe(u8, value);
            existing.author = try self.allocator.dupe(u8, author);
            existing.version = self.global_version;
            existing.timestamp_ns = timestamp;
        } else {
            const owned_key = try self.allocator.dupe(u8, key);
            errdefer self.allocator.free(owned_key);

            try self.recordChange(key, 0, self.global_version, author, timestamp);

            try self.entries.put(self.allocator, owned_key, .{
                .value = try self.allocator.dupe(u8, value),
                .author = try self.allocator.dupe(u8, author),
                .version = self.global_version,
                .timestamp_ns = timestamp,
            });
        }
    }

    /// Read a value from the blackboard. Returns null if key doesn't exist.
    pub fn get(self: *Blackboard, key: []const u8) ?Entry {
        self.mutex.lock();
        defer self.mutex.unlock();

        const stored = self.entries.get(key) orelse return null;
        return Entry{
            .key = key,
            .value = stored.value,
            .author = stored.author,
            .version = stored.version,
            .timestamp_ns = stored.timestamp_ns,
        };
    }

    /// Read a value only if it's newer than the given version.
    pub fn getIfNewer(self: *Blackboard, key: []const u8, since_version: u64) ?Entry {
        self.mutex.lock();
        defer self.mutex.unlock();

        const stored = self.entries.get(key) orelse return null;
        if (stored.version <= since_version) return null;
        return Entry{
            .key = key,
            .value = stored.value,
            .author = stored.author,
            .version = stored.version,
            .timestamp_ns = stored.timestamp_ns,
        };
    }

    /// Remove a key from the blackboard.
    pub fn remove(self: *Blackboard, key: []const u8, author: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.entries.fetchRemove(key)) |kv| {
            self.global_version += 1;
            const timestamp = time.timestampNs();
            try self.recordChange(key, kv.value.version, self.global_version, author, timestamp);

            self.allocator.free(kv.value.value);
            self.allocator.free(kv.value.author);
            self.allocator.free(kv.key);
        }
    }

    /// Get all keys currently on the blackboard.
    pub fn keys(self: *Blackboard, allocator: std.mem.Allocator) ![]const []const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var result: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer result.deinit(allocator);

        var iter = self.entries.iterator();
        while (iter.next()) |kv| {
            try result.append(allocator, try allocator.dupe(u8, kv.key_ptr.*));
        }

        return result.toOwnedSlice(allocator);
    }

    /// Get all keys written by a specific author.
    pub fn keysByAuthor(self: *Blackboard, allocator: std.mem.Allocator, author: []const u8) ![]const []const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var result: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer result.deinit(allocator);

        var iter = self.entries.iterator();
        while (iter.next()) |kv| {
            if (std.mem.eql(u8, kv.value_ptr.author, author)) {
                try result.append(allocator, try allocator.dupe(u8, kv.key_ptr.*));
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Get all entries that changed since a given global version.
    pub fn changesSince(self: *Blackboard, allocator: std.mem.Allocator, since_version: u64) ![]const Entry {
        self.mutex.lock();
        defer self.mutex.unlock();

        var result: std.ArrayListUnmanaged(Entry) = .empty;
        errdefer result.deinit(allocator);

        var iter = self.entries.iterator();
        while (iter.next()) |kv| {
            if (kv.value_ptr.version > since_version) {
                try result.append(allocator, .{
                    .key = kv.key_ptr.*,
                    .value = kv.value_ptr.value,
                    .author = kv.value_ptr.author,
                    .version = kv.value_ptr.version,
                    .timestamp_ns = kv.value_ptr.timestamp_ns,
                });
            }
        }

        return result.toOwnedSlice(allocator);
    }

    /// Take a snapshot of the current blackboard state.
    pub fn snapshot(self: *Blackboard, allocator: std.mem.Allocator) !Snapshot {
        self.mutex.lock();
        defer self.mutex.unlock();

        var entries_list: std.ArrayListUnmanaged(Entry) = .empty;
        errdefer {
            for (entries_list.items) |entry| {
                allocator.free(entry.key);
                allocator.free(entry.value);
                allocator.free(entry.author);
            }
            entries_list.deinit(allocator);
        }

        var iter = self.entries.iterator();
        while (iter.next()) |kv| {
            try entries_list.append(allocator, .{
                .key = try allocator.dupe(u8, kv.key_ptr.*),
                .value = try allocator.dupe(u8, kv.value_ptr.value),
                .author = try allocator.dupe(u8, kv.value_ptr.author),
                .version = kv.value_ptr.version,
                .timestamp_ns = kv.value_ptr.timestamp_ns,
            });
        }

        return Snapshot{
            .entries = try entries_list.toOwnedSlice(allocator),
            .taken_at_ns = time.timestampNs(),
            .version = self.global_version,
        };
    }

    /// Number of entries on the blackboard.
    pub fn count(self: *Blackboard) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.entries.count();
    }

    /// Current global version.
    pub fn currentVersion(self: *Blackboard) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.global_version;
    }

    fn recordChange(self: *Blackboard, key: []const u8, old_ver: u64, new_ver: u64, author: []const u8, timestamp: u64) !void {
        if (self.max_history == 0) return;

        if (self.history.items.len >= self.max_history) {
            const oldest = self.history.orderedRemove(0);
            self.allocator.free(oldest.key);
            self.allocator.free(oldest.author);
        }

        try self.history.append(self.allocator, .{
            .key = try self.allocator.dupe(u8, key),
            .old_version = old_ver,
            .new_version = new_ver,
            .author = try self.allocator.dupe(u8, author),
            .timestamp_ns = timestamp,
        });
    }
};

// ============================================================================
// Tests
// ============================================================================

test "blackboard put and get" {
    var bb = Blackboard.init(std.testing.allocator, 100);
    defer bb.deinit();

    try bb.put("task:input", "analyze this code", "system");
    const entry = bb.get("task:input");
    try std.testing.expect(entry != null);
    try std.testing.expectEqualStrings("analyze this code", entry.?.value);
    try std.testing.expectEqualStrings("system", entry.?.author);
    try std.testing.expectEqual(@as(u64, 1), entry.?.version);
}

test "blackboard versioned updates" {
    var bb = Blackboard.init(std.testing.allocator, 100);
    defer bb.deinit();

    try bb.put("findings", "none yet", "reviewer");
    try bb.put("findings", "found 3 bugs", "reviewer");

    const entry = bb.get("findings");
    try std.testing.expect(entry != null);
    try std.testing.expectEqualStrings("found 3 bugs", entry.?.value);
    try std.testing.expectEqual(@as(u64, 2), entry.?.version);
}

test "blackboard getIfNewer" {
    var bb = Blackboard.init(std.testing.allocator, 100);
    defer bb.deinit();

    try bb.put("status", "running", "agent-1");

    const entry = bb.getIfNewer("status", 0);
    try std.testing.expect(entry != null);

    const no_change = bb.getIfNewer("status", 1);
    try std.testing.expect(no_change == null);
}

test "blackboard remove" {
    var bb = Blackboard.init(std.testing.allocator, 100);
    defer bb.deinit();

    try bb.put("temp", "data", "agent-1");
    try std.testing.expectEqual(@as(usize, 1), bb.count());

    try bb.remove("temp", "agent-1");
    try std.testing.expectEqual(@as(usize, 0), bb.count());
    try std.testing.expect(bb.get("temp") == null);
}

test "blackboard keys and keysByAuthor" {
    var bb = Blackboard.init(std.testing.allocator, 100);
    defer bb.deinit();

    try bb.put("reviewer:bugs", "3 bugs", "reviewer");
    try bb.put("reviewer:style", "2 issues", "reviewer");
    try bb.put("tester:coverage", "85%", "tester");

    const all_keys = try bb.keys(std.testing.allocator);
    defer {
        for (all_keys) |k| std.testing.allocator.free(k);
        std.testing.allocator.free(all_keys);
    }
    try std.testing.expectEqual(@as(usize, 3), all_keys.len);

    const reviewer_keys = try bb.keysByAuthor(std.testing.allocator, "reviewer");
    defer {
        for (reviewer_keys) |k| std.testing.allocator.free(k);
        std.testing.allocator.free(reviewer_keys);
    }
    try std.testing.expectEqual(@as(usize, 2), reviewer_keys.len);
}

test "blackboard changesSince" {
    var bb = Blackboard.init(std.testing.allocator, 100);
    defer bb.deinit();

    try bb.put("a", "1", "agent");
    try bb.put("b", "2", "agent");
    const v = bb.currentVersion();
    try bb.put("c", "3", "agent");

    const changes = try bb.changesSince(std.testing.allocator, v);
    defer std.testing.allocator.free(changes);
    try std.testing.expectEqual(@as(usize, 1), changes.len);
    try std.testing.expectEqualStrings("c", changes[0].key);
}

test "blackboard history bounded" {
    var bb = Blackboard.init(std.testing.allocator, 3);
    defer bb.deinit();

    try bb.put("k", "v1", "a");
    try bb.put("k", "v2", "a");
    try bb.put("k", "v3", "a");
    try bb.put("k", "v4", "a");

    try std.testing.expectEqual(@as(usize, 3), bb.history.items.len);
}

test "blackboard snapshot" {
    var bb = Blackboard.init(std.testing.allocator, 100);
    defer bb.deinit();

    try bb.put("x", "1", "agent-1");
    try bb.put("y", "2", "agent-2");

    const snap = try bb.snapshot(std.testing.allocator);
    defer snap.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), snap.entries.len);
    try std.testing.expectEqual(@as(u64, 2), snap.version);
}

test {
    std.testing.refAllDecls(@This());
}
