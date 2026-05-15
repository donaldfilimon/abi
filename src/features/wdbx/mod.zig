const std = @import("std");

pub const Store = struct {
    allocator: std.mem.Allocator,
    entries: std.StringHashMapUnmanaged([]const u8) = .empty,

    pub fn init(a: std.mem.Allocator) Store {
        return .{ .allocator = a };
    }

    pub fn deinit(self: *Store) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.entries.deinit(self.allocator);
    }

    pub fn store(self: *Store, key: []const u8, val: []const u8) !void {
        if (key.len == 0) return error.InvalidKey;

        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);

        const owned_val = try self.allocator.dupe(u8, val);
        errdefer self.allocator.free(owned_val);

        const result = try self.entries.getOrPut(self.allocator, owned_key);
        if (result.found_existing) {
            self.allocator.free(owned_key);
            self.allocator.free(result.key_ptr.*);
            self.allocator.free(result.value_ptr.*);
        }
        result.key_ptr.* = owned_key;
        result.value_ptr.* = owned_val;
    }

    pub fn get(self: *const Store, key: []const u8) ?[]const u8 {
        return self.entries.get(key);
    }

    pub fn count(self: *const Store) usize {
        return self.entries.count();
    }
};

test "Store owns and replaces entries" {
    var store_obj = Store.init(std.testing.allocator);
    defer store_obj.deinit();

    try store_obj.store("agent:abbey", "queued");
    try store_obj.store("agent:abbey", "trained");

    try std.testing.expectEqual(@as(usize, 1), store_obj.count());
    try std.testing.expectEqualStrings("trained", store_obj.get("agent:abbey") orelse return error.MissingEntry);
}
