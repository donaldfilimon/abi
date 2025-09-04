const std = @import("std");
const agent = @import("../../agent.zig");

pub const Config = struct {
    shard_count: u32 = 3,
};

const prime_numbers = [_]u64{31, 37, 43, 47, 53, 59, 61, 67, 71, 73};

pub fn primeHash(data: []const u8, seed: u64) u64 {
    var hash: u64 = seed | 1;
    for (data) |b| {
        hash ^= @intCast(u64, b);
        hash *= 0x9e3779b97f4a7c15;
        hash = (hash << 7) | (hash >> 57);
    }
    return hash;
}

pub fn calculateShard(key: []const u8, prime: u64) usize {
    return @intCast(usize, primeHash(key, 0) % prime);
}

pub const Entry = struct {
    key: []u8,
    value: []u8,
    persona: agent.PersonaType,
    version: u64,
};

pub const Shard = struct {
    id: u64,
    entries: std.ArrayList(Entry),

    pub fn init(alloc: std.mem.Allocator, id: u64) !Shard {
        return Shard{ .id = id, .entries = std.ArrayList(Entry).init(alloc) };
    }

    pub fn store(self: *Shard, key: []const u8, value: []const u8, p: agent.PersonaType) !void {
        try self.entries.append(.{
            .key = try self.entries.allocator.dupe(u8, key),
            .value = try self.entries.allocator.dupe(u8, value),
            .persona = p,
            .version = @as(u64, @intCast(std.time.nanoTimestamp())),
        });
    }

    pub fn retrieve(self: *Shard, key: []const u8) ?Entry {
        var i: usize = self.entries.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.entries.items[i].key, key)) {
                return self.entries.items[i];
            }
        }
        return null;
    }
};

pub const Database = struct {
    allocator: std.mem.Allocator,
    shards: []Shard,
    prime: u64,

    pub fn init(alloc: std.mem.Allocator, cfg: Config) !Database {
        const count = cfg.shard_count;
        if (count == 0 or count > prime_numbers.len) return error.InvalidShardCount;
        var shards = try alloc.alloc(Shard, count);
        for (shards, 0..) |*s, i| {
            s.* = try Shard.init(alloc, prime_numbers[i]);
        }
        return Database{ .allocator = alloc, .shards = shards, .prime = prime_numbers[count - 1] };
    }

    pub fn deinit(self: *Database) void {
        for (self.shards) |*s| {
            for (s.entries.items) |e| {
                self.allocator.free(e.key);
                self.allocator.free(e.value);
            }
            s.entries.deinit();
        }
        self.allocator.free(self.shards);
    }

    fn shardIndex(self: *Database, key: []const u8) usize {
        return calculateShard(key, self.prime) % self.shards.len;
    }

    pub fn storeInteraction(self: *Database, req: []const u8, resp: []const u8, p: agent.PersonaType) !void {
        const idx = self.shardIndex(req);
        try self.shards[idx].store(req, resp, p);
    }

    pub fn retrieve(self: *Database, key: []const u8) ?Entry {
        const idx = self.shardIndex(key);
        return self.shards[idx].retrieve(key);
    }
};

pub const WDBXError = error{InvalidShardCount};

pub test "basic store and retrieve" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var db = try Database.init(gpa.allocator(), .{ .shard_count = 2 });
    defer db.deinit();
    try db.storeInteraction("hello", "world", .EmpatheticAnalyst);
    const e = db.retrieve("hello") orelse return error.NotFound;
    try std.testing.expectEqualStrings(e.value, "world");
}
