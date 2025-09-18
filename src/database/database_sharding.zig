//! Simple sharded database wrapper
//! Provides round-robin insertion and fan-out search across shards.

const std = @import("std");
const database = @import("database");

pub const GlobalResult = struct {
    shard: usize,
    index: u64,
    score: f32,
};

pub const ShardedDb = struct {
    allocator: std.mem.Allocator,
    shards: [](*database.Db),
    dim: u16,
    next_shard: usize = 0,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, paths: []const []const u8, dim: u16) !*Self {
        if (paths.len == 0) return error.InvalidInput;

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        var shards = try allocator.alloc(*database.Db, paths.len);
        errdefer allocator.free(shards);

        var i: usize = 0;
        while (i < paths.len) : (i += 1) {
            var db = try database.Db.open(paths[i], true);
            errdefer db.close();
            try db.init(dim);
            shards[i] = db;
        }

        self.* = .{
            .allocator = allocator,
            .shards = shards,
            .dim = dim,
            .next_shard = 0,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.shards) |db| db.close();
        self.allocator.free(self.shards);
        self.allocator.destroy(self);
    }

    pub fn getDimension(self: *const Self) u16 {
        return self.dim;
    }

    pub fn getTotalRowCount(self: *const Self) u64 {
        var total: u64 = 0;
        for (self.shards) |db| total += db.getRowCount();
        return total;
    }

    pub fn addEmbedding(self: *Self, embedding: []const f32) !struct { shard: usize, local_index: u64 } {
        if (embedding.len != self.dim) return error.DimensionMismatch;
        const shard_index = self.next_shard % self.shards.len;
        self.next_shard = (self.next_shard + 1) % self.shards.len;
        const local = try self.shards[shard_index].addEmbedding(embedding);
        return .{ .shard = shard_index, .local_index = local };
    }

    pub fn search(self: *Self, query: []const f32, k: usize, allocator: std.mem.Allocator) ![]GlobalResult {
        if (query.len != self.dim) return error.DimensionMismatch;
        if (k == 0) return try allocator.alloc(GlobalResult, 0);

        var all = try std.ArrayList(GlobalResult).initCapacity(allocator, self.shards.len * k);
        defer all.deinit(allocator);

        for (self.shards, 0..) |db, shard_idx| {
            const results = db.search(query, k, allocator) catch |err| {
                // Skip shard on error
                _ = err;
                continue;
            };
            defer allocator.free(results);

            for (results) |r| {
                try all.append(allocator, .{ .shard = shard_idx, .index = r.index, .score = r.score });
            }
        }

        var items = all.items;
        std.sort.block(GlobalResult, items, {}, struct {
            fn lessThan(_: void, a: GlobalResult, b: GlobalResult) bool {
                return a.score < b.score; // smaller is better
            }
        }.lessThan);

        const out_len = @min(items.len, k);
        const out = try allocator.alloc(GlobalResult, out_len);
        @memcpy(out, items[0..out_len]);
        return out;
    }
};

const testing = std.testing;

test "sharded db basic insert and search" {
    const allocator = testing.allocator;

    const p0 = "shard0_test.wdbx";
    const p1 = "shard1_test.wdbx";
    defer std.fs.cwd().deleteFile(p0) catch {};
    defer std.fs.cwd().deleteFile(p1) catch {};
    defer std.fs.cwd().deleteFile("shard0_test.wdbx.wal") catch {};
    defer std.fs.cwd().deleteFile("shard1_test.wdbx.wal") catch {};

    const paths = [_][]const u8{ p0, p1 };
    const dim: u16 = 4;

    var sharded = try ShardedDb.init(allocator, &paths, dim);
    defer sharded.deinit();

    // Insert predictable vectors
    var v0 = [_]f32{ 1, 0, 0, 0 };
    var v1 = [_]f32{ 0, 1, 0, 0 };
    var v2 = [_]f32{ 0, 0, 1, 0 };
    _ = try sharded.addEmbedding(&v0);
    _ = try sharded.addEmbedding(&v1);
    _ = try sharded.addEmbedding(&v2);

    // Query similar to v0
    const q = v0;
    const results = try sharded.search(&q, 2, allocator);
    defer allocator.free(results);
    try testing.expect(results.len > 0);
}
