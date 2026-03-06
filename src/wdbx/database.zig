//! WDBX Distributed Vector Database
//!
//! Provides sharded vector storage with consistent hashing, MVCC versioning,
//! blockchain audit trail, and HNSW-backed ANN search.

const std = @import("std");
const Allocator = std.mem.Allocator;
const hnsw_mod = @import("hnsw.zig");
const simd = @import("simd.zig");

pub const Config = struct {
    num_shards: u32 = 16,
    virtual_nodes: u32 = 256,
    dimension: u32 = 768,
    max_versions: u32 = 10,
    enable_audit: bool = true,
    consistency: Consistency = .quorum,

    pub const Consistency = enum { one, quorum, all };
};

pub const Record = struct {
    id: []const u8,
    vector: []f32,
    metadata: ?[]const u8,
    version: u64,
    timestamp: i64,
    deleted: bool,
    ttl: ?i64,
};

pub const SearchResult = struct {
    id: []const u8,
    distance: f32,
    score: f32,
    metadata: ?[]const u8,
};

pub const Operation = struct {
    op_type: enum { insert, update, delete },
    record_id: []const u8,
    timestamp: i64,
};

pub const Block = struct {
    index: u64,
    timestamp: i64,
    prev_hash: [32]u8,
    hash: [32]u8,
    merkle_root: [32]u8,
    operations: std.ArrayList(Operation),

    pub fn computeHash(self: *Block) void {
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        // Hash block index + timestamp + prev_hash + merkle_root.
        const idx_bytes = std.mem.asBytes(&self.index);
        hasher.update(idx_bytes);
        const ts_bytes = std.mem.asBytes(&self.timestamp);
        hasher.update(ts_bytes);
        hasher.update(&self.prev_hash);
        hasher.update(&self.merkle_root);
        hasher.final(&self.hash);
    }

    pub fn verify(self: *const Block) bool {
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        const idx_bytes = std.mem.asBytes(&self.index);
        hasher.update(idx_bytes);
        const ts_bytes = std.mem.asBytes(&self.timestamp);
        hasher.update(ts_bytes);
        hasher.update(&self.prev_hash);
        hasher.update(&self.merkle_root);
        var computed: [32]u8 = undefined;
        hasher.final(&computed);
        return std.mem.eql(u8, &computed, &self.hash);
    }
};

/// A single shard holding an HNSW index and record metadata.
const Shard = struct {
    index: hnsw_mod.HnswIndex,
    records: std.StringHashMap(Record),
    id_to_node: std.StringHashMap(u64),
    node_to_id: std.AutoHashMap(u64, []const u8),
    next_node_id: u64,

    fn init(allocator: Allocator, dimension: u32) Shard {
        return .{
            .index = hnsw_mod.HnswIndex.init(allocator, .{ .dimension = dimension }),
            .records = std.StringHashMap(Record).init(allocator),
            .id_to_node = std.StringHashMap(u64).init(allocator),
            .node_to_id = std.AutoHashMap(u64, []const u8).init(allocator),
            .next_node_id = 0,
        };
    }

    fn deinit(self: *Shard, allocator: Allocator) void {
        // Free record data.
        var it = self.records.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.value_ptr.id);
            allocator.free(entry.value_ptr.vector);
            if (entry.value_ptr.metadata) |m| allocator.free(m);
        }
        self.records.deinit();
        self.id_to_node.deinit();
        self.node_to_id.deinit();
        self.index.deinit();
    }
};

/// Consistent hash ring for shard assignment.
const HashRing = struct {
    ring: std.ArrayList(VNode),

    const VNode = struct {
        hash: u64,
        shard: u32,

        fn lessThan(_: void, a: VNode, b: VNode) bool {
            return a.hash < b.hash;
        }
    };

    fn init(allocator: Allocator, num_shards: u32, virtual_nodes: u32) !HashRing {
        var ring = std.ArrayList(VNode).init(allocator);
        for (0..num_shards) |shard| {
            for (0..virtual_nodes) |vn| {
                var buf: [32]u8 = undefined;
                const key = std.fmt.bufPrint(&buf, "s{d}v{d}", .{ shard, vn }) catch &buf;
                const hash = std.hash.Wyhash.hash(0, key);
                try ring.append(.{ .hash = hash, .shard = @intCast(shard) });
            }
        }
        std.mem.sort(VNode, ring.items, {}, VNode.lessThan);
        return .{ .ring = ring };
    }

    fn deinit(self: *HashRing) void {
        self.ring.deinit();
    }

    fn getShard(self: *const HashRing, key: []const u8) u32 {
        const hash = std.hash.Wyhash.hash(0, key);
        // Binary search for first vnode >= hash.
        var lo: usize = 0;
        var hi: usize = self.ring.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.ring.items[mid].hash < hash) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo >= self.ring.items.len) lo = 0;
        return self.ring.items[lo].shard;
    }
};

pub const Database = struct {
    const Self = @This();

    allocator: Allocator,
    config: Config,
    shards: []Shard,
    hash_ring: HashRing,
    blocks: std.ArrayList(Block),
    block_count: u64,
    total_records: u64,

    pub fn init(allocator: Allocator, config: Config) !Self {
        const shards = try allocator.alloc(Shard, config.num_shards);
        for (shards) |*s| s.* = Shard.init(allocator, config.dimension);

        const ring = try HashRing.init(allocator, config.num_shards, config.virtual_nodes);

        return .{
            .allocator = allocator,
            .config = config,
            .shards = shards,
            .hash_ring = ring,
            .blocks = std.ArrayList(Block).init(allocator),
            .block_count = 0,
            .total_records = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.shards) |*s| s.deinit(self.allocator);
        self.allocator.free(self.shards);
        self.hash_ring.deinit();
        for (self.blocks.items) |*b| b.operations.deinit();
        self.blocks.deinit();
    }

    pub fn insert(self: *Self, id: []const u8, vector: []const f32, metadata: ?[]const u8) !void {
        const shard_idx = self.hash_ring.getShard(id);
        const shard = &self.shards[shard_idx];

        // Clone data for ownership.
        const id_owned = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(id_owned);
        const vec_owned = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(vec_owned);
        const meta_owned = if (metadata) |m| try self.allocator.dupe(u8, m) else null;
        errdefer if (meta_owned) |m| self.allocator.free(m);

        const now = std.time.timestamp();

        const record = Record{
            .id = id_owned,
            .vector = vec_owned,
            .metadata = meta_owned,
            .version = 1,
            .timestamp = now,
            .deleted = false,
            .ttl = null,
        };

        const node_id = shard.next_node_id;
        shard.next_node_id += 1;

        try shard.index.insert(node_id, vector);
        try shard.records.put(id_owned, record);
        try shard.id_to_node.put(id_owned, node_id);
        try shard.node_to_id.put(node_id, id_owned);

        self.total_records += 1;

        // Audit trail.
        if (self.config.enable_audit) {
            try self.appendOperation(.{
                .op_type = .insert,
                .record_id = id_owned,
                .timestamp = now,
            });
        }
    }

    pub fn search(self: *Self, query: []const f32, k: u32) ![]SearchResult {
        var all_results = std.ArrayList(SearchResult).init(self.allocator);
        defer all_results.deinit();

        // Search each shard and merge.
        for (self.shards) |*shard| {
            if (shard.index.len() == 0) continue;
            const hnsw_results = try shard.index.search(query, k);
            defer self.allocator.free(hnsw_results);

            for (hnsw_results) |hr| {
                const id = shard.node_to_id.get(hr.id) orelse continue;
                const rec = shard.records.get(id) orelse continue;
                if (rec.deleted) continue;

                try all_results.append(.{
                    .id = id,
                    .distance = hr.distance,
                    .score = 1.0 / (1.0 + hr.distance),
                    .metadata = rec.metadata,
                });
            }
        }

        // Sort by distance ascending.
        std.mem.sort(SearchResult, all_results.items, {}, struct {
            fn lt(_: void, a: SearchResult, b: SearchResult) bool {
                return a.distance < b.distance;
            }
        }.lt);

        const count = @min(k, @as(u32, @intCast(all_results.items.len)));
        return try self.allocator.dupe(SearchResult, all_results.items[0..count]);
    }

    pub fn get(self: *Self, id: []const u8) ?*const Record {
        const shard_idx = self.hash_ring.getShard(id);
        const shard = &self.shards[shard_idx];
        return shard.records.getPtr(id);
    }

    pub fn delete(self: *Self, id: []const u8) !bool {
        const shard_idx = self.hash_ring.getShard(id);
        const shard = &self.shards[shard_idx];

        if (shard.records.getPtr(id)) |rec| {
            rec.deleted = true;
            rec.version += 1;
            self.total_records -|= 1;

            if (self.config.enable_audit) {
                try self.appendOperation(.{
                    .op_type = .delete,
                    .record_id = rec.id,
                    .timestamp = std.time.timestamp(),
                });
            }
            return true;
        }
        return false;
    }

    pub fn recordCount(self: *const Self) u64 {
        return self.total_records;
    }

    pub fn blockCount(self: *const Self) u64 {
        return self.block_count;
    }

    // ── Audit / Blockchain ───────────────────────────────────────────

    fn appendOperation(self: *Self, op: Operation) !void {
        // Create a new block for each operation (simplified; production would batch).
        var block = Block{
            .index = self.block_count,
            .timestamp = std.time.timestamp(),
            .prev_hash = if (self.blocks.items.len > 0)
                self.blocks.items[self.blocks.items.len - 1].hash
            else
                std.mem.zeroes([32]u8),
            .hash = undefined,
            .merkle_root = std.mem.zeroes([32]u8),
            .operations = std.ArrayList(Operation).init(self.allocator),
        };
        try block.operations.append(op);
        block.computeHash();
        try self.blocks.append(block);
        self.block_count += 1;
    }

    /// Verify the integrity of the entire blockchain.
    pub fn verifyChain(self: *const Self) bool {
        for (self.blocks.items, 0..) |*block, i| {
            if (!block.verify()) return false;
            if (i > 0) {
                if (!std.mem.eql(u8, &block.prev_hash, &self.blocks.items[i - 1].hash)) {
                    return false;
                }
            }
        }
        return true;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "database insert and search" {
    const allocator = std.testing.allocator;

    var db = try Database.init(allocator, .{
        .num_shards = 2,
        .virtual_nodes = 4,
        .dimension = 4,
        .enable_audit = true,
    });
    defer db.deinit();

    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const v2 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    try db.insert("doc-1", &v1, "first document");
    try db.insert("doc-2", &v2, "second document");

    try std.testing.expectEqual(@as(u64, 2), db.recordCount());
    try std.testing.expect(db.blockCount() > 0);

    // Search.
    const query = [_]f32{ 0.9, 0.1, 0.0, 0.0 };
    const results = try db.search(&query, 2);
    defer allocator.free(results);
    try std.testing.expect(results.len > 0);
}

test "database delete" {
    const allocator = std.testing.allocator;

    var db = try Database.init(allocator, .{
        .num_shards = 1,
        .virtual_nodes = 1,
        .dimension = 4,
        .enable_audit = false,
    });
    defer db.deinit();

    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    try db.insert("doc-1", &v1, null);
    try std.testing.expectEqual(@as(u64, 1), db.recordCount());

    const deleted = try db.delete("doc-1");
    try std.testing.expect(deleted);
    try std.testing.expectEqual(@as(u64, 0), db.recordCount());
}

test "blockchain verification" {
    const allocator = std.testing.allocator;

    var db = try Database.init(allocator, .{
        .num_shards = 1,
        .virtual_nodes = 1,
        .dimension = 4,
        .enable_audit = true,
    });
    defer db.deinit();

    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    try db.insert("a", &v1, null);
    try db.insert("b", &v1, null);

    try std.testing.expect(db.verifyChain());
}
