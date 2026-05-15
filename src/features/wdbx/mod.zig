const std = @import("std");
const build_options = @import("build_options");
const gpu = if (build_options.feat_gpu) @import("../gpu/mod.zig") else @import("../gpu/stub.zig");

pub const MAX_LAYERS = 4;

pub const VectorRecord = struct {
    id: u32,
    values: []f32,
};

pub const SearchResult = struct {
    id: u32,
    score: f32,
};

pub const ConversationBlock = struct {
    id: [32]u8,
    prev_id: [32]u8,
    timestamp_ms: i64,
    profile: []const u8,
    query_id: u32,
    response_id: u32,
    metadata: []const u8,
};

pub const AccelerationStatus = struct {
    backend: gpu.Backend,
    mode: gpu.ExecutionMode,
    message: []const u8,
};

pub const Store = struct {
    allocator: std.mem.Allocator,
    entries: std.StringHashMapUnmanaged([]const u8) = .empty,
    vectors: std.ArrayListUnmanaged(VectorRecord) = .empty,
    blocks: std.ArrayListUnmanaged(ConversationBlock) = .empty,
    next_vector_id: u32 = 1,
    vector_dimensions: ?usize = null,
    acceleration: AccelerationStatus = defaultAcceleration(),

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

        for (self.vectors.items) |record| self.allocator.free(record.values);
        self.vectors.deinit(self.allocator);

        for (self.blocks.items) |block| {
            self.allocator.free(block.profile);
            self.allocator.free(block.metadata);
        }
        self.blocks.deinit(self.allocator);
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

    pub fn putVector(self: *Store, values: []const f32) !u32 {
        if (values.len == 0) return error.InvalidVector;
        if (self.vector_dimensions) |dims| {
            if (dims != values.len) return error.DimensionMismatch;
        } else {
            self.vector_dimensions = values.len;
        }

        const owned_values = try self.allocator.dupe(f32, values);
        errdefer self.allocator.free(owned_values);

        const id = self.next_vector_id;
        self.next_vector_id += 1;
        try self.vectors.append(self.allocator, .{ .id = id, .values = owned_values });
        self.acceleration = try runAccelerationKernel("wdbx.putVector", values.len);
        return id;
    }

    pub fn search(self: *Store, query: []const f32, limit: usize) ![]SearchResult {
        if (query.len == 0) return error.InvalidVector;
        if (self.vector_dimensions) |dims| if (dims != query.len) return error.DimensionMismatch;

        const result_count = @min(limit, self.vectors.items.len);
        const results = try self.allocator.alloc(SearchResult, result_count);
        errdefer self.allocator.free(results);

        var scratch = try self.allocator.alloc(SearchResult, self.vectors.items.len);
        defer self.allocator.free(scratch);

        const ops = gpu.vectorOps();
        for (self.vectors.items, 0..) |record, i| {
            scratch[i] = .{ .id = record.id, .score = try ops.cosineSimilarity(query, record.values) };
        }
        std.mem.sort(SearchResult, scratch, {}, greaterScore);
        @memcpy(results, scratch[0..result_count]);
        self.acceleration = try runAccelerationKernel("wdbx.search", query.len * self.vectors.items.len);
        return results;
    }

    pub fn appendBlock(self: *Store, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8) ![32]u8 {
        if (profile.len == 0) return error.InvalidProfile;
        const prev_id = if (self.blocks.items.len == 0) zeroId() else self.blocks.items[self.blocks.items.len - 1].id;
        var block = ConversationBlock{
            .id = undefined,
            .prev_id = prev_id,
            .timestamp_ms = @intCast(@divTrunc(std.time.nanoTimestamp(), std.time.ns_per_ms)),
            .profile = try self.allocator.dupe(u8, profile),
            .query_id = query_id,
            .response_id = response_id,
            .metadata = try self.allocator.dupe(u8, metadata),
        };
        errdefer self.allocator.free(block.profile);
        errdefer self.allocator.free(block.metadata);

        block.id = hashBlock(block);
        try self.blocks.append(self.allocator, block);
        return block.id;
    }

    pub fn blockCount(self: *const Store) usize {
        return self.blocks.items.len;
    }

    pub fn accelerationStatus(self: *const Store) AccelerationStatus {
        return self.acceleration;
    }
};

fn defaultAcceleration() AccelerationStatus {
    const status = gpu.detectBackend();
    return .{
        .backend = status.backend,
        .mode = if (status.accelerated) .native_gpu else .simulated_gpu,
        .message = status.message,
    };
}

fn runAccelerationKernel(name: []const u8, work_items: usize) !AccelerationStatus {
    const result = try gpu.executeKernel(.{ .name = name, .work_items = work_items });
    return .{ .backend = result.backend, .mode = result.mode, .message = result.message };
}

fn zeroId() [32]u8 {
    return [_]u8{0} ** 32;
}

fn greaterScore(_: void, lhs: SearchResult, rhs: SearchResult) bool {
    return lhs.score > rhs.score;
}

fn hashBlock(block: ConversationBlock) [32]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(&block.prev_id);
    var scalar_buf: [24]u8 = undefined;
    std.mem.writeInt(u64, scalar_buf[0..8], @intCast(block.timestamp_ms), .little);
    std.mem.writeInt(u32, scalar_buf[8..12], block.query_id, .little);
    std.mem.writeInt(u32, scalar_buf[12..16], block.response_id, .little);
    std.mem.writeInt(u64, scalar_buf[16..24], block.metadata.len, .little);
    hasher.update(&scalar_buf);
    hasher.update(block.profile);
    hasher.update(block.metadata);
    var out: [32]u8 = undefined;
    hasher.final(&out);
    return out;
}

test "Store owns and replaces entries" {
    var store_obj = Store.init(std.testing.allocator);
    defer store_obj.deinit();

    try store_obj.store("agent:abbey", "queued");
    try store_obj.store("agent:abbey", "trained");

    try std.testing.expectEqual(@as(usize, 1), store_obj.count());
    try std.testing.expectEqualStrings("trained", store_obj.get("agent:abbey") orelse return error.MissingEntry);
}

test "Store accelerates vector search and block chain memory" {
    var store_obj = Store.init(std.testing.allocator);
    defer store_obj.deinit();

    const q = try store_obj.putVector(&.{ 1, 0, 0, 0 });
    const r = try store_obj.putVector(&.{ 0.9, 0.1, 0, 0 });
    _ = try store_obj.putVector(&.{ 0, 1, 0, 0 });

    const results = try store_obj.search(&.{ 1, 0, 0, 0 }, 2);
    defer std.testing.allocator.free(results);
    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(q, results[0].id);
    try std.testing.expect(results[0].score >= results[1].score);

    _ = try store_obj.appendBlock("abbey", q, r, "accelerated=true");
    try std.testing.expectEqual(@as(usize, 1), store_obj.blockCount());
}
