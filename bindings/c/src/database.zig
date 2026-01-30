//! C-compatible database operation exports.
//! Provides vector database operations for FFI.

const std = @import("std");
const errors = @import("errors.zig");

/// Opaque database handle for C API.
pub const DatabaseHandle = opaque {};

/// Database config matching C header (abi_database_config_t).
pub const DatabaseConfig = extern struct {
    name: [*:0]const u8,
    dimension: usize,
    initial_capacity: usize,
};

/// Search result matching C header (abi_search_result_t).
pub const SearchResult = extern struct {
    id: u64,
    score: f32,
    vector: [*]const f32,
    vector_len: usize,
};

/// Internal database state.
const DatabaseState = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    dimension: usize,
    vectors: std.AutoHashMap(u64, []f32),

    pub fn init(allocator: std.mem.Allocator, config: DatabaseConfig) !*DatabaseState {
        const state = try allocator.create(DatabaseState);
        errdefer allocator.destroy(state);

        const name = std.mem.span(config.name);
        const name_copy = try allocator.dupe(u8, name);
        errdefer allocator.free(name_copy);

        state.* = .{
            .allocator = allocator,
            .name = name_copy,
            .dimension = config.dimension,
            .vectors = std.AutoHashMap(u64, []f32).init(allocator),
        };

        return state;
    }

    pub fn deinit(self: *DatabaseState) void {
        // Free all stored vectors
        var it = self.vectors.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.vectors.deinit();
        self.allocator.free(self.name);
        self.allocator.destroy(self);
    }

    pub fn insert(self: *DatabaseState, id: u64, vector: []const f32) !void {
        if (vector.len != self.dimension) {
            return error.InvalidArgument;
        }

        // Remove existing if present
        if (self.vectors.fetchRemove(id)) |kv| {
            self.allocator.free(kv.value);
        }

        const vector_copy = try self.allocator.dupe(f32, vector);
        try self.vectors.put(id, vector_copy);
    }

    pub fn delete(self: *DatabaseState, id: u64) bool {
        if (self.vectors.fetchRemove(id)) |kv| {
            self.allocator.free(kv.value);
            return true;
        }
        return false;
    }

    pub fn count(self: *const DatabaseState) usize {
        return self.vectors.count();
    }

    const ResultWithScore = struct {
        id: u64,
        score: f32,
        vector: []f32,
    };

    pub fn search(
        self: *const DatabaseState,
        query: []const f32,
        k: usize,
        out_results: []SearchResult,
    ) usize {
        if (query.len != self.dimension or k == 0) {
            return 0;
        }

        // Collect all scores using ArrayListUnmanaged (Zig 0.16)
        var scores = std.ArrayListUnmanaged(ResultWithScore).empty;
        defer scores.deinit(self.allocator);

        var it = self.vectors.iterator();
        while (it.next()) |entry| {
            const score = cosineSimilarity(query, entry.value_ptr.*);
            scores.append(self.allocator, .{
                .id = entry.key_ptr.*,
                .score = score,
                .vector = entry.value_ptr.*,
            }) catch continue;
        }

        // Sort by score descending
        std.mem.sort(ResultWithScore, scores.items, {}, struct {
            fn lessThan(_: void, a: ResultWithScore, b: ResultWithScore) bool {
                return a.score > b.score;
            }
        }.lessThan);

        // Copy top-k results
        const result_count = @min(k, scores.items.len);
        for (scores.items[0..result_count], out_results[0..result_count]) |item, *out| {
            out.* = .{
                .id = item.id,
                .score = item.score,
                .vector = item.vector.ptr,
                .vector_len = item.vector.len,
            };
        }

        return result_count;
    }
};

/// Simple cosine similarity for search.
fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len or a.len == 0) return 0.0;

    var dot: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;

    for (a, b) |va, vb| {
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    norm_a = @sqrt(norm_a);
    norm_b = @sqrt(norm_b);

    if (norm_a == 0.0 or norm_b == 0.0) return 0.0;
    return dot / (norm_a * norm_b);
}

// Global allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

/// Create database.
pub export fn abi_database_create(
    config: *const DatabaseConfig,
    out_db: *?*DatabaseHandle,
) errors.Error {
    const allocator = gpa.allocator();
    const state = DatabaseState.init(allocator, config.*) catch |err| {
        return errors.fromZigError(err);
    };
    out_db.* = @ptrCast(state);
    return errors.OK;
}

/// Close database.
pub export fn abi_database_close(handle: ?*DatabaseHandle) void {
    if (handle) |h| {
        const state: *DatabaseState = @ptrCast(@alignCast(h));
        state.deinit();
    }
}

/// Insert vector.
pub export fn abi_database_insert(
    handle: ?*DatabaseHandle,
    id: u64,
    vector: [*]const f32,
    vector_len: usize,
) errors.Error {
    const state: *DatabaseState = @ptrCast(@alignCast(handle orelse return errors.NOT_INITIALIZED));
    state.insert(id, vector[0..vector_len]) catch |err| {
        return errors.fromZigError(err);
    };
    return errors.OK;
}

/// Search vectors.
pub export fn abi_database_search(
    handle: ?*DatabaseHandle,
    query: [*]const f32,
    query_len: usize,
    k: usize,
    out_results: [*]SearchResult,
    out_count: *usize,
) errors.Error {
    const state: *const DatabaseState = @ptrCast(@alignCast(handle orelse return errors.NOT_INITIALIZED));
    out_count.* = state.search(query[0..query_len], k, out_results[0..k]);
    return errors.OK;
}

/// Delete vector.
pub export fn abi_database_delete(handle: ?*DatabaseHandle, id: u64) errors.Error {
    const state: *DatabaseState = @ptrCast(@alignCast(handle orelse return errors.NOT_INITIALIZED));
    _ = state.delete(id);
    return errors.OK;
}

/// Get count.
pub export fn abi_database_count(handle: ?*DatabaseHandle, out_count: *usize) errors.Error {
    const state: *const DatabaseState = @ptrCast(@alignCast(handle orelse return errors.NOT_INITIALIZED));
    out_count.* = state.count();
    return errors.OK;
}

test "database exports" {
    const config = DatabaseConfig{
        .name = "test",
        .dimension = 4,
        .initial_capacity = 100,
    };

    var db: ?*DatabaseHandle = null;
    try std.testing.expectEqual(errors.OK, abi_database_create(&config, &db));
    try std.testing.expect(db != null);

    // Insert vectors
    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const v2 = [_]f32{ 0.9, 0.1, 0.0, 0.0 };
    const v3 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };

    try std.testing.expectEqual(errors.OK, abi_database_insert(db, 1, &v1, 4));
    try std.testing.expectEqual(errors.OK, abi_database_insert(db, 2, &v2, 4));
    try std.testing.expectEqual(errors.OK, abi_database_insert(db, 3, &v3, 4));

    // Check count
    var count: usize = 0;
    try std.testing.expectEqual(errors.OK, abi_database_count(db, &count));
    try std.testing.expectEqual(@as(usize, 3), count);

    // Search
    var results: [3]SearchResult = undefined;
    var result_count: usize = 0;
    try std.testing.expectEqual(errors.OK, abi_database_search(db, &v1, 4, 3, &results, &result_count));
    try std.testing.expect(result_count > 0);
    // First result should be v1 (exact match)
    try std.testing.expectEqual(@as(u64, 1), results[0].id);

    // Delete
    try std.testing.expectEqual(errors.OK, abi_database_delete(db, 1));
    try std.testing.expectEqual(errors.OK, abi_database_count(db, &count));
    try std.testing.expectEqual(@as(usize, 2), count);

    // Close
    abi_database_close(db);
}
