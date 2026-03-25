//! Integration tests for database core modules: store operations, retrieval indexes,
//! storage format, distributed types, and error handling paths.
//!
//! Complements database_surface_test.zig (type defaults, boundary checks) and
//! database_test.zig (block chain lifecycle, MVCC, routing weights) by covering
//! areas those files do not: Store CRUD, search, diagnostics, retrieval index
//! type availability, storage format constants, and stub error paths.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const db = abi.database;

// ---------------------------------------------------------------------------
// 1. Retrieval: HNSW, DiskANN, ScaNN, clustering, quantization
// ---------------------------------------------------------------------------

test "database core: retrieval.hnsw namespace is available" {
    comptime {
        _ = db.retrieval.hnsw;
    }
}

test "database core: retrieval.diskann.VamanaIndex type is available" {
    comptime {
        _ = db.retrieval.diskann;
        _ = db.retrieval.diskann.VamanaIndex;
    }
}

test "database core: retrieval.scann.ScaNNIndex type is available" {
    comptime {
        _ = db.retrieval.scann;
        _ = db.retrieval.scann.ScaNNIndex;
    }
}

test "database core: retrieval.KMeans type is available" {
    comptime {
        _ = db.retrieval.KMeans;
    }
}

test "database core: retrieval.clustering helpers compile" {
    comptime {
        _ = db.retrieval.clustering.euclideanDistance;
        _ = db.retrieval.clustering.cosineSimilarity;
    }
}

test "database core: retrieval.quantization types compile" {
    comptime {
        _ = db.retrieval.quantization.ScalarQuantizer;
        _ = db.retrieval.quantization.ProductQuantizer;
    }
}

// ---------------------------------------------------------------------------
// 2. Storage: format, compression
// ---------------------------------------------------------------------------

test "database core: storage format constants" {
    const magic = db.storage.MAGIC;
    try std.testing.expectEqualSlices(u8, &.{ 'W', 'D', 'B', 'X' }, &magic);
    try std.testing.expect(db.storage.FORMAT_VERSION >= 3);
}

test "database core: storage.SectionType enum variants" {
    const ST = db.storage.SectionType;
    try std.testing.expectEqual(@as(u16, 1), @intFromEnum(ST.metadata));
    try std.testing.expectEqual(@as(u16, 2), @intFromEnum(ST.vectors));
    try std.testing.expectEqual(@as(u16, 3), @intFromEnum(ST.bloom_filter));
}

test "database core: storage.StorageConfig defaults" {
    const cfg: db.storage.StorageConfig = .{};
    try std.testing.expect(cfg.verify_checksums);
    try std.testing.expect(cfg.include_index);
}

test "database core: storage.compression namespace is available" {
    comptime {
        _ = db.storage.compression;
    }
}

// ---------------------------------------------------------------------------
// 3. Store lifecycle (feature-gated)
// ---------------------------------------------------------------------------

test "database core: Store.open returns error when disabled" {
    if (build_options.feat_database) return error.SkipZigTest;

    const result = db.Store.open(std.testing.allocator, "test-db");
    try std.testing.expectError(error.DatabaseDisabled, result);
}

test "database core: Store.open succeeds when enabled" {
    if (!build_options.feat_database) return error.SkipZigTest;

    var store = try db.Store.open(std.testing.allocator, "core-lifecycle-test");
    defer store.deinit();

    const s = store.stats();
    try std.testing.expectEqual(@as(usize, 0), s.count);
}

test "database core: Store insert and retrieve roundtrip" {
    if (!build_options.feat_database) return error.SkipZigTest;

    var store = try db.Store.open(std.testing.allocator, "core-roundtrip");
    defer store.deinit();

    const vec = [_]f32{ 1.0, 0.0, 0.5, 0.25 };
    try store.insert(42, &vec, "test-meta");

    const view = store.get(42);
    try std.testing.expect(view != null);
    if (view) |v| {
        try std.testing.expectEqual(@as(u64, 42), v.id);
    }
}

test "database core: Store remove returns false for missing id" {
    if (!build_options.feat_database) return error.SkipZigTest;

    var store = try db.Store.open(std.testing.allocator, "core-remove-miss");
    defer store.deinit();

    const removed = store.remove(999);
    try std.testing.expect(!removed);
}

test "database core: Store insert and remove roundtrip" {
    if (!build_options.feat_database) return error.SkipZigTest;

    var store = try db.Store.open(std.testing.allocator, "core-remove-hit");
    defer store.deinit();

    const vec = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    try store.insert(7, &vec, null);

    const removed = store.remove(7);
    try std.testing.expect(removed);

    try std.testing.expect(store.get(7) == null);
}

test "database core: Store search returns results" {
    if (!build_options.feat_database) return error.SkipZigTest;

    var store = try db.Store.open(std.testing.allocator, "core-search");
    defer store.deinit();

    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const v2 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };
    try store.insert(1, &v1, null);
    try store.insert(2, &v2, null);

    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const results = try store.search(&query, 2);
    defer store.allocator().free(results);

    try std.testing.expect(results.len > 0);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
}

test "database core: Store diagnostics on fresh store" {
    if (!build_options.feat_database) return error.SkipZigTest;

    var store = try db.Store.open(std.testing.allocator, "core-diag");
    defer store.deinit();

    const diag = store.diagnostics();
    try std.testing.expect(diag.isHealthy());
}

test "database core: Store searchInto with pre-allocated buffer" {
    if (!build_options.feat_database) return error.SkipZigTest;

    var store = try db.Store.open(std.testing.allocator, "core-search-into");
    defer store.deinit();

    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    try store.insert(10, &v1, null);

    var buf: [4]db.SearchResult = undefined;
    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const n = store.searchInto(&query, 4, &buf);
    try std.testing.expect(n >= 1);
    try std.testing.expectEqual(@as(u64, 10), buf[0].id);
}

// ---------------------------------------------------------------------------
// 4. Memory: BlockChain addBlock + getBlock retrieval
// ---------------------------------------------------------------------------

test "database core: BlockChain addBlock and retrieve by id" {
    var chain = db.memory.BlockChain.init(std.testing.allocator, "core-retrieve");
    defer chain.deinit();

    const embedding = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const block_id = try chain.addBlock(.{
        .query_embedding = &embedding,
        .profile_tag = .{ .primary_profile = .abbey, .blend_coefficient = 0.7 },
        .routing_weights = .{ .abbey_weight = 0.8, .aviva_weight = 0.1, .abi_weight = 0.1 },
        .intent = .empathy_seeking,
        .risk_score = 0.05,
    });

    // getBlock should return a non-null block with a valid timestamp
    if (chain.getBlock(block_id)) |block| {
        try std.testing.expect(block.commit_timestamp > 0);
    } else {
        return error.TestUnexpectedResult;
    }
}

test "database core: memory.MemoryBlock type is available" {
    comptime {
        _ = db.memory.MemoryBlock;
        _ = db.memory.InfluenceTrace;
        _ = db.memory.RetrievalHit;
        _ = db.memory.Lineage;
    }
}

// ---------------------------------------------------------------------------
// 5. Distributed types availability
// ---------------------------------------------------------------------------

test "database core: distributed types compile" {
    comptime {
        _ = db.distributed.ShardManager;
        _ = db.distributed.VersionVector;
    }
}

// ---------------------------------------------------------------------------
// 6. isEnabled flag consistency
// ---------------------------------------------------------------------------

test "database core: isEnabled matches build_options" {
    try std.testing.expectEqual(build_options.feat_database, db.isEnabled());
}

// ---------------------------------------------------------------------------
// 7. Store stub error paths (disabled builds)
// ---------------------------------------------------------------------------

test "database core: stub Store methods return expected errors" {
    if (build_options.feat_database) return error.SkipZigTest;

    var store: db.Store = undefined;
    try std.testing.expectError(error.DatabaseDisabled, store.insert(1, &.{ 0.0, 0.0 }, null));
    try std.testing.expectError(error.DatabaseDisabled, store.search(&.{0.0}, 1));
    try std.testing.expectError(error.DatabaseDisabled, store.optimize());
    try std.testing.expectError(error.DatabaseDisabled, store.save("path"));
    try std.testing.expectError(error.DatabaseDisabled, store.loadInto("path"));

    try std.testing.expect(!store.remove(1));
    try std.testing.expect(store.get(1) == null);
    const s = store.stats();
    try std.testing.expectEqual(@as(usize, 0), s.count);
}

test "database core: stub Context.init returns error" {
    if (build_options.feat_database) return error.SkipZigTest;

    const result = db.Context.init(std.testing.allocator, .{});
    try std.testing.expectError(error.DatabaseDisabled, result);
}

test {
    std.testing.refAllDecls(@This());
}
