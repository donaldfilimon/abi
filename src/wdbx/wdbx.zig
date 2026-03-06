//! WDBX — Neural Vector Search Platform
//!
//! High-performance approximate nearest-neighbour search backed by a real
//! multi-layer HNSW graph, SIMD-accelerated metrics, a segmented embedding
//! cache, an OpenAI-compatible HTTP client, and binary persistence.
//!
//! ## Quick Start
//! ```zig
//! const wdbx = @import("wdbx");
//!
//! var engine = try wdbx.Engine.init(allocator, .{});
//! defer engine.deinit();
//!
//! // Index with pre-computed embeddings
//! try engine.indexByVector("doc-1", &embedding, .{ .text = "doc" });
//!
//! // Or connect AI for dynamic embeddings
//! try engine.connectAI("https://api.openai.com", api_key);
//! try engine.index("doc-2", "The quick brown fox", .{ .text = "doc" });
//!
//! // Search
//! const results = try engine.search("fast animals", .{ .k = 5 });
//! defer allocator.free(results);
//!
//! // Persist to disk
//! try wdbx.save(&engine, "index.wdbx");
//! var loaded = try wdbx.load(allocator, "index.wdbx");
//! defer loaded.deinit();
//! ```

const std = @import("std");

// ─── Public re-exports ───────────────────────────────────────────────────────

pub const Config = @import("config.zig").Config;
pub const DistanceMetric = @import("config.zig").DistanceMetric;

pub const Engine = @import("engine.zig").Engine;
pub const EngineVector = @import("engine.zig").EngineVector;
pub const Metadata = @import("engine.zig").Metadata;
pub const SearchOptions = @import("engine.zig").SearchOptions;
pub const SearchResult = @import("engine.zig").SearchResult;

pub const HNSW = @import("hnsw.zig").HNSW;
pub const Cache = @import("cache.zig").Cache;
pub const AIClient = @import("ai_client.zig").AIClient;
pub const AIClientError = @import("ai_client.zig").AIClientError;

pub const Distance = @import("distance.zig").Distance;
pub const SIMDFeatures = @import("simd.zig").Features;
pub const dotProduct = @import("simd.zig").dotProduct;
pub const norm = @import("simd.zig").norm;
pub const normalize = @import("simd.zig").normalize;

// ─── Persistence ─────────────────────────────────────────────────────────────

const persistence = @import("persistence.zig");
pub const PersistenceError = persistence.PersistenceError;
pub const save = persistence.save;
pub const load = persistence.load;

// ─── Convenience factory ─────────────────────────────────────────────────────

/// Shorthand: create an engine with default config.
pub fn defaultEngine(allocator: std.mem.Allocator) !Engine {
    return Engine.init(allocator, .{});
}

// ─── Test discovery ──────────────────────────────────────────────────────────

test {
    std.testing.refAllDecls(@This());
    _ = @import("simd.zig");
    _ = @import("distance.zig");
    _ = @import("config.zig");
    _ = @import("cache.zig");
    _ = @import("hnsw.zig");
    _ = @import("ai_client.zig");
    _ = @import("engine.zig");
    _ = @import("sync_compat.zig");
    _ = @import("persistence.zig");
}
