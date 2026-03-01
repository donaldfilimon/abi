//! WDBX - Neural Vector Search Platform
//!
//! High-performance vector similarity search with HNSW indexing,
//! SIMD acceleration, and dynamic AI client integration.
//!
//! Quick Start:
//! ```zig
//! const wdbx = @import("wdbx");
//! var engine = try wdbx.Engine.init(allocator, .{});
//! defer engine.deinit();
//! ```

const std = @import("std");

pub const Config = @import("config.zig").Config;
pub const Engine = @import("engine.zig").Engine;
pub const SearchOptions = @import("engine.zig").SearchOptions;
pub const SearchResult = @import("engine.zig").SearchResult;
pub const Metadata = @import("engine.zig").Metadata;

pub const DistanceMetric = @import("config.zig").DistanceMetric;
pub const SIMD = @import("simd.zig").Features;
pub const Distance = @import("distance.zig").Distance;
pub const AIClient = @import("ai_client.zig").AIClient;

test {
    std.testing.refAllDecls(@This());

    // Assure all implementations compile inside nested tests natively
    _ = @import("simd.zig");
    _ = @import("distance.zig");
    _ = @import("config.zig");
    _ = @import("cache.zig");
    _ = @import("hnsw.zig");
    _ = @import("ai_client.zig");
    _ = @import("engine.zig");
}
