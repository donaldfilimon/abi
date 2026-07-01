const std = @import("std");
const build_options = @import("build_options");
const foundation_pool = @import("../../foundation/pool_allocator.zig");
const gpu = if (build_options.feat_gpu) @import("../gpu/mod.zig") else @import("../gpu/stub.zig");

pub const MAX_LAYERS = 4;
pub const HNSW_DIMENSIONS = 128;
pub const VECTOR_PADDED_BYTES = HNSW_DIMENSIONS * @sizeOf(f32);

pub const StoreConfig = struct {
    pool_alloc: ?*foundation_pool.PoolAllocator = null,
};

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

pub const StoreStats = struct {
    kv_entries: usize,
    vectors: usize,
    blocks: usize,
    spatial_records: usize,
    temporal_nodes: usize,
    temporal_edges: usize,
    vector_dimensions: ?usize,
    next_vector_id: u32,
    acceleration: AccelerationStatus,
};

test {
    std.testing.refAllDecls(@This());
}
