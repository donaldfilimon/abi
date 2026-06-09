const foundation_pool = @import("../../foundation/pool_allocator.zig");
const runtime = @import("runtime.zig");

pub const MAX_LAYERS = 4;
pub const HNSW_DIMENSIONS = 128;
pub const VECTOR_PADDED_BYTES = HNSW_DIMENSIONS * @sizeOf(f32);

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

pub const AccelerationStatus = runtime.AccelerationStatus;

pub const StoreStats = struct {
    kv_entries: usize,
    vectors: usize,
    blocks: usize,
    spatial_records: usize,
    vector_dimensions: ?usize,
    next_vector_id: u32,
    acceleration: AccelerationStatus,
};

pub const StoreConfig = struct {
    /// Optional fixed-block pool allocator used for hot-path padded vector
    /// buffers and small spatial payload copies. Block size must be at least
    /// `VECTOR_PADDED_BYTES`. Larger payloads fall back to the heap allocator.
    /// The pool is borrowed; the caller owns its lifecycle.
    pool_alloc: ?*foundation_pool.PoolAllocator = null,
};
