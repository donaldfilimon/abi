const std = @import("std");
const foundation_pool = @import("../../foundation/pool_allocator.zig");
const foundation_time = @import("../../foundation/time.zig");
const memory = @import("../../core/memory.zig");
const runtime = @import("runtime.zig");
const types = @import("types.zig");

pub const index = @import("hnsw.zig");
pub const storage = @import("chain.zig");
pub const spatial_3d = @import("spatial_3d.zig");
pub const persistence = @import("persistence.zig");
pub const wal = @import("wal.zig");
pub const temporal = @import("temporal.zig");
pub const cluster = @import("cluster.zig");
pub const cluster_rpc = @import("cluster_rpc.zig");
pub const compression = @import("compression.zig");
pub const neural_compress = @import("neural_compress.zig");
pub const entropy = @import("entropy.zig");
pub const crypto_he = @import("crypto_he.zig");
pub const fhe = @import("fhe.zig");
pub const compute = @import("compute.zig");
pub const remote_compute = @import("remote_compute.zig");
pub const rest = @import("rest.zig");
pub const recovery = @import("recovery.zig");
pub const retrieval = @import("retrieval.zig");
pub const segments = @import("segments.zig");
pub const durable_store = @import("durable_store.zig");
pub const store_module = @import("store.zig");

pub const MAX_LAYERS = types.MAX_LAYERS;
pub const HNSW_DIMENSIONS = types.HNSW_DIMENSIONS;
pub const VECTOR_PADDED_BYTES = types.VECTOR_PADDED_BYTES;
pub const VectorRecord = types.VectorRecord;
pub const SearchResult = types.SearchResult;
pub const ConversationBlock = types.ConversationBlock;
pub const AccelerationStatus = types.AccelerationStatus;
pub const StoreStats = types.StoreStats;
pub const StoreConfig = types.StoreConfig;

pub const Store = store_module.Store;

test {
    _ = @import("hnsw.zig");
    _ = @import("chain.zig");
    _ = @import("wal.zig");
    _ = @import("temporal.zig");
    _ = @import("cluster.zig");
    _ = @import("cluster_rpc.zig");
    _ = @import("neural_compress.zig");
    _ = @import("fhe.zig");
    _ = @import("remote_compute.zig");
    _ = @import("compression.zig");
    _ = @import("entropy.zig");
    _ = @import("crypto_he.zig");
    _ = @import("compute.zig");
    _ = @import("rest.zig");
    _ = @import("runtime.zig");
    _ = @import("types.zig");
    _ = @import("store.zig");
    std.testing.refAllDecls(@This());
}
