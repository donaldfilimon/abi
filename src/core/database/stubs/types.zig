const std = @import("std");

pub const DatabaseHandle = struct { db: ?*anyopaque = null };
pub const SearchResult = struct { id: u64 = 0, score: f32 = 0.0 };
pub const VectorView = struct { id: u64 = 0, vector: []const f32 = &.{}, metadata: ?[]const u8 = null };
pub const Stats = struct {
    count: usize = 0,
    dimension: usize = 0,
    memory_bytes: usize = 0,
    norm_cache_enabled: bool = false,
};
pub const DatabaseConfig = struct {
    cache_norms: bool = true,
    initial_capacity: usize = 0,
    use_vector_pool: bool = false,
    thread_safe: bool = false,
};
pub const BatchItem = struct { id: u64 = 0, vector: []const f32 = &.{}, metadata: ?[]const u8 = null };
pub const DiagnosticsInfo = struct {
    name: []const u8 = "",
    vector_count: usize = 0,
    dimension: usize = 0,
    pub fn isHealthy(_: DiagnosticsInfo) bool {
        return false;
    }
};

test {
    std.testing.refAllDecls(@This());
}
