const std = @import("std");

pub const DatabaseConfig = struct {
    path: []const u8 = "./abi.db",
    index_type: IndexType = .flat,
    wal_enabled: bool = false,
    cache_size: usize = 0,
    auto_optimize: bool = false,

    pub const IndexType = enum {
        hnsw,
        ivf_pq,
        flat,
    };

    pub fn defaults() DatabaseConfig {
        return .{};
    }

    pub fn inMemory() DatabaseConfig {
        return .{ .path = ":memory:" };
    }
};
