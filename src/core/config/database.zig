//! Database Configuration
//!
//! Configuration for vector database (WDBX) including storage,
//! index type, and caching settings.

const std = @import("std");

/// Vector database configuration.
pub const DatabaseConfig = struct {
    /// Database file path.
    path: []const u8 = "./abi.db",

    /// Index type for vector search.
    index_type: IndexType = .hnsw,

    /// Enable write-ahead logging.
    wal_enabled: bool = true,

    /// Cache size in bytes.
    cache_size: usize = 64 * 1024 * 1024, // 64MB

    /// Auto-optimize on startup.
    auto_optimize: bool = false,

    pub const IndexType = enum {
        hnsw,
        ivf_pq,
        flat,
    };

    pub fn defaults() DatabaseConfig {
        return .{};
    }

    /// In-memory database configuration.
    pub fn inMemory() DatabaseConfig {
        return .{
            .path = ":memory:",
            .wal_enabled = false,
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}
