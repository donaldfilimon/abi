//! Shared types for the DiskANN index implementation.

const std = @import("std");
const mmap = @import("../formats/mmap.zig");

/// DiskANN persistence file magic bytes
pub const DISKANN_MAGIC: [4]u8 = .{ 'D', 'A', 'N', 'N' };
/// DiskANN persistence format version
pub const DISKANN_FORMAT_VERSION: u32 = 1;
/// Header size (padded to sector alignment)
pub const DISKANN_HEADER_SIZE: usize = 32;

/// DiskANN configuration parameters
pub const DiskANNConfig = struct {
    /// Number of dimensions in vectors
    dimensions: u32 = 128,
    /// Maximum out-degree for Vamana graph
    max_degree: u32 = 64,
    /// Build-time search list size (L_build)
    build_list_size: u32 = 100,
    /// Query-time search list size (L_search)
    search_list_size: u32 = 100,
    /// Alpha parameter for Vamana pruning (>1.0)
    alpha: f32 = 1.2,
    /// Number of PQ subspaces
    pq_subspaces: u32 = 32,
    /// Bits per subspace code
    pq_bits: u32 = 8,
    /// Sector size for disk I/O alignment
    sector_size: u32 = 4096,
    /// Number of sectors per node (for prefetch)
    sectors_per_node: u32 = 1,
    /// Enable memory-mapped I/O
    use_mmap: bool = true,
    /// Cache size for graph nodes (number of nodes)
    node_cache_size: u32 = 100_000,
    /// Enable beam search with prefetching
    beam_search: bool = true,
    /// Beam width for search
    beam_width: u32 = 4,
};

/// Graph node stored on disk
pub const DiskNode = struct {
    /// Node ID
    id: u32,
    /// Number of neighbors
    num_neighbors: u32,
    /// Neighbor IDs (max_degree elements)
    neighbors: []u32,
    /// PQ codes for compressed vector
    pq_codes: []u8,
    /// Full vector (optional, for reranking)
    full_vector: ?[]f32 = null,

    pub fn getSectorAlignedSize(config: DiskANNConfig) usize {
        const base_size = @sizeOf(u32) * 2 + // id + num_neighbors
            config.max_degree * @sizeOf(u32) + // neighbors
            config.pq_subspaces; // pq_codes

        // Align to sector boundary
        return ((base_size + config.sector_size - 1) / config.sector_size) * config.sector_size;
    }
};

/// Search candidate for priority queue
pub const SearchCandidate = struct {
    id: u32,
    distance: f32,

    pub fn lessThan(_: void, a: SearchCandidate, b: SearchCandidate) std.math.Order {
        return std.math.order(a.distance, b.distance);
    }
};

/// Index statistics
pub const IndexStats = struct {
    num_vectors: u32 = 0,
    vectors_indexed: u32 = 0,
    queries_processed: u64 = 0,
    memory_bytes: u64 = 0,
    build_complete: bool = false,

    pub fn report(self: *const IndexStats) void {
        std.log.info("DiskANN Index Statistics:", .{});
        std.log.info("  Vectors: {d}", .{self.num_vectors});
        std.log.info("  Queries: {d}", .{self.queries_processed});
        std.log.info("  Memory: {d:.2} MB", .{@as(f64, @floatFromInt(self.memory_bytes)) / (1024 * 1024)});
    }
};

/// Persistence error set for save/load operations
pub const PersistError = error{
    NotBuilt,
    OpenFailed,
    WriteFailed,
    SeekFailed,
    InvalidMagic,
    UnsupportedVersion,
    CorruptedFile,
    DimensionMismatch,
} || mmap.MmapError || std.mem.Allocator.Error;

// Persistence helpers

/// Align a byte size up to the given sector boundary.
pub fn alignToSector(size: usize, sector: u32) usize {
    const s: usize = sector;
    return ((size + s - 1) / s) * s;
}

/// Write all bytes to a file descriptor, handling partial writes.
pub fn writeAllFd(fd: std.posix.fd_t, buf: []const u8) !void {
    var written: usize = 0;
    while (written < buf.len) {
        const n = std.posix.write(fd, buf[written..]) catch return error.WriteFailed;
        if (n == 0) return error.WriteFailed;
        written += n;
    }
}

/// Write zero-padding of the given length.
pub fn writePadding(fd: std.posix.fd_t, len: usize) !void {
    const zeros = [_]u8{0} ** 4096;
    var remaining = len;
    while (remaining > 0) {
        const chunk = @min(remaining, zeros.len);
        writeAllFd(fd, zeros[0..chunk]) catch return error.WriteFailed;
        remaining -= chunk;
    }
}

// Helper functions

pub const computeL2DistanceSquared = @import("../../../foundation/mod.zig").simd.distances.l2DistanceSquared;
