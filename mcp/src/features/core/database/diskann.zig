//! DiskANN: Disk-based Approximate Nearest Neighbor Search
//!
//! Implements a graph-based ANN index optimized for disk I/O, enabling
//! billion-scale vector search with memory efficiency. Based on the
//! Microsoft Research DiskANN paper.
//!
//! Key features:
//! - Graph-based index stored on disk with efficient I/O patterns
//! - Vamana graph construction for high recall
//! - PQ (Product Quantization) compressed vectors for memory efficiency
//! - Beam search with disk prefetching
//!
//! Performance targets:
//! - Billion-scale datasets with <100GB memory
//! - Sub-millisecond query latency with SSD
//! - >95% recall@10 on standard benchmarks

// Re-export sub-modules
pub const types = @import("diskann/types.zig");
pub const codebook = @import("diskann/codebook.zig");
pub const index = @import("diskann/index.zig");
pub const vamana = @import("diskann/vamana.zig");

// Re-export primary types at top level for backward compatibility
pub const DiskANNConfig = types.DiskANNConfig;
pub const DiskNode = types.DiskNode;
pub const SearchCandidate = types.SearchCandidate;
pub const IndexStats = types.IndexStats;
pub const PersistError = types.PersistError;

pub const PQCodebook = codebook.PQCodebook;

pub const DiskANNIndex = index.DiskANNIndex;

pub const VamanaConfig = vamana.VamanaConfig;
pub const VamanaSearchResult = vamana.VamanaSearchResult;
pub const VamanaIndex = vamana.VamanaIndex;

// Re-export helpers for test access
pub const computeL2DistanceSquared = types.computeL2DistanceSquared;

// Constants
pub const DISKANN_MAGIC = types.DISKANN_MAGIC;
pub const DISKANN_FORMAT_VERSION = types.DISKANN_FORMAT_VERSION;
pub const DISKANN_HEADER_SIZE = types.DISKANN_HEADER_SIZE;

// Pull in sub-modules for test discovery
comptime {
    if (@import("builtin").is_test) {
        _ = types;
        _ = codebook;
        _ = index;
        _ = vamana;
    }
}
