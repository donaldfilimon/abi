//! HNSW Persistence Helpers
//!
//! Binary format documentation for HNSW index serialization.
//!
//! Binary format (little-endian):
//! - u32: node count
//! - u32: m (max neighbors per node)
//! - u32: entry point node ID (0 if none)
//! - i32: max layer
//! - u32: ef_construction
//! - For each node:
//!   - u32: layer count
//!   - For each layer:
//!     - u32: neighbor count
//!     - u32[]: neighbor node IDs
//!
//! Note: SearchStatePool, DistanceCache, and GPU accelerator are not persisted.
//! Use enableGpuAcceleration() after loading to restore GPU support.

// Save/load methods are implemented directly on HnswIndex in mod.zig.
// This module provides format documentation and future persistence utilities.
