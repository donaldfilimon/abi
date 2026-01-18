> **Codebase Status:** Synced with repository as of 2026-01-18.

//! # WDBX Vector Database
//!
//! High-performance vector database with HNSW indexing, hybrid search, and batch operations.
//!
//! ## Features
//!
//! - **HNSW Indexing**: Hierarchical Navigable Small World graphs for fast approximate nearest neighbor search
//! - **Hybrid Search**: Combined vector similarity + full-text search + metadata filtering
//! - **Batch Operations**: Efficient bulk insert, update, and delete operations
//! - **Full-Text Search**: Integrated text search capabilities
//! - **Filtering**: Advanced metadata filtering during search
//! - **Sharding**: Distributed sharding support for large datasets
//! - **HTTP API**: RESTful API for remote access
//!
//! ## Usage
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Create database
//! var db = try abi.wdbx.createDatabase(allocator, "vectors.db", .{
//!     .dimensions = 384,
//!     .distance_metric = .cosine,
//!     .ef_construction = 200,
//!     .max_connections = 16,
//! });
//! defer abi.wdbx.closeDatabase(db);
//!
//! // Insert vector
//! const vector = [_]f32{ 0.1, 0.2, 0.3, ... };
//! try abi.wdbx.insertVector(db, 1, &vector, .{ .label = "example" });
//!
//! // Search
//! const query = [_]f32{ 0.15, 0.25, 0.35, ... };
//! const results = try abi.wdbx.searchVectors(db, &query, 10);
//! defer allocator.free(results);
//! ```
//!
//! ## Sub-modules
//!
//! - `mod.zig` - Public API and database management
//! - `hnsw.zig` - HNSW index implementation
//! - `batch.zig` - Batch insert/update/delete operations
//! - `fulltext.zig` - Full-text search integration
//! - `filter.zig` - Metadata filtering
//! - `reindex.zig` - Background reindexing
//! - `shard.zig` - Distributed sharding
//! - `http.zig` - HTTP/REST API server
//!
//! ## Distance Metrics
//!
//! | Metric | Description | Use Case |
//! |--------|-------------|----------|
//! | `cosine` | Cosine similarity | Text embeddings, normalized vectors |
//! | `euclidean` | L2 distance | General purpose |
//! | `dot_product` | Inner product | When vectors are normalized |
//!
//! ## Backup & Restore
//!
//! > **Security**: Backup/restore paths are restricted to `backups/` directory.
//! > Paths cannot contain `..`, absolute paths, or Windows drive letters.
//!
//! ```zig
//! // Backup
//! try abi.wdbx.backup(db, "mybackup.db");
//!
//! // Restore
//! try abi.wdbx.restore(allocator, "mybackup.db", "vectors.db");
//! ```
//!
//! ## See Also
//!
//! - [Database Documentation](../../../docs/database.md)
//! - [API Reference](../../../API_REFERENCE.md)

