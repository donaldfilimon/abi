//! Database Diagnostics
//!
//! Comprehensive debugging and monitoring information for the
//! in-memory vector database. Provides memory usage, configuration
//! status, and health metrics.

const std = @import("std");
const database_mod = @import("database.zig");
const database_storage = @import("database_storage.zig");
const Database = database_mod.Database;
const VectorPool = database_storage.VectorPool;

/// Memory statistics for database storage
pub const MemoryStats = struct {
    /// Total bytes used by vector data
    vector_bytes: usize,
    /// Total bytes used by norm cache
    norm_cache_bytes: usize,
    /// Total bytes used by metadata
    metadata_bytes: usize,
    /// Total bytes used by index structures
    index_bytes: usize,
    /// Total memory footprint
    total_bytes: usize,
    /// Memory efficiency (data bytes / total bytes)
    efficiency: f32,
};

/// Performance statistics for database operations
pub const PerformanceStats = struct {
    /// Number of search operations performed
    search_count: u64,
    /// Number of insert operations performed
    insert_count: u64,
    /// Number of delete operations performed
    delete_count: u64,
    /// Number of update operations performed
    update_count: u64,
    /// Total vectors scanned across all searches
    vectors_scanned: u64,
    /// Average vectors per search
    avg_vectors_per_search: f32,
};

/// Configuration status for debugging
pub const ConfigStatus = struct {
    /// Whether norm caching is enabled
    norm_cache_enabled: bool,
    /// Whether vector pooling is enabled
    vector_pool_enabled: bool,
    /// Whether thread-safe mode is enabled
    thread_safe_enabled: bool,
    /// Initial capacity configured
    initial_capacity: usize,
};

/// Comprehensive diagnostics information for the database
pub const DiagnosticsInfo = struct {
    /// Database name
    name: []const u8,
    /// Number of vectors stored
    vector_count: usize,
    /// Vector dimension (0 if empty)
    dimension: usize,
    /// Memory usage statistics
    memory: MemoryStats,
    /// Configuration status
    config: ConfigStatus,
    /// Vector pool statistics (if enabled)
    pool_stats: ?VectorPool.PoolStats,
    /// Index health: ratio of index entries to records (should be 1.0)
    index_health: f32,
    /// Norm cache health: ratio of cached norms to records (should be 1.0 if enabled)
    norm_cache_health: f32,

    /// Collect diagnostics from a database instance
    pub fn collect(db: *Database) DiagnosticsInfo {
        const vector_count = db.records.items.len;
        const dimension = if (vector_count > 0) db.records.items[0].vector.len else 0;

        // Calculate memory usage
        const vector_bytes = vector_count * dimension * @sizeOf(f32);
        const norm_cache_bytes = db.cached_norms.items.len * @sizeOf(f32);

        var metadata_bytes: usize = 0;
        for (db.records.items) |record| {
            if (record.metadata) |meta| {
                metadata_bytes += meta.len;
            }
        }

        // Estimate index overhead (hash map internals)
        const index_bytes = db.id_index.capacity() * (@sizeOf(u64) + @sizeOf(usize) + @sizeOf(u32));

        const total_bytes = vector_bytes + norm_cache_bytes + metadata_bytes + index_bytes;
        const data_bytes = vector_bytes + metadata_bytes;
        const efficiency: f32 = if (total_bytes > 0)
            @as(f32, @floatFromInt(data_bytes)) / @as(f32, @floatFromInt(total_bytes))
        else
            1.0;

        // Calculate health metrics
        const index_health: f32 = if (vector_count > 0)
            @as(f32, @floatFromInt(db.id_index.count())) / @as(f32, @floatFromInt(vector_count))
        else
            1.0;

        const norm_cache_health: f32 = if (db.config.cache_norms and vector_count > 0)
            @as(f32, @floatFromInt(db.cached_norms.items.len)) / @as(f32, @floatFromInt(vector_count))
        else if (!db.config.cache_norms)
            1.0 // N/A, report as healthy
        else
            1.0;

        return .{
            .name = db.name,
            .vector_count = vector_count,
            .dimension = dimension,
            .memory = .{
                .vector_bytes = vector_bytes,
                .norm_cache_bytes = norm_cache_bytes,
                .metadata_bytes = metadata_bytes,
                .index_bytes = index_bytes,
                .total_bytes = total_bytes,
                .efficiency = efficiency,
            },
            .config = .{
                .norm_cache_enabled = db.config.cache_norms,
                .vector_pool_enabled = db.config.use_vector_pool,
                .thread_safe_enabled = db.config.thread_safe,
                .initial_capacity = db.config.initial_capacity,
            },
            .pool_stats = if (db.vector_pool) |pool| pool.getStats() else null,
            .index_health = index_health,
            .norm_cache_health = norm_cache_health,
        };
    }

    /// Format diagnostics to a writer
    pub fn format(self: DiagnosticsInfo, writer: anytype) !void {
        try writer.print("=== Database Diagnostics: {s} ===\n", .{self.name});
        try writer.print("Vectors: {d} (dimension: {d})\n", .{ self.vector_count, self.dimension });
        try writer.writeAll("\nMemory Usage:\n");
        try writer.print("  Vector data:  {d} bytes\n", .{self.memory.vector_bytes});
        try writer.print("  Norm cache:   {d} bytes\n", .{self.memory.norm_cache_bytes});
        try writer.print("  Metadata:     {d} bytes\n", .{self.memory.metadata_bytes});
        try writer.print("  Index:        {d} bytes\n", .{self.memory.index_bytes});
        try writer.print("  Total:        {d} bytes\n", .{self.memory.total_bytes});
        try writer.print("  Efficiency:   {d:.1}%\n", .{self.memory.efficiency * 100.0});

        try writer.writeAll("\nConfiguration:\n");
        try writer.print("  Norm cache:   {s}\n", .{if (self.config.norm_cache_enabled) "enabled" else "disabled"});
        try writer.print("  Vector pool:  {s}\n", .{if (self.config.vector_pool_enabled) "enabled" else "disabled"});
        try writer.print("  Thread-safe:  {s}\n", .{if (self.config.thread_safe_enabled) "enabled" else "disabled"});

        try writer.writeAll("\nHealth:\n");
        try writer.print("  Index health:      {d:.1}%\n", .{self.index_health * 100.0});
        try writer.print("  Norm cache health: {d:.1}%\n", .{self.norm_cache_health * 100.0});

        if (self.pool_stats) |ps| {
            try writer.writeAll("\nVector Pool:\n");
            try writer.print("  Allocations:  {d}\n", .{ps.alloc_count});
            try writer.print("  Frees:        {d}\n", .{ps.free_count});
            try writer.print("  Active:       {d}\n", .{ps.active_count});
            try writer.print("  Total bytes:  {d}\n", .{ps.total_bytes});
        }
    }

    /// Format diagnostics to an allocated string
    pub fn formatToString(self: DiagnosticsInfo, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayListUnmanaged(u8).empty;
        errdefer buf.deinit(allocator);

        var aw: std.Io.Writer.Allocating = .fromArrayList(allocator, &buf);
        try self.format(&aw.writer);

        return aw.toArrayList().toOwnedSlice(allocator);
    }

    /// Check if database is healthy (all health metrics >= 1.0)
    pub fn isHealthy(self: DiagnosticsInfo) bool {
        return self.index_health >= 0.99 and self.norm_cache_health >= 0.99;
    }
};
