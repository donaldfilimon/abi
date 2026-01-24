//! In-memory vector database with persistence helpers.
//!
//! Performance optimizations:
//! - VectorPool: Size-class based pooling for common dimensions
//! - CachedNorms: Pre-computed L2 norms for faster cosine similarity
//! - SIMD batch operations: Vectorized similarity computation
//! - Cache-aligned storage: 64-byte alignment for hot data

const std = @import("std");
const simd = @import("../shared/simd.zig");

pub const DatabaseError = error{
    DuplicateId,
    VectorNotFound,
    InvalidDimension,
    PoolExhausted,
    PersistenceError,
    ConcurrencyError,
};

// ============================================================================
// Diagnostics - Comprehensive debugging information
// ============================================================================

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

// ============================================================================
// Vector Pool - Size-class based memory pooling for vectors
// ============================================================================

/// Size classes for common vector dimensions.
/// Pools pre-allocate chunks for each size class to reduce allocation overhead.
// ============================================================================
// Cache-Aligned Structures for Hot Data
// ============================================================================

/// Cache line size for alignment (64 bytes on most modern CPUs)
pub const CACHE_LINE_SIZE = 64;

/// Hot data structure optimized for cache access patterns.
/// Separates frequently-accessed data (vectors, norms) from cold data (metadata).
pub const HotVectorData = struct {
    /// Contiguous vector storage - cache-aligned
    vectors: []align(CACHE_LINE_SIZE) f32,
    /// Parallel array of norms - cache-aligned
    norms: []align(CACHE_LINE_SIZE) f32,
    /// Number of vectors stored
    count: usize,
    /// Dimension of each vector
    dimension: usize,
    /// Total capacity
    capacity: usize,

    pub fn init(allocator: std.mem.Allocator, dimension: usize, capacity: usize) !HotVectorData {
        const vectors = try allocator.alignedAlloc(f32, CACHE_LINE_SIZE, capacity * dimension);
        const norms = try allocator.alignedAlloc(f32, CACHE_LINE_SIZE, capacity);
        return .{
            .vectors = vectors,
            .norms = norms,
            .count = 0,
            .dimension = dimension,
            .capacity = capacity,
        };
    }

    pub fn deinit(self: *HotVectorData, allocator: std.mem.Allocator) void {
        allocator.free(self.vectors);
        allocator.free(self.norms);
    }

    /// Get vector at index as a slice
    pub fn getVector(self: *const HotVectorData, index: usize) []const f32 {
        const start = index * self.dimension;
        return self.vectors[start..][0..self.dimension];
    }

    /// Get mutable vector at index
    pub fn getVectorMut(self: *HotVectorData, index: usize) []f32 {
        const start = index * self.dimension;
        return self.vectors[start..][0..self.dimension];
    }

    /// Get norm at index
    pub fn getNorm(self: *const HotVectorData, index: usize) f32 {
        return self.norms[index];
    }

    /// Add a vector and its norm
    pub fn append(self: *HotVectorData, vector: []const f32, norm: f32) !void {
        if (self.count >= self.capacity) return error.PoolExhausted;
        const dest = self.getVectorMut(self.count);
        @memcpy(dest, vector);
        self.norms[self.count] = norm;
        self.count += 1;
    }

    /// Prefetch vector at index for upcoming access
    pub fn prefetch(self: *const HotVectorData, index: usize) void {
        if (index < self.count) {
            const start = index * self.dimension;
            const ptr: [*]const f32 = @ptrCast(&self.vectors[start]);
            @prefetch(ptr, .{ .rw = .read, .locality = 3, .cache = .data });
        }
    }
};

/// Cold data structure for infrequently accessed data.
pub const ColdVectorData = struct {
    /// Vector IDs
    ids: std.ArrayListUnmanaged(u64),
    /// Metadata (optional per vector)
    metadata: std.ArrayListUnmanaged(?[]const u8),

    pub fn init() ColdVectorData {
        return .{
            .ids = .{},
            .metadata = .{},
        };
    }

    pub fn deinit(self: *ColdVectorData, allocator: std.mem.Allocator) void {
        for (self.metadata.items) |meta| {
            if (meta) |m| allocator.free(m);
        }
        self.ids.deinit(allocator);
        self.metadata.deinit(allocator);
    }

    pub fn append(self: *ColdVectorData, allocator: std.mem.Allocator, id: u64, metadata: ?[]const u8) !void {
        try self.ids.append(allocator, id);
        const meta_copy: ?[]const u8 = if (metadata) |m| try allocator.dupe(u8, m) else null;
        try self.metadata.append(allocator, meta_copy);
    }
};

// ============================================================================
// Vector Pool - Allocator wrapper for vector memory
// ============================================================================

/// Vector allocator wrapper.
/// Currently a thin wrapper around the allocator. Future versions may implement
/// size-class based pooling for common dimensions (128, 256, 384, 512, 768, 1024, 1536, 4096).
///
/// Usage is optional - Database works with or without VectorPool.
pub const VectorPool = struct {
    allocator: std.mem.Allocator,
    /// Statistics for monitoring
    alloc_count: usize,
    free_count: usize,
    total_bytes: usize,

    pub fn init(allocator: std.mem.Allocator) VectorPool {
        return .{
            .allocator = allocator,
            .alloc_count = 0,
            .free_count = 0,
            .total_bytes = 0,
        };
    }

    pub fn deinit(self: *VectorPool) void {
        // No cleanup needed - allocator owns all memory
        self.* = undefined;
    }

    /// Allocate a vector of the given dimension.
    pub fn alloc(self: *VectorPool, dimension: usize) ![]f32 {
        const vec = try self.allocator.alloc(f32, dimension);
        self.alloc_count += 1;
        self.total_bytes += dimension * @sizeOf(f32);
        return vec;
    }

    /// Free a vector.
    pub fn free(self: *VectorPool, vector: []f32) void {
        self.free_count += 1;
        self.total_bytes -|= vector.len * @sizeOf(f32);
        self.allocator.free(vector);
    }

    /// Get pool statistics.
    pub fn getStats(self: *const VectorPool) PoolStats {
        return .{
            .alloc_count = self.alloc_count,
            .free_count = self.free_count,
            .active_count = self.alloc_count -| self.free_count,
            .total_bytes = self.total_bytes,
        };
    }

    pub const PoolStats = struct {
        alloc_count: usize,
        free_count: usize,
        active_count: usize,
        total_bytes: usize,
    };
};

// ============================================================================
// Cached Vector Record - Includes pre-computed norm
// ============================================================================

/// Extended vector record with cached L2 norm for faster similarity computation.
pub const VectorRecordCached = struct {
    id: u64,
    vector: []f32,
    metadata: ?[]const u8,
    /// Pre-computed L2 norm for fast cosine similarity
    norm: f32,

    /// Compute and cache the L2 norm.
    pub fn computeNorm(self: *VectorRecordCached) void {
        self.norm = simd.vectorL2Norm(self.vector);
    }
};

pub const VectorRecord = struct {
    id: u64,
    vector: []f32,
    metadata: ?[]const u8,
};

pub const VectorView = struct {
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8,
};

pub const SearchResult = struct {
    id: u64,
    score: f32,
};

pub const Stats = struct {
    count: usize,
    dimension: usize,
    /// Memory used by vectors (approximate)
    memory_bytes: usize,
    /// Cache hit rate for norms (if applicable)
    norm_cache_enabled: bool,
};

/// Database configuration options.
pub const DatabaseConfig = struct {
    /// Enable cached L2 norms for faster similarity computation
    cache_norms: bool = true,
    /// Pre-allocate capacity for this many vectors
    initial_capacity: usize = 0,
    /// Use vector pool for common dimensions
    use_vector_pool: bool = false,
    /// Enable thread-safe RW locking for concurrent access
    thread_safe: bool = false,
};

pub const Database = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    records: std.ArrayListUnmanaged(VectorRecord),
    /// O(1) lookup index: id -> array index for fast findIndex operations
    id_index: std.AutoHashMapUnmanaged(u64, usize),
    /// Cached L2 norms for each vector (parallel array)
    cached_norms: std.ArrayListUnmanaged(f32),
    /// Configuration
    config: DatabaseConfig,
    /// Optional vector pool
    vector_pool: ?*VectorPool,
    /// Read-write lock for thread-safe concurrent access
    rw_lock: std.Thread.RwLock,

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !Database {
        return initWithConfig(allocator, name, .{});
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, name: []const u8, config: DatabaseConfig) !Database {
        var records = std.ArrayListUnmanaged(VectorRecord).empty;
        var id_index = std.AutoHashMapUnmanaged(u64, usize){};
        var cached_norms = std.ArrayListUnmanaged(f32).empty;

        if (config.initial_capacity > 0) {
            try records.ensureTotalCapacity(allocator, config.initial_capacity);
            try id_index.ensureTotalCapacity(allocator, @intCast(config.initial_capacity));
            if (config.cache_norms) {
                try cached_norms.ensureTotalCapacity(allocator, config.initial_capacity);
            }
        }

        var vector_pool: ?*VectorPool = null;
        if (config.use_vector_pool) {
            const pool = try allocator.create(VectorPool);
            pool.* = VectorPool.init(allocator);
            vector_pool = pool;
        }

        return .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .records = records,
            .id_index = id_index,
            .cached_norms = cached_norms,
            .config = config,
            .vector_pool = vector_pool,
            .rw_lock = .{},
        };
    }

    pub const BatchItem = struct {
        id: u64,
        vector: []const f32,
        metadata: ?[]const u8 = null,
    };

    /// Acquire read lock for concurrent read operations.
    /// Call unlockRead() when done.
    pub fn lockRead(self: *Database) void {
        if (self.config.thread_safe) {
            self.rw_lock.lockShared();
        }
    }

    /// Release read lock.
    pub fn unlockRead(self: *Database) void {
        if (self.config.thread_safe) {
            self.rw_lock.unlockShared();
        }
    }

    /// Acquire write lock for exclusive write operations.
    /// Call unlockWrite() when done.
    pub fn lockWrite(self: *Database) void {
        if (self.config.thread_safe) {
            self.rw_lock.lock();
        }
    }

    /// Release write lock.
    pub fn unlockWrite(self: *Database) void {
        if (self.config.thread_safe) {
            self.rw_lock.unlock();
        }
    }

    pub fn deinit(self: *Database) void {
        for (self.records.items) |record| {
            if (self.vector_pool) |pool| {
                pool.free(record.vector);
            } else {
                self.allocator.free(record.vector);
            }
            if (record.metadata) |meta| {
                self.allocator.free(meta);
            }
        }
        self.records.deinit(self.allocator);
        self.id_index.deinit(self.allocator);
        self.cached_norms.deinit(self.allocator);
        if (self.vector_pool) |pool| {
            pool.deinit();
            self.allocator.destroy(pool);
        }
        self.allocator.free(self.name);
        self.* = undefined;
    }

    pub fn insert(self: *Database, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        if (self.findIndex(id) != null) return DatabaseError.DuplicateId;

        // Validate vector dimensions against existing records
        if (self.records.items.len > 0 and vector.len != self.records.items[0].vector.len) {
            return DatabaseError.InvalidDimension;
        }

        const vector_copy = try self.cloneVector(vector);
        errdefer if (self.vector_pool) |pool| pool.free(vector_copy) else self.allocator.free(vector_copy);

        const metadata_copy = if (metadata) |meta|
            try self.allocator.dupe(u8, meta)
        else
            null;
        const new_index = self.records.items.len;
        try self.records.append(self.allocator, .{
            .id = id,
            .vector = vector_copy,
            .metadata = metadata_copy,
        });

        // Compute and cache L2 norm for fast similarity computation
        if (self.config.cache_norms) {
            const norm = simd.vectorL2Norm(vector_copy);
            try self.cached_norms.append(self.allocator, norm);
        }

        // Maintain O(1) lookup index
        try self.id_index.put(self.allocator, id, new_index);
    }

    pub fn reserve(self: *Database, additional: usize) !void {
        const target = self.records.items.len + additional;
        try self.records.ensureTotalCapacity(self.allocator, target);
        try self.id_index.ensureTotalCapacity(self.allocator, @intCast(target));
        if (self.config.cache_norms) {
            try self.cached_norms.ensureTotalCapacity(self.allocator, target);
        }
    }

    pub fn insertBatch(self: *Database, items: []const BatchItem) !void {
        if (items.len == 0) return;

        const base_dim = if (self.records.items.len > 0)
            self.records.items[0].vector.len
        else
            items[0].vector.len;

        for (items) |item| {
            if (item.vector.len != base_dim) return DatabaseError.InvalidDimension;
        }

        try self.reserve(items.len);
        for (items) |item| {
            try self.insert(item.id, item.vector, item.metadata);
        }
    }

    pub fn update(self: *Database, id: u64, vector: []const f32) !bool {
        const index = self.findIndex(id) orelse return false;
        const vector_copy = try self.cloneVector(vector);

        // Free old vector
        if (self.vector_pool) |pool| {
            pool.free(self.records.items[index].vector);
        } else {
            self.allocator.free(self.records.items[index].vector);
        }

        self.records.items[index].vector = vector_copy;

        // Update cached norm
        if (self.config.cache_norms and index < self.cached_norms.items.len) {
            self.cached_norms.items[index] = simd.vectorL2Norm(vector_copy);
        }

        return true;
    }

    pub fn delete(self: *Database, id: u64) bool {
        const index = self.findIndex(id) orelse return false;
        const record = self.records.swapRemove(index);

        // Free vector using pool or allocator
        if (self.vector_pool) |pool| {
            pool.free(record.vector);
        } else {
            self.allocator.free(record.vector);
        }
        if (record.metadata) |meta| {
            self.allocator.free(meta);
        }

        // Maintain cached norms parallel array
        if (self.config.cache_norms and self.cached_norms.items.len > 0) {
            _ = self.cached_norms.swapRemove(index);
        }

        // Remove from O(1) index
        _ = self.id_index.remove(id);
        // If swapRemove moved the last element to fill the gap, update its index
        if (index < self.records.items.len) {
            const moved_id = self.records.items[index].id;
            // Use getPtr to safely update existing entry without allocation
            if (self.id_index.getPtr(moved_id)) |idx_ptr| {
                idx_ptr.* = index;
            }
        }
        return true;
    }

    pub fn get(self: *Database, id: u64) ?VectorView {
        const index = self.findIndex(id) orelse return null;
        const record = self.records.items[index];
        return VectorView{
            .id = record.id,
            .vector = record.vector,
            .metadata = record.metadata,
        };
    }

    pub fn list(self: *Database, allocator: std.mem.Allocator, limit: usize) ![]VectorView {
        const count = @min(limit, self.records.items.len);
        const output = try allocator.alloc(VectorView, count);
        for (output, 0..) |*view, i| {
            const record = self.records.items[i];
            view.* = .{
                .id = record.id,
                .vector = record.vector,
                .metadata = record.metadata,
            };
        }
        return output;
    }

    /// Optimized search using single-pass algorithm with heap-based top-k selection.
    /// Uses cached norms when available for faster cosine similarity computation.
    pub fn search(
        self: *Database,
        allocator: std.mem.Allocator,
        query: []const f32,
        top_k: usize,
    ) ![]SearchResult {
        const qlen = query.len;
        if (qlen == 0 or self.records.items.len == 0 or top_k == 0) {
            return allocator.alloc(SearchResult, 0);
        }

        // Pre-compute query norm once
        const query_norm = simd.vectorL2Norm(query);
        if (query_norm == 0.0) {
            return allocator.alloc(SearchResult, 0);
        }

        // Single allocation: results buffer sized to top_k
        var results = try std.ArrayListUnmanaged(SearchResult).initCapacity(allocator, top_k);
        errdefer results.deinit(allocator);

        // Track minimum score in results for early rejection
        var min_score: f32 = -std.math.inf(f32);
        var min_idx: usize = 0;

        // Check if we have cached norms
        const use_cached_norms = self.config.cache_norms and
            self.cached_norms.items.len == self.records.items.len;

        // Single-pass: compute similarity and maintain top-k in-place
        for (self.records.items, 0..) |record, idx| {
            if (record.vector.len != qlen) continue;

            // Use optimized similarity with pre-computed norms
            const score = if (use_cached_norms)
                computeCosineSimilarityFast(query, query_norm, record.vector, self.cached_norms.items[idx])
            else
                simd.cosineSimilarity(query, record.vector);

            if (results.items.len < top_k) {
                // Still filling results, always add
                try results.append(allocator, .{ .id = record.id, .score = score });
                // Track new minimum
                if (score < min_score or results.items.len == 1) {
                    min_score = score;
                    min_idx = results.items.len - 1;
                }
            } else if (score > min_score) {
                // Replace minimum with this better result
                results.items[min_idx] = .{ .id = record.id, .score = score };
                // Find new minimum
                min_score = results.items[0].score;
                min_idx = 0;
                for (results.items, 0..) |r, i| {
                    if (r.score < min_score) {
                        min_score = r.score;
                        min_idx = i;
                    }
                }
            }
        }

        // Final sort for output ordering (only top_k elements, not full dataset)
        sortResults(results.items);
        return results.toOwnedSlice(allocator);
    }

    /// Batch search: compute similarity against multiple queries efficiently.
    pub fn searchBatch(
        self: *Database,
        allocator: std.mem.Allocator,
        queries: []const []const f32,
        top_k: usize,
    ) ![][]SearchResult {
        var all_results = try allocator.alloc([]SearchResult, queries.len);
        errdefer {
            for (all_results) |res| {
                if (res.len > 0) allocator.free(res);
            }
            allocator.free(all_results);
        }

        for (queries, 0..) |query, i| {
            all_results[i] = try self.search(allocator, query, top_k);
        }

        return all_results;
    }

    /// Thread-safe search with automatic locking.
    pub fn searchThreadSafe(
        self: *Database,
        allocator: std.mem.Allocator,
        query: []const f32,
        top_k: usize,
    ) ![]SearchResult {
        self.lockRead();
        defer self.unlockRead();
        return self.search(allocator, query, top_k);
    }

    /// Thread-safe insert with automatic locking.
    pub fn insertThreadSafe(self: *Database, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        self.lockWrite();
        defer self.unlockWrite();
        return self.insert(id, vector, metadata);
    }

    /// Thread-safe get with automatic locking.
    pub fn getThreadSafe(self: *Database, id: u64) ?VectorView {
        self.lockRead();
        defer self.unlockRead();
        return self.get(id);
    }

    /// Thread-safe delete with automatic locking.
    pub fn deleteThreadSafe(self: *Database, id: u64) bool {
        self.lockWrite();
        defer self.unlockWrite();
        return self.delete(id);
    }

    pub fn stats(self: *Database) Stats {
        if (self.records.items.len == 0) {
            return .{ .count = 0, .dimension = 0, .memory_bytes = 0, .norm_cache_enabled = self.config.cache_norms };
        }
        const dim = self.records.items[0].vector.len;
        const vector_bytes = self.records.items.len * dim * @sizeOf(f32);
        const norm_bytes = if (self.config.cache_norms) self.cached_norms.items.len * @sizeOf(f32) else 0;
        return .{
            .count = self.records.items.len,
            .dimension = dim,
            .memory_bytes = vector_bytes + norm_bytes,
            .norm_cache_enabled = self.config.cache_norms,
        };
    }

    /// Get comprehensive diagnostics for debugging and monitoring.
    /// Returns detailed information about memory usage, configuration, and health.
    pub fn diagnostics(self: *Database) DiagnosticsInfo {
        return DiagnosticsInfo.collect(self);
    }

    /// Rebuild norm cache (useful after bulk load or if cache becomes inconsistent).
    pub fn rebuildNormCache(self: *Database) !void {
        if (!self.config.cache_norms) return;

        self.cached_norms.clearRetainingCapacity();
        try self.cached_norms.ensureTotalCapacity(self.allocator, self.records.items.len);

        for (self.records.items) |record| {
            self.cached_norms.appendAssumeCapacity(simd.vectorL2Norm(record.vector));
        }
    }

    pub fn optimize(self: *Database) void {
        self.records.shrinkAndFree(self.allocator, self.records.items.len);
    }

    /// O(1) lookup using hash index instead of O(n) linear scan
    fn findIndex(self: *Database, id: u64) ?usize {
        return self.id_index.get(id);
    }

    fn cloneVector(self: *Database, vector: []const f32) ![]f32 {
        const copy = if (self.vector_pool) |pool|
            try pool.alloc(vector.len)
        else
            try self.allocator.alloc(f32, vector.len);
        std.mem.copyForwards(f32, copy, vector);
        return copy;
    }

    pub fn saveToFile(self: *const Database, path: []const u8) !void {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);

        // Build JSON manually in memory for Zig 0.16 compatibility
        var json_buf = std.ArrayListUnmanaged(u8).empty;
        defer json_buf.deinit(self.allocator);

        try json_buf.appendSlice(self.allocator, "[\n");
        for (self.records.items, 0..) |record, idx| {
            if (idx > 0) try json_buf.appendSlice(self.allocator, ",\n");
            try json_buf.appendSlice(self.allocator, "  {\"id\":");
            try json_buf.print(self.allocator, "{d}", .{record.id});
            try json_buf.appendSlice(self.allocator, ",\"vector\":[");
            for (record.vector, 0..) |v, vi| {
                if (vi > 0) try json_buf.append(self.allocator, ',');
                try json_buf.print(self.allocator, "{d:.8}", .{v});
            }
            try json_buf.append(self.allocator, ']');
            if (record.metadata) |meta| {
                try json_buf.appendSlice(self.allocator, ",\"metadata\":\"");
                // Escape string for JSON
                for (meta) |c| {
                    switch (c) {
                        '"' => try json_buf.appendSlice(self.allocator, "\\\""),
                        '\\' => try json_buf.appendSlice(self.allocator, "\\\\"),
                        '\n' => try json_buf.appendSlice(self.allocator, "\\n"),
                        '\r' => try json_buf.appendSlice(self.allocator, "\\r"),
                        '\t' => try json_buf.appendSlice(self.allocator, "\\t"),
                        else => try json_buf.append(self.allocator, c),
                    }
                }
                try json_buf.append(self.allocator, '"');
            }
            try json_buf.append(self.allocator, '}');
        }
        try json_buf.appendSlice(self.allocator, "\n]\n");

        // Write using writeStreamingAll for Zig 0.16 compatibility
        try file.writeStreamingAll(io, json_buf.items);
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Database {
        var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024 * 1024)); // 1GB limit
        defer allocator.free(content);

        var parsed = try std.json.parseFromSlice([]VectorRecord, allocator, content, .{});
        defer parsed.deinit();

        var records = std.ArrayListUnmanaged(VectorRecord).empty;
        try records.ensureTotalCapacity(allocator, parsed.value.len);

        // Build O(1) lookup index during load
        var id_index = std.AutoHashMapUnmanaged(u64, usize){};
        try id_index.ensureTotalCapacity(allocator, @intCast(parsed.value.len));

        // Build norm cache during load
        var cached_norms = std.ArrayListUnmanaged(f32).empty;
        try cached_norms.ensureTotalCapacity(allocator, parsed.value.len);

        for (parsed.value, 0..) |record, idx| {
            const vector_copy = try allocator.dupe(f32, record.vector);
            const metadata_copy = if (record.metadata) |m| try allocator.dupe(u8, m) else null;

            try records.append(allocator, .{
                .id = record.id,
                .vector = vector_copy,
                .metadata = metadata_copy,
            });
            id_index.putAssumeCapacity(record.id, idx);
            cached_norms.appendAssumeCapacity(simd.vectorL2Norm(vector_copy));
        }

        return Database{
            .allocator = allocator,
            .name = try allocator.dupe(u8, std.fs.path.basename(path)),
            .records = records,
            .id_index = id_index,
            .cached_norms = cached_norms,
            .config = .{ .cache_norms = true },
            .vector_pool = null,
        };
    }

    pub fn insertOwned(self: *Database, id: u64, vector: []f32, metadata: ?[]u8) !void {
        if (self.findIndex(id) != null) {
            self.freeVector(vector);
            if (metadata) |meta| self.allocator.free(meta);
            return DatabaseError.DuplicateId;
        }
        if (self.records.items.len > 0 and vector.len != self.records.items[0].vector.len) {
            self.freeVector(vector);
            if (metadata) |meta| self.allocator.free(meta);
            return DatabaseError.InvalidDimension;
        }
        errdefer {
            self.freeVector(vector);
            if (metadata) |meta| self.allocator.free(meta);
        }
        const new_index = self.records.items.len;
        try self.records.append(self.allocator, .{
            .id = id,
            .vector = vector,
            .metadata = metadata,
        });

        // Compute and cache L2 norm
        if (self.config.cache_norms) {
            const norm = simd.vectorL2Norm(vector);
            try self.cached_norms.append(self.allocator, norm);
        }

        // Maintain O(1) lookup index
        try self.id_index.put(self.allocator, id, new_index);
    }

    /// Free a vector using pool or allocator as appropriate.
    fn freeVector(self: *Database, vector: []f32) void {
        if (self.vector_pool) |pool| {
            pool.free(vector);
        } else {
            self.allocator.free(vector);
        }
    }
};

fn sortResults(results: []SearchResult) void {
    std.sort.pdq(SearchResult, results, {}, struct {
        fn lessThan(_: void, lhs: SearchResult, rhs: SearchResult) bool {
            return lhs.score > rhs.score;
        }
    }.lessThan);
}

/// Fast cosine similarity with pre-computed norms.
/// Avoids redundant norm computation in hot search path.
inline fn computeCosineSimilarityFast(a: []const f32, a_norm: f32, b: []const f32, b_norm: f32) f32 {
    if (a_norm == 0.0 or b_norm == 0.0) return 0.0;
    const dot = simd.vectorDot(a, b);
    return dot / (a_norm * b_norm);
}

test "search sorts by descending similarity and truncates" {
    var db = try Database.init(std.testing.allocator, "search-test");
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0 }, null);
    try db.insert(2, &.{ 0.0, 1.0 }, null);
    try db.insert(3, &.{ 1.0, 1.0 }, null);

    const results = try db.search(std.testing.allocator, &.{ 1.0, 0.0 }, 2);
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
    try std.testing.expectEqual(@as(u64, 3), results[1].id);
}

test "database with cached norms" {
    var db = try Database.initWithConfig(std.testing.allocator, "cached-norms-test", .{
        .cache_norms = true,
        .initial_capacity = 10,
    });
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0, 0.0, 0.0 }, null);
    try db.insert(2, &.{ 0.0, 1.0, 0.0, 0.0 }, null);
    try db.insert(3, &.{ 0.5, 0.5, 0.5, 0.5 }, null);

    // Check that norms are cached
    try std.testing.expectEqual(@as(usize, 3), db.cached_norms.items.len);

    // Verify stats include norm cache info
    const s = db.stats();
    try std.testing.expect(s.norm_cache_enabled);
    try std.testing.expect(s.memory_bytes > 0);

    // Search should use cached norms
    const results = try db.search(std.testing.allocator, &.{ 1.0, 0.0, 0.0, 0.0 }, 2);
    defer std.testing.allocator.free(results);

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqual(@as(u64, 1), results[0].id);
}

test "database update maintains norm cache" {
    var db = try Database.initWithConfig(std.testing.allocator, "update-cache-test", .{
        .cache_norms = true,
    });
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0 }, null);
    const original_norm = db.cached_norms.items[0];

    // Update vector
    _ = try db.update(1, &.{ 0.0, 1.0 });

    // Norm should be updated
    try std.testing.expectEqual(@as(usize, 1), db.cached_norms.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), db.cached_norms.items[0], 1e-6);
    try std.testing.expect(db.cached_norms.items[0] == original_norm); // Both are 1.0
}

test "database delete maintains norm cache consistency" {
    var db = try Database.initWithConfig(std.testing.allocator, "delete-cache-test", .{
        .cache_norms = true,
    });
    defer db.deinit();

    try db.insert(1, &.{ 1.0, 0.0 }, null);
    try db.insert(2, &.{ 0.0, 1.0 }, null);
    try db.insert(3, &.{ 0.5, 0.5 }, null);

    try std.testing.expectEqual(@as(usize, 3), db.cached_norms.items.len);

    // Delete middle element
    try std.testing.expect(db.delete(2));

    // Norm cache should remain consistent
    try std.testing.expectEqual(@as(usize, 2), db.cached_norms.items.len);
    try std.testing.expectEqual(@as(usize, 2), db.records.items.len);
}

test "fast cosine similarity matches regular" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };

    const a_norm = simd.vectorL2Norm(&a);
    const b_norm = simd.vectorL2Norm(&b);

    const fast_result = computeCosineSimilarityFast(&a, a_norm, &b, b_norm);
    const regular_result = simd.cosineSimilarity(&a, &b);

    try std.testing.expectApproxEqAbs(regular_result, fast_result, 1e-6);
}

test "database diagnostics" {
    var db = try Database.initWithConfig(std.testing.allocator, "diagnostics-test", .{
        .cache_norms = true,
        .initial_capacity = 100,
    });
    defer db.deinit();

    // Insert some test data with metadata
    try db.insert(1, &.{ 1.0, 0.0, 0.0, 0.0 }, "metadata1");
    try db.insert(2, &.{ 0.0, 1.0, 0.0, 0.0 }, "metadata2");
    try db.insert(3, &.{ 0.5, 0.5, 0.5, 0.5 }, null);

    // Get diagnostics
    const diag = db.diagnostics();

    // Verify basic info
    try std.testing.expectEqualStrings("diagnostics-test", diag.name);
    try std.testing.expectEqual(@as(usize, 3), diag.vector_count);
    try std.testing.expectEqual(@as(usize, 4), diag.dimension);

    // Verify memory stats
    try std.testing.expectEqual(@as(usize, 3 * 4 * @sizeOf(f32)), diag.memory.vector_bytes);
    try std.testing.expectEqual(@as(usize, 3 * @sizeOf(f32)), diag.memory.norm_cache_bytes);
    try std.testing.expect(diag.memory.metadata_bytes > 0);
    try std.testing.expect(diag.memory.total_bytes > 0);
    try std.testing.expect(diag.memory.efficiency > 0.0 and diag.memory.efficiency <= 1.0);

    // Verify config status
    try std.testing.expect(diag.config.norm_cache_enabled);
    try std.testing.expect(!diag.config.vector_pool_enabled);
    try std.testing.expect(!diag.config.thread_safe_enabled);

    // Verify health metrics
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), diag.index_health, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), diag.norm_cache_health, 0.01);
    try std.testing.expect(diag.isHealthy());

    // Verify formatting works
    const formatted = try diag.formatToString(std.testing.allocator);
    defer std.testing.allocator.free(formatted);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "diagnostics-test") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "Vectors: 3") != null);
}

test "diagnostics on empty database" {
    var db = try Database.init(std.testing.allocator, "empty-test");
    defer db.deinit();

    const diag = db.diagnostics();

    try std.testing.expectEqual(@as(usize, 0), diag.vector_count);
    try std.testing.expectEqual(@as(usize, 0), diag.dimension);
    try std.testing.expectEqual(@as(usize, 0), diag.memory.vector_bytes);
    try std.testing.expect(diag.isHealthy());
}
