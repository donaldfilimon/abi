//! In-memory vector database with persistence helpers.
//!
//! Performance optimizations:
//! - VectorPool: Size-class based pooling for common dimensions
//! - CachedNorms: Pre-computed L2 norms for faster cosine similarity
//! - SIMD batch operations: Vectorized similarity computation
//! - Cache-aligned storage: 64-byte alignment for hot data

const std = @import("std");
const simd = @import("../../services/shared/simd/mod.zig");

const sync = @import("../../services/shared/sync.zig");
const Mutex = sync.Mutex;

// Re-exports from extracted modules
const database_storage = @import("database_storage.zig");
pub const CACHE_LINE_SIZE = database_storage.CACHE_LINE_SIZE;
pub const HotVectorData = database_storage.HotVectorData;
pub const ColdVectorData = database_storage.ColdVectorData;
pub const VectorPool = database_storage.VectorPool;

const database_diagnostics = @import("database_diagnostics.zig");
pub const MemoryStats = database_diagnostics.MemoryStats;
pub const PerformanceStats = database_diagnostics.PerformanceStats;
pub const ConfigStatus = database_diagnostics.ConfigStatus;
pub const DiagnosticsInfo = database_diagnostics.DiagnosticsInfo;

// Zig 0.16 compatibility: Simple RwLock
const RwLock = struct {
    state: std.atomic.Value(i32) = std.atomic.Value(i32).init(0),
    pub fn lockShared(self: *RwLock) void {
        while (true) {
            const current = self.state.load(.acquire);
            if (current >= 0) {
                if (self.state.cmpxchgWeak(current, current + 1, .acquire, .monotonic) == null) break;
            }
            std.atomic.spinLoopHint();
        }
    }
    pub fn unlockShared(self: *RwLock) void {
        _ = self.state.fetchSub(1, .release);
    }
    pub fn lock(self: *RwLock) void {
        while (self.state.cmpxchgWeak(0, -1, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }
    pub fn unlock(self: *RwLock) void {
        self.state.store(0, .release);
    }
};

pub const DatabaseError = error{
    DuplicateId,
    VectorNotFound,
    InvalidDimension,
    PoolExhausted,
    PersistenceError,
    ConcurrencyError,
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
    rw_lock: RwLock,

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
        const query_norm = simd.vectorL2Norm(query);
        return self.searchAllocWithNorm(allocator, query, query_norm, top_k);
    }

    /// Fill a caller-provided buffer with up to top_k results.
    /// Returns the number of results written (sorted by score).
    pub fn searchInto(
        self: *Database,
        query: []const f32,
        top_k: usize,
        results: []SearchResult,
    ) usize {
        const query_norm = simd.vectorL2Norm(query);
        return self.searchIntoWithNorm(query, query_norm, top_k, results);
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

        if (queries.len == 0) return all_results;

        const query_norms = try allocator.alloc(f32, queries.len);
        defer allocator.free(query_norms);

        for (queries, 0..) |query, i| {
            query_norms[i] = simd.vectorL2Norm(query);
        }

        for (queries, 0..) |query, i| {
            all_results[i] = try self.searchAllocWithNorm(allocator, query, query_norms[i], top_k);
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
        if (self.config.cache_norms) {
            self.cached_norms.shrinkAndFree(self.allocator, self.cached_norms.items.len);
        }

        var compact_index = std.AutoHashMapUnmanaged(u64, usize){};
        if (compact_index.ensureTotalCapacity(self.allocator, @intCast(self.records.items.len))) {
            for (self.records.items, 0..) |record, idx| {
                compact_index.putAssumeCapacity(record.id, idx);
            }
            self.id_index.deinit(self.allocator);
            self.id_index = compact_index;
        } else |_| {
            compact_index.deinit(self.allocator);
        }
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

    fn searchAllocWithNorm(
        self: *Database,
        allocator: std.mem.Allocator,
        query: []const f32,
        query_norm: f32,
        top_k: usize,
    ) ![]SearchResult {
        if (query.len == 0 or self.records.items.len == 0 or top_k == 0) {
            return allocator.alloc(SearchResult, 0);
        }
        if (query_norm == 0.0) {
            return allocator.alloc(SearchResult, 0);
        }

        const alloc_len = @min(top_k, self.records.items.len);
        if (alloc_len == 0) {
            return allocator.alloc(SearchResult, 0);
        }

        var results = try allocator.alloc(SearchResult, alloc_len);
        errdefer allocator.free(results);

        const count = self.searchIntoWithNorm(query, query_norm, top_k, results);
        if (count == results.len) return results;
        return allocator.realloc(results, count);
    }

    fn searchIntoWithNorm(
        self: *Database,
        query: []const f32,
        query_norm: f32,
        top_k: usize,
        results: []SearchResult,
    ) usize {
        const qlen = query.len;
        if (qlen == 0 or self.records.items.len == 0 or top_k == 0 or results.len == 0) {
            return 0;
        }
        if (query_norm == 0.0) return 0;

        const limit = @min(top_k, results.len);

        // Track minimum score in results for early rejection
        var min_score: f32 = -std.math.inf(f32);
        var min_idx: usize = 0;
        var count: usize = 0;

        // Check if we have cached norms
        const use_cached_norms = self.config.cache_norms and
            self.cached_norms.items.len == self.records.items.len;

        // Single-pass: compute similarity and maintain top-k in-place
        // Prefetch distance for better cache performance on large datasets
        const prefetch_distance: usize = 4;

        for (self.records.items, 0..) |record, idx| {
            // Prefetch upcoming vectors to hide memory latency
            if (idx + prefetch_distance < self.records.items.len) {
                const future_record = &self.records.items[idx + prefetch_distance];
                @prefetch(future_record.vector.ptr, .{ .rw = .read, .locality = 3, .cache = .data });
            }

            if (record.vector.len != qlen) continue;

            // Use optimized similarity with pre-computed norms
            const score = if (use_cached_norms)
                computeCosineSimilarityFast(query, query_norm, record.vector, self.cached_norms.items[idx])
            else
                simd.cosineSimilarity(query, record.vector);

            if (count < limit) {
                results[count] = .{ .id = record.id, .score = score };
                if (count == 0 or score < min_score) {
                    min_score = score;
                    min_idx = count;
                }
                count += 1;
            } else if (score > min_score) {
                // Replace minimum with this better result
                results[min_idx] = .{ .id = record.id, .score = score };
                // Find new minimum
                min_score = results[0].score;
                min_idx = 0;
                for (results[0..count], 0..) |r, i| {
                    if (r.score < min_score) {
                        min_score = r.score;
                        min_idx = i;
                    }
                }
            }
        }

        // Final sort for output ordering (only top_k elements, not full dataset)
        if (count > 1) {
            sortResults(results[0..count]);
        }
        return count;
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
pub inline fn computeCosineSimilarityFast(a: []const f32, a_norm: f32, b: []const f32, b_norm: f32) f32 {
    if (a_norm == 0.0 or b_norm == 0.0) return 0.0;
    const dot = simd.vectorDot(a, b);
    return dot / (a_norm * b_norm);
}

test {
    _ = @import("database_storage.zig");
    _ = @import("database_diagnostics.zig");
    _ = @import("database_test.zig");
}
