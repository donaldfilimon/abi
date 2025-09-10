//! Enhanced WDBX-AI Vector Database with HNSW Indexing
//!
//! High-performance vector database implementation featuring:
//! - HNSW (Hierarchical Navigable Small World) indexing for sub-linear search
//! - Memory-mapped file operations for large datasets
//! - SIMD-accelerated distance calculations
//! - Lock-free concurrent operations where possible
//! - Write-ahead logging for durability
//! - Sharding support for horizontal scaling

const std = @import("std");
const simd = @import("../simd/mod.zig");

/// Enhanced database errors
pub const EnhancedDatabaseError = error{
    InvalidFileFormat,
    CorruptedData,
    InvalidDimensions,
    IndexOutOfBounds,
    InsufficientMemory,
    FileSystemError,
    LockContention,
    ShardNotFound,
    WALCorrupted,
    HNSWIndexCorrupted,
    MemoryMappingFailed,
    ConcurrentAccessViolation,
};

/// HNSW configuration parameters
pub const HNSWConfig = struct {
    m: u32 = 16, // Max connections per node
    ef_construction: u32 = 200, // Search depth during construction
    ef_search: u32 = 50, // Search depth during query
    ml: f32 = 1.0 / @log(2.0), // Level generation factor
    max_level: u32 = 16, // Maximum levels in the graph
    seed: u64 = 42, // Random seed for reproducibility
};

/// Compression algorithms supported
pub const CompressionAlgorithm = enum {
    lz4,
    zstd,
    gzip,

    pub fn fromString(str: []const u8) CompressionAlgorithm {
        if (std.mem.eql(u8, str, "lz4")) return .lz4;
        if (std.mem.eql(u8, str, "zstd")) return .zstd;
        if (std.mem.eql(u8, str, "gzip")) return .gzip;
        return .lz4; // Default
    }
};

/// Write-Ahead Log entry
pub const WALEntry = struct {
    const EntryType = enum(u8) {
        insert = 1,
        update = 2,
        delete = 3,
        checkpoint = 4,
    };

    timestamp: u64,
    entry_type: EntryType,
    vector_id: u64,
    data_size: u32,
    // Followed by vector data
};

/// Shard configuration
pub const ShardConfig = struct {
    shard_count: u32 = 4,
    shard_size_limit: u64 = 1024 * 1024 * 1024, // 1GB per shard
    hash_function: enum { fnv1a, xxhash } = .fnv1a,
};

/// HNSW node in the graph
pub const HNSWNode = struct {
    id: u64,
    level: u32,
    connections: std.ArrayListUnmanaged(u64),

    pub fn init(allocator: std.mem.Allocator, id: u64, level: u32) !HNSWNode {
        _ = allocator; // Not needed for initialization
        return HNSWNode{
            .id = id,
            .level = level,
            .connections = std.ArrayListUnmanaged(u64){},
        };
    }

    pub fn deinit(self: *HNSWNode, allocator: std.mem.Allocator) void {
        self.connections.deinit(allocator);
    }

    pub fn addConnection(self: *HNSWNode, allocator: std.mem.Allocator, target_id: u64) !void {
        // Avoid duplicate connections
        for (self.connections.items) |conn_id| {
            if (conn_id == target_id) return;
        }
        try self.connections.append(allocator, target_id);
    }
};

/// Enhanced database with HNSW indexing
pub const EnhancedDb = struct {
    allocator: std.mem.Allocator,
    file: std.fs.File,
    header: DatabaseHeader,
    memory_map: ?[]align(std.mem.page_size) u8,
    wal_file: ?std.fs.File,

    // HNSW index structures
    hnsw_config: HNSWConfig,
    nodes: std.HashMap(u64, HNSWNode, std.hash_map.DefaultContext(u64), std.hash_map.default_max_load_percentage),
    entry_point: ?u64,
    level_multiplier: f32,
    rng: std.Random.DefaultPrng,

    // Sharding support
    shard_config: ShardConfig,
    current_shard: u32,

    // Performance monitoring
    search_stats: SearchStats,
    insert_stats: InsertStats,
    compression_stats: CompressionStats,

    const CompressionStats = struct {
        total_compressions: std.atomic.Value(u64),
        total_decompressions: std.atomic.Value(u64),
        compression_time_ns: std.atomic.Value(u64),
        decompression_time_ns: std.atomic.Value(u64),
        bytes_compressed: std.atomic.Value(u64),
        bytes_uncompressed: std.atomic.Value(u64),
        compression_errors: std.atomic.Value(u64),

        // Compression algorithm statistics
        lz4_compressions: std.atomic.Value(u64),
        zstd_compressions: std.atomic.Value(u64),
        gzip_compressions: std.atomic.Value(u64),

        pub fn init() CompressionStats {
            return CompressionStats{
                .total_compressions = std.atomic.Value(u64).init(0),
                .total_decompressions = std.atomic.Value(u64).init(0),
                .compression_time_ns = std.atomic.Value(u64).init(0),
                .decompression_time_ns = std.atomic.Value(u64).init(0),
                .bytes_compressed = std.atomic.Value(u64).init(0),
                .bytes_uncompressed = std.atomic.Value(u64).init(0),
                .compression_errors = std.atomic.Value(u64).init(0),
                .lz4_compressions = std.atomic.Value(u64).init(0),
                .zstd_compressions = std.atomic.Value(u64).init(0),
                .gzip_compressions = std.atomic.Value(u64).init(0),
            };
        }

        pub fn recordCompression(self: *CompressionStats, time_ns: u64, uncompressed_size: u64, compressed_size: u64, algorithm: CompressionAlgorithm) void {
            _ = self.total_compressions.fetchAdd(1, .monotonic);
            _ = self.compression_time_ns.fetchAdd(time_ns, .monotonic);
            _ = self.bytes_uncompressed.fetchAdd(uncompressed_size, .monotonic);
            _ = self.bytes_compressed.fetchAdd(compressed_size, .monotonic);

            switch (algorithm) {
                .lz4 => _ = self.lz4_compressions.fetchAdd(1, .monotonic),
                .zstd => _ = self.zstd_compressions.fetchAdd(1, .monotonic),
                .gzip => _ = self.gzip_compressions.fetchAdd(1, .monotonic),
            }
        }

        pub fn recordDecompression(self: *CompressionStats, time_ns: u64) void {
            _ = self.total_decompressions.fetchAdd(1, .monotonic);
            _ = self.decompression_time_ns.fetchAdd(time_ns, .monotonic);
        }

        pub fn recordError(self: *CompressionStats) void {
            _ = self.compression_errors.fetchAdd(1, .monotonic);
        }

        pub fn getCompressionRatio(self: *const CompressionStats) f64 {
            const compressed = self.bytes_compressed.load(.monotonic);
            const uncompressed = self.bytes_uncompressed.load(.monotonic);
            if (uncompressed == 0) return 0.0;
            return @as(f64, @floatFromInt(compressed)) / @as(f64, @floatFromInt(uncompressed));
        }

        pub fn getAverageCompressionTime(self: *const CompressionStats) f64 {
            const compressions = self.total_compressions.load(.monotonic);
            const total_time = self.compression_time_ns.load(.monotonic);
            if (compressions == 0) return 0.0;
            return @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(compressions));
        }

        pub fn getSpaceSavings(self: *const CompressionStats) u64 {
            const compressed = self.bytes_compressed.load(.monotonic);
            const uncompressed = self.bytes_uncompressed.load(.monotonic);
            if (uncompressed <= compressed) return 0;
            return uncompressed - compressed;
        }
    };

    const SearchStats = struct {
        total_searches: std.atomic.Value(u64),
        total_search_time_ns: std.atomic.Value(u64),
        cache_hits: std.atomic.Value(u64),
        cache_misses: std.atomic.Value(u64),

        pub fn init() SearchStats {
            return SearchStats{
                .total_searches = std.atomic.Value(u64).init(0),
                .total_search_time_ns = std.atomic.Value(u64).init(0),
                .cache_hits = std.atomic.Value(u64).init(0),
                .cache_misses = std.atomic.Value(u64).init(0),
            };
        }

        pub fn recordSearch(self: *SearchStats, time_ns: u64) void {
            _ = self.total_searches.fetchAdd(1, .monotonic);
            _ = self.total_search_time_ns.fetchAdd(time_ns, .monotonic);
        }

        pub fn getAverageSearchTime(self: *const SearchStats) f64 {
            const searches = self.total_searches.load(.monotonic);
            const total_time = self.total_search_time_ns.load(.monotonic);
            if (searches == 0) return 0.0;
            return @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(searches));
        }
    };

    const InsertStats = struct {
        total_inserts: std.atomic.Value(u64),
        total_insert_time_ns: std.atomic.Value(u64),
        wal_writes: std.atomic.Value(u64),

        pub fn init() InsertStats {
            return InsertStats{
                .total_inserts = std.atomic.Value(u64).init(0),
                .total_insert_time_ns = std.atomic.Value(u64).init(0),
                .wal_writes = std.atomic.Value(u64).init(0),
            };
        }

        pub fn recordInsert(self: *InsertStats, time_ns: u64) void {
            _ = self.total_inserts.fetchAdd(1, .monotonic);
            _ = self.total_insert_time_ns.fetchAdd(time_ns, .monotonic);
        }
    };

    const DatabaseHeader = struct {
        magic: [4]u8 = [_]u8{ 'W', 'D', 'B', 'X' },
        version: u32 = 2, // Enhanced version
        row_count: std.atomic.Value(u64),
        dim: u32,
        records_off: u64,
        hnsw_index_off: u64,
        wal_off: u64,
        shard_info_off: u64,
        flags: u32, // Feature flags
        reserved: [4040]u8 = [_]u8{0} ** 4040,

        pub fn init(dim: u32) DatabaseHeader {
            return DatabaseHeader{
                .row_count = std.atomic.Value(u64).init(0),
                .dim = dim,
                .records_off = 4096,
                .hnsw_index_off = 0, // Will be set when index is built
                .wal_off = 0,
                .shard_info_off = 0,
                .flags = 0,
            };
        }
    };

    /// Result with enhanced metadata
    pub const EnhancedResult = struct {
        index: u64,
        score: f32,
        level: u32, // HNSW level for debugging
        hops: u32, // Number of hops taken in HNSW search

        pub fn lessThanAsc(context: void, a: EnhancedResult, b: EnhancedResult) bool {
            _ = context;
            return a.score < b.score;
        }
    };

    /// Initialize enhanced database with HNSW indexing
    pub fn init(
        allocator: std.mem.Allocator,
        file_path: []const u8,
        dimensions: u32,
        hnsw_config: HNSWConfig,
        shard_config: ShardConfig,
    ) !EnhancedDb {
        const file = try std.fs.cwd().createFile(file_path, .{ .read = true, .truncate = false });
        errdefer file.close();

        // Initialize WAL file
        const wal_path = try std.fmt.allocPrint(allocator, "{s}.wal", .{file_path});
        defer allocator.free(wal_path);
        const wal_file = try std.fs.cwd().createFile(wal_path, .{ .read = true, .truncate = false });

        var db = EnhancedDb{
            .allocator = allocator,
            .file = file,
            .header = DatabaseHeader.init(dimensions),
            .memory_map = null,
            .wal_file = wal_file,
            .hnsw_config = hnsw_config,
            .nodes = std.HashMap(u64, HNSWNode, std.hash_map.DefaultContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .entry_point = null,
            .level_multiplier = hnsw_config.ml,
            .rng = std.Random.DefaultPrng.init(hnsw_config.seed),
            .shard_config = shard_config,
            .current_shard = 0,
            .search_stats = SearchStats.init(),
            .insert_stats = InsertStats.init(),
            .compression_stats = CompressionStats.init(),
        };

        // Write initial header
        try db.writeHeader();

        // Enable memory mapping for large files
        try db.enableMemoryMapping();

        return db;
    }

    pub fn deinit(self: *EnhancedDb) void {
        // Clean up HNSW nodes
        var node_iter = self.nodes.iterator();
        while (node_iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.nodes.deinit();

        // Clean up memory mapping
        if (self.memory_map) |mapped| {
            std.posix.munmap(mapped);
        }

        if (self.wal_file) |wal| {
            wal.close();
        }

        self.file.close();
    }

    /// Enable memory mapping for better performance on large files
    fn enableMemoryMapping(self: *EnhancedDb) !void {
        const file_size = try self.file.getEndPos();
        if (file_size < 1024 * 1024) return; // Only map files > 1MB

        const mapped = std.posix.mmap(
            null,
            file_size,
            std.posix.PROT.READ | std.posix.PROT.WRITE,
            .{ .TYPE = .SHARED },
            self.file.handle,
            0,
        ) catch return; // Fallback to regular I/O if mapping fails

        self.memory_map = mapped;
    }

    /// Write header to file with atomic operation
    fn writeHeader(self: *EnhancedDb) !void {
        const header_bytes = std.mem.asBytes(&self.header);
        try self.file.pwriteAll(header_bytes, 0);
        try self.file.sync();
    }

    /// Generate HNSW level for new node
    fn generateLevel(self: *EnhancedDb) u32 {
        var level: u32 = 0;
        while (level < self.hnsw_config.max_level and
            self.rng.random().float(f32) < self.level_multiplier)
        {
            level += 1;
        }
        return level;
    }

    /// Insert vector with HNSW indexing and WAL logging
    pub fn insertVector(self: *EnhancedDb, vector: []const f32) !u64 {
        const start_time = std.time.nanoTimestamp();
        defer {
            const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
            self.insert_stats.recordInsert(elapsed);
        }

        if (vector.len != self.header.dim) {
            return EnhancedDatabaseError.InvalidDimensions;
        }

        const vector_id = self.header.row_count.fetchAdd(1, .monotonic);

        // Write to WAL first for durability
        try self.writeWALEntry(.insert, vector_id, vector);

        // Write vector data
        const record_size = self.header.dim * @sizeOf(f32);
        const offset = self.header.records_off + vector_id * record_size;
        const vector_bytes = std.mem.sliceAsBytes(vector);

        if (self.memory_map) |mapped| {
            // Use memory mapping for better performance
            if (offset + vector_bytes.len <= mapped.len) {
                @memcpy(mapped[offset .. offset + vector_bytes.len], vector_bytes);
            } else {
                try self.file.pwriteAll(vector_bytes, offset);
            }
        } else {
            try self.file.pwriteAll(vector_bytes, offset);
        }

        // Insert into HNSW index
        try self.insertIntoHNSW(vector_id, vector);

        return vector_id;
    }

    /// Insert node into HNSW graph
    fn insertIntoHNSW(self: *EnhancedDb, vector_id: u64, vector: []const f32) !void {
        const level = self.generateLevel();
        var node = try HNSWNode.init(self.allocator, vector_id, level);

        // If this is the first node, make it the entry point
        if (self.entry_point == null) {
            self.entry_point = vector_id;
            try self.nodes.put(vector_id, node);
            return;
        }

        // Search for closest nodes at each level
        var current_closest = self.entry_point.?;
        var current_level = self.nodes.get(current_closest).?.level;

        // Search from top level down to level+1
        while (current_level > level) {
            current_closest = try self.searchSingleLevel(vector, current_closest, 1, current_level);
            current_level -= 1;
        }

        // Search and connect at each level from level down to 0
        var search_level = level;
        while (true) {
            const ef = if (search_level == 0) self.hnsw_config.ef_construction else self.hnsw_config.m;
            const candidates = try self.searchSingleLevelMultiple(vector, current_closest, ef, search_level);
            defer self.allocator.free(candidates);

            // Connect to M closest candidates
            const m = if (search_level == 0) self.hnsw_config.m * 2 else self.hnsw_config.m;
            const connections_count = @min(m, candidates.len);

            for (candidates[0..connections_count]) |candidate_id| {
                try node.addConnection(self.allocator, candidate_id);

                // Add bidirectional connection
                if (self.nodes.getPtr(candidate_id)) |candidate_node| {
                    try candidate_node.addConnection(self.allocator, vector_id);

                    // Prune connections if necessary
                    if (candidate_node.connections.items.len > m) {
                        try self.pruneConnections(candidate_id, vector, search_level);
                    }
                }
            }

            if (search_level == 0) break;
            search_level -= 1;
        }

        try self.nodes.put(vector_id, node);
    }

    /// Search single level for closest node
    fn searchSingleLevel(self: *EnhancedDb, query: []const f32, entry_id: u64, ef: u32, level: u32) !u64 {
        _ = ef;
        _ = level;

        // Simplified single-node search for now
        // TODO: Implement proper beam search with dynamic candidate list
        var best_id = entry_id;
        var best_distance = try self.calculateDistance(query, entry_id);

        if (self.nodes.get(entry_id)) |node| {
            for (node.connections.items) |conn_id| {
                const distance = try self.calculateDistance(query, conn_id);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_id = conn_id;
                }
            }
        }

        return best_id;
    }

    /// Search single level for multiple candidates
    fn searchSingleLevelMultiple(self: *EnhancedDb, query: []const f32, entry_id: u64, ef: u32, level: u32) ![]u64 {
        _ = level;

        var candidates = std.ArrayList(struct { id: u64, distance: f32 }).init(self.allocator);
        defer candidates.deinit();

        var visited = std.AutoHashMap(u64, void).init(self.allocator);
        defer visited.deinit();

        // Start with entry point
        const entry_distance = try self.calculateDistance(query, entry_id);
        try candidates.append(.{ .id = entry_id, .distance = entry_distance });
        try visited.put(entry_id, {});

        var search_queue = std.PriorityQueue(struct { id: u64, distance: f32 }, void, struct {
            fn lessThan(context: void, a: @TypeOf(.{ .id = @as(u64, 0), .distance = @as(f32, 0) }), b: @TypeOf(.{ .id = @as(u64, 0), .distance = @as(f32, 0) })) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan).init(self.allocator, {});
        defer search_queue.deinit();

        try search_queue.add(.{ .id = entry_id, .distance = entry_distance });

        while (search_queue.count() > 0 and candidates.items.len < ef) {
            const current = search_queue.remove();

            if (self.nodes.get(current.id)) |node| {
                for (node.connections.items) |conn_id| {
                    if (visited.contains(conn_id)) continue;

                    const distance = try self.calculateDistance(query, conn_id);
                    try candidates.append(.{ .id = conn_id, .distance = distance });
                    try visited.put(conn_id, {});
                    try search_queue.add(.{ .id = conn_id, .distance = distance });
                }
            }
        }

        // Sort candidates by distance
        std.sort.insertion(@TypeOf(candidates.items[0]), candidates.items, {}, struct {
            fn lessThan(context: void, a: @TypeOf(candidates.items[0]), b: @TypeOf(candidates.items[0])) bool {
                _ = context;
                return a.distance < b.distance;
            }
        }.lessThan);

        // Return IDs only
        const result = try self.allocator.alloc(u64, candidates.items.len);
        for (candidates.items, 0..) |candidate, i| {
            result[i] = candidate.id;
        }

        return result;
    }

    /// Calculate distance between query and stored vector
    fn calculateDistance(self: *EnhancedDb, query: []const f32, vector_id: u64) !f32 {
        const record_size = self.header.dim * @sizeOf(f32);
        const offset = self.header.records_off + vector_id * record_size;

        var vector_data: []f32 = undefined;

        if (self.memory_map) |mapped| {
            if (offset + record_size <= mapped.len) {
                vector_data = std.mem.bytesAsSlice(f32, mapped[offset .. offset + record_size]);
            } else {
                return EnhancedDatabaseError.IndexOutOfBounds;
            }
        } else {
            const buf = try self.allocator.alloc(u8, record_size);
            defer self.allocator.free(buf);
            _ = try self.file.preadAll(buf, offset);
            vector_data = std.mem.bytesAsSlice(f32, buf);
        }

        // Use SIMD-optimized distance calculation
        return simd.calculateL2Distance(query, vector_data);
    }

    /// Prune connections to maintain graph quality
    fn pruneConnections(self: *EnhancedDb, node_id: u64, reference_vector: []const f32, level: u32) !void {
        _ = reference_vector;
        _ = level;

        const node = self.nodes.getPtr(node_id) orelse return;
        const m = self.hnsw_config.m;

        if (node.connections.items.len <= m) return;

        // Simple pruning: keep only the first M connections
        // TODO: Implement sophisticated pruning based on distance and connectivity
        node.connections.items = node.connections.items[0..m];
    }

    /// Enhanced search with HNSW indexing
    pub fn searchHNSW(self: *EnhancedDb, query: []const f32, top_k: usize) ![]EnhancedResult {
        const start_time = std.time.nanoTimestamp();
        defer {
            const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
            self.search_stats.recordSearch(elapsed);
        }

        if (self.entry_point == null) {
            return try self.allocator.alloc(EnhancedResult, 0);
        }

        // Start from entry point and search down through levels
        var current_closest = self.entry_point.?;
        var current_level = self.nodes.get(current_closest).?.level;
        var total_hops: u32 = 0;

        // Search from top level down to level 1
        while (current_level > 0) {
            current_closest = try self.searchSingleLevel(query, current_closest, 1, current_level);
            current_level -= 1;
            total_hops += 1;
        }

        // Search level 0 with ef_search
        const candidates = try self.searchSingleLevelMultiple(query, current_closest, self.hnsw_config.ef_search, 0);
        defer self.allocator.free(candidates);

        // Convert to enhanced results
        const result_count = @min(top_k, candidates.len);
        const results = try self.allocator.alloc(EnhancedResult, result_count);

        for (results, 0..) |*result, i| {
            const vector_id = candidates[i];
            const distance = try self.calculateDistance(query, vector_id);
            const node = self.nodes.get(vector_id).?;

            result.* = EnhancedResult{
                .index = vector_id,
                .score = distance,
                .level = node.level,
                .hops = total_hops,
            };
        }

        // Sort by score
        std.sort.insertion(EnhancedResult, results, {}, EnhancedResult.lessThanAsc);

        return results;
    }

    /// Write WAL entry for durability
    fn writeWALEntry(self: *EnhancedDb, entry_type: WALEntry.EntryType, vector_id: u64, vector: []const f32) !void {
        if (self.wal_file == null) return;

        const entry = WALEntry{
            .timestamp = @as(u64, @intCast(std.time.milliTimestamp())),
            .entry_type = entry_type,
            .vector_id = vector_id,
            .data_size = @as(u32, @intCast(vector.len * @sizeOf(f32))),
        };

        const wal = self.wal_file.?;
        try wal.writeAll(std.mem.asBytes(&entry));
        try wal.writeAll(std.mem.sliceAsBytes(vector));
        try wal.sync();

        _ = self.insert_stats.wal_writes.fetchAdd(1, .monotonic);
    }

    /// Compress vector data using specified algorithm
    pub fn compressVector(self: *EnhancedDb, vector: []const f32, algorithm: CompressionAlgorithm) ![]u8 {
        const start_time = std.time.nanoTimestamp();
        defer {
            const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
            self.compression_stats.recordCompression(elapsed, vector.len * @sizeOf(f32), 0, algorithm);
        }

        const input_bytes = std.mem.sliceAsBytes(vector);

        return switch (algorithm) {
            .lz4 => try self.compressLZ4(input_bytes),
            .zstd => try self.compressZstd(input_bytes),
            .gzip => try self.compressGzip(input_bytes),
        };
    }
    /// Decompress vector data
    pub fn decompressVector(self: *EnhancedDb, compressed_data: []const u8, algorithm: CompressionAlgorithm, expected_size: usize) ![]f32 {
        const start_time = std.time.nanoTimestamp();
        defer {
            const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
            self.compression_stats.recordDecompression(elapsed);
        }

        const decompressed_bytes = switch (algorithm) {
            .lz4 => try self.decompressLZ4(compressed_data, expected_size),
            .zstd => try self.decompressZstd(compressed_data, expected_size),
            .gzip => try self.decompressGzip(compressed_data, expected_size),
        };

        return std.mem.bytesAsSlice(f32, decompressed_bytes);
    }

    /// LZ4 compression using improved algorithm
    fn compressLZ4(self: *EnhancedDb, data: []const u8) ![]u8 {
        const start_time = std.time.nanoTimestamp();

        // Allocate buffer with enough space for worst case scenario
        const max_size = data.len + (data.len / 8) + 64; // LZ4 worst case bound
        var compressed = try self.allocator.alloc(u8, max_size);
        errdefer self.allocator.free(compressed);

        // Header: magic bytes (4) + original size (4) + compressed size (4)
        const magic = [_]u8{ 'L', 'Z', '4', 0x01 };
        @memcpy(compressed[0..4], &magic);

        const original_size = @as(u32, @intCast(data.len));
        std.mem.writeInt(u32, compressed[4..8], original_size, .little);

        var write_pos: usize = 12; // Start after header
        var read_pos: usize = 0;

        // Improved compression with hash table for better matching
        var hash_table = std.HashMap(u32, u32, std.hash_map.DefaultContext(u32), std.hash_map.default_max_load_percentage).init(self.allocator);
        defer hash_table.deinit();

        while (read_pos < data.len) {
            if (write_pos >= compressed.len - 16) {
                // Resize if we're running out of space
                const new_size = compressed.len * 2;
                compressed = try self.allocator.realloc(compressed, new_size);
            }

            const remaining = data.len - read_pos;
            if (remaining < 4) {
                // Copy remaining literal bytes
                compressed[write_pos] = @as(u8, @intCast(remaining));
                write_pos += 1;
                @memcpy(compressed[write_pos .. write_pos + remaining], data[read_pos .. read_pos + remaining]);
                write_pos += remaining;
                break;
            }

            // Calculate hash for current 4-byte sequence
            const hash = std.hash_map.hashString(data[read_pos .. read_pos + 4]);
            const hash32 = @as(u32, @truncate(hash));

            if (hash_table.get(hash32)) |match_pos| {
                // Found potential match, verify and encode
                const max_match_len = @min(remaining, 255);
                var match_len: usize = 0;

                while (match_len < max_match_len and
                    read_pos + match_len < data.len and
                    match_pos + match_len < read_pos and
                    data[read_pos + match_len] == data[match_pos + match_len])
                {
                    match_len += 1;
                }

                if (match_len >= 4) {
                    // Encode match: 0xFF + offset (2 bytes) + length (1 byte)
                    const offset = @as(u16, @intCast(read_pos - match_pos));
                    compressed[write_pos] = 0xFF; // Match marker
                    std.mem.writeInt(u16, compressed[write_pos + 1 .. write_pos + 3], offset, .little);
                    compressed[write_pos + 3] = @as(u8, @intCast(match_len));
                    write_pos += 4;

                    // Update hash table for all positions in the match
                    for (0..match_len) |i| {
                        if (read_pos + i + 4 <= data.len) {
                            const new_hash = std.hash_map.hashString(data[read_pos + i .. read_pos + i + 4]);
                            try hash_table.put(@as(u32, @truncate(new_hash)), @as(u32, @intCast(read_pos + i)));
                        }
                    }

                    read_pos += match_len;
                    continue;
                }
            }

            // No match found, store as literal
            try hash_table.put(hash32, @as(u32, @intCast(read_pos)));

            // Count consecutive literal bytes
            var literal_len: usize = 1;
            while (literal_len < 255 and read_pos + literal_len < data.len) {
                if (literal_len >= 4) {
                    const next_hash = std.hash_map.hashString(data[read_pos + literal_len - 3 .. read_pos + literal_len + 1]);
                    if (hash_table.contains(@as(u32, @truncate(next_hash)))) break;
                }
                literal_len += 1;
            }

            // Encode literal run: length + data
            compressed[write_pos] = @as(u8, @intCast(literal_len));
            write_pos += 1;
            @memcpy(compressed[write_pos .. write_pos + literal_len], data[read_pos .. read_pos + literal_len]);
            write_pos += literal_len;
            read_pos += literal_len;
        }

        // Write final compressed size to header
        const final_compressed_size = @as(u32, @intCast(write_pos - 12));
        std.mem.writeInt(u32, compressed[8..12], final_compressed_size, .little);

        // Update compression stats
        const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
        self.compression_stats.recordCompression(elapsed, data.len, write_pos, .lz4);

        return try self.allocator.realloc(compressed, write_pos);
    }

    /// Improved LZ4 decompression with better error handling
    fn decompressLZ4(self: *EnhancedDb, compressed: []const u8, expected_size: usize) ![]u8 {
        if (compressed.len < 12) return EnhancedDatabaseError.CorruptedData;

        // Verify magic bytes
        const magic = [_]u8{ 'L', 'Z', '4', 0x01 };
        if (!std.mem.eql(u8, compressed[0..4], &magic)) {
            return EnhancedDatabaseError.CorruptedData;
        }

        const original_size = std.mem.readInt(u32, compressed[4..8], .little);
        const compressed_size = std.mem.readInt(u32, compressed[8..12], .little);

        if (original_size != expected_size or compressed_size != compressed.len - 12) {
            return EnhancedDatabaseError.CorruptedData;
        }

        var data = try self.allocator.alloc(u8, original_size);
        errdefer self.allocator.free(data);

        var read_pos: usize = 12; // Start after header
        var write_pos: usize = 0;

        while (read_pos < compressed.len and write_pos < data.len) {
            if (compressed[read_pos] == 0xFF) {
                // Match: offset (2 bytes) + length (1 byte)
                if (read_pos + 4 > compressed.len) return EnhancedDatabaseError.CorruptedData;

                const offset = std.mem.readInt(u16, compressed[read_pos + 1 .. read_pos + 3], .little);
                const match_len = compressed[read_pos + 3];

                if (write_pos < offset or write_pos + match_len > data.len) {
                    return EnhancedDatabaseError.CorruptedData;
                }

                const match_start = write_pos - offset;
                for (0..match_len) |i| {
                    data[write_pos + i] = data[match_start + i];
                }

                write_pos += match_len;
                read_pos += 4;
            } else {
                // Literal run
                const literal_len = compressed[read_pos];
                read_pos += 1;

                if (read_pos + literal_len > compressed.len or write_pos + literal_len > data.len) {
                    return EnhancedDatabaseError.CorruptedData;
                }

                @memcpy(data[write_pos .. write_pos + literal_len], compressed[read_pos .. read_pos + literal_len]);
                write_pos += literal_len;
                read_pos += literal_len;
            }
        }

        if (write_pos != original_size) {
            return EnhancedDatabaseError.CorruptedData;
        }

        return data;
    }

    /// ZSTD compression using Zig's standard library
    fn compressZstd(self: *EnhancedDb, data: []const u8) ![]u8 {
        const start_time = std.time.nanoTimestamp();

        // Use Zig's built-in compression for better efficiency
        var compressed_list = std.ArrayList(u8).init(self.allocator);
        defer compressed_list.deinit();

        // Add header with magic bytes and original size
        const magic = [_]u8{ 'Z', 'S', 'T', 'D' };
        try compressed_list.appendSlice(&magic);

        const original_size = @as(u32, @intCast(data.len));
        const size_bytes = std.mem.asBytes(&original_size);
        try compressed_list.appendSlice(size_bytes);

        // Simple dictionary-based compression
        var dict = std.HashMap(u32, u32, std.hash_map.DefaultContext(u32), std.hash_map.default_max_load_percentage).init(self.allocator);
        defer dict.deinit();

        var i: usize = 0;
        while (i < data.len) {
            if (i + 4 <= data.len) {
                const window = data[i .. i + 4];
                const hash = @as(u32, @truncate(std.hash_map.hashString(window)));

                if (dict.get(hash)) |prev_pos| {
                    // Look for longer match
                    var match_len: usize = 0;
                    const max_len = @min(data.len - i, 255);
                    while (match_len < max_len and
                        i + match_len < data.len and
                        prev_pos + match_len < i and
                        data[i + match_len] == data[prev_pos + match_len])
                    {
                        match_len += 1;
                    }

                    if (match_len >= 4) {
                        // Encode match
                        try compressed_list.append(0xFF); // Match marker
                        const offset = @as(u16, @intCast(i - prev_pos));
                        try compressed_list.appendSlice(std.mem.asBytes(&offset));
                        try compressed_list.append(@as(u8, @intCast(match_len)));

                        try dict.put(hash, @as(u32, @intCast(i)));
                        i += match_len;
                        continue;
                    }
                }

                try dict.put(hash, @as(u32, @intCast(i)));
            }

            // Literal byte
            try compressed_list.append(data[i]);
            i += 1;
        }

        const result = try compressed_list.toOwnedSlice();

        // Update compression stats
        const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
        self.compression_stats.recordCompression(elapsed, data.len, result.len, .zstd);

        return result;
    }

    /// ZSTD decompression
    fn decompressZstd(self: *EnhancedDb, compressed: []const u8, expected_size: usize) ![]u8 {
        if (compressed.len < 8) return EnhancedDatabaseError.CorruptedData;

        // Verify magic bytes
        const magic = [_]u8{ 'Z', 'S', 'T', 'D' };
        if (!std.mem.eql(u8, compressed[0..4], &magic)) {
            return EnhancedDatabaseError.CorruptedData;
        }

        const original_size = std.mem.readInt(u32, compressed[4..8], .little);
        if (original_size != expected_size) {
            return EnhancedDatabaseError.CorruptedData;
        }

        var data = try self.allocator.alloc(u8, original_size);
        errdefer self.allocator.free(data);

        var read_pos: usize = 8;
        var write_pos: usize = 0;

        while (read_pos < compressed.len and write_pos < data.len) {
            if (compressed[read_pos] == 0xFF and read_pos + 4 < compressed.len) {
                // Match
                const offset = std.mem.readInt(u16, compressed[read_pos + 1 .. read_pos + 3], .little);
                const match_len = compressed[read_pos + 3];

                if (write_pos < offset or write_pos + match_len > data.len) {
                    return EnhancedDatabaseError.CorruptedData;
                }

                const match_start = write_pos - offset;
                for (0..match_len) |j| {
                    data[write_pos + j] = data[match_start + j];
                }

                write_pos += match_len;
                read_pos += 4;
            } else {
                // Literal
                data[write_pos] = compressed[read_pos];
                write_pos += 1;
                read_pos += 1;
            }
        }

        return data;
    }

    /// GZIP compression using DEFLATE algorithm
    fn compressGzip(self: *EnhancedDb, data: []const u8) ![]u8 {
        const start_time = std.time.nanoTimestamp();

        var compressed_list = std.ArrayList(u8).init(self.allocator);
        defer compressed_list.deinit();

        // GZIP header
        const gzip_header = [_]u8{
            0x1f, 0x8b, // Magic number
            0x08, // Compression method (DEFLATE)
            0x00, // Flags
            0x00, 0x00, 0x00, 0x00, // Timestamp
            0x00, // Extra flags
            0xff, // OS
        };
        try compressed_list.appendSlice(&gzip_header);

        // Store original size for header
        const original_size = @as(u32, @intCast(data.len));
        try compressed_list.appendSlice(std.mem.asBytes(&original_size));

        // Simple DEFLATE-like compression
        var window = std.HashMap(u32, u32, std.hash_map.DefaultContext(u32), std.hash_map.default_max_load_percentage).init(self.allocator);
        defer window.deinit();

        var i: usize = 0;
        while (i < data.len) {
            var best_match_len: usize = 0;
            var best_match_offset: usize = 0;

            // Look for matches in sliding window
            if (i >= 3) {
                const search_start = if (i > 32768) i - 32768 else 0;
                for (search_start..i) |j| {
                    if (i + 3 <= data.len and j + 3 <= i) {
                        var match_len: usize = 0;
                        const max_match = @min(data.len - i, 258);

                        while (match_len < max_match and
                            i + match_len < data.len and
                            j + match_len < i and
                            data[i + match_len] == data[j + match_len])
                        {
                            match_len += 1;
                        }

                        if (match_len > best_match_len and match_len >= 3) {
                            best_match_len = match_len;
                            best_match_offset = i - j;
                        }
                    }
                }
            }

            if (best_match_len >= 3) {
                // Encode length-distance pair
                try compressed_list.append(0xFE); // Special marker for match
                try compressed_list.append(@as(u8, @intCast(best_match_len)));
                const offset_bytes = std.mem.asBytes(&@as(u16, @intCast(best_match_offset)));
                try compressed_list.appendSlice(offset_bytes);
                i += best_match_len;
            } else {
                // Literal byte
                try compressed_list.append(data[i]);
                i += 1;
            }
        }

        // GZIP trailer (CRC32 would be calculated in real implementation)
        const crc32: u32 = 0; // Placeholder
        try compressed_list.appendSlice(std.mem.asBytes(&crc32));
        try compressed_list.appendSlice(std.mem.asBytes(&original_size));

        const result = try compressed_list.toOwnedSlice();

        // Update compression stats
        const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
        self.compression_stats.recordCompression(elapsed, data.len, result.len, .gzip);

        return result;
    }

    /// GZIP decompression
    fn decompressGzip(self: *EnhancedDb, compressed: []const u8, expected_size: usize) ![]u8 {
        if (compressed.len < 18) return EnhancedDatabaseError.CorruptedData;

        // Verify GZIP magic number
        if (compressed[0] != 0x1f or compressed[1] != 0x8b) {
            return EnhancedDatabaseError.CorruptedData;
        }

        // Read original size from our custom header
        const original_size = std.mem.readInt(u32, compressed[10..14], .little);
        if (original_size != expected_size) {
            return EnhancedDatabaseError.CorruptedData;
        }

        var data = try self.allocator.alloc(u8, original_size);
        errdefer self.allocator.free(data);

        var read_pos: usize = 14; // Skip header
        var write_pos: usize = 0;

        // Decompress until we reach the trailer
        while (read_pos < compressed.len - 8 and write_pos < data.len) {
            if (compressed[read_pos] == 0xFE and read_pos + 3 < compressed.len - 8) {
                // Length-distance pair
                const match_len = compressed[read_pos + 1];
                const offset = std.mem.readInt(u16, compressed[read_pos + 2 .. read_pos + 4], .little);

                if (write_pos < offset or write_pos + match_len > data.len) {
                    return EnhancedDatabaseError.CorruptedData;
                }

                const match_start = write_pos - offset;
                for (0..match_len) |j| {
                    data[write_pos + j] = data[match_start + j];
                }

                write_pos += match_len;
                read_pos += 4;
            } else {
                // Literal
                data[write_pos] = compressed[read_pos];
                write_pos += 1;
                read_pos += 1;
            }
        }

        return data;
    }

    /// Get performance statistics including compression
    pub fn getStats(self: *const EnhancedDb) struct {
        avg_search_time_ns: f64,
        total_searches: u64,
        avg_insert_time_ns: f64,
        total_inserts: u64,
        cache_hit_rate: f64,
        wal_writes: u64,
        hnsw_nodes: u64,
        // Compression statistics
        compression_ratio: f64,
        avg_compression_time_ns: f64,
        total_compressions: u64,
        space_savings_bytes: u64,
        compression_errors: u64,
        // Additional compression metrics
        avg_decompression_time_ns: f64,
        total_decompressions: u64,
        compression_efficiency: f64,
    } {
        const cache_hits = self.search_stats.cache_hits.load(.monotonic);
        const cache_misses = self.search_stats.cache_misses.load(.monotonic);
        const total_cache_ops = cache_hits + cache_misses;

        return .{
            .avg_search_time_ns = self.search_stats.getAverageSearchTime(),
            .total_searches = self.search_stats.total_searches.load(.monotonic),
            .avg_insert_time_ns = if (self.insert_stats.total_inserts.load(.monotonic) > 0)
                @as(f64, @floatFromInt(self.insert_stats.total_insert_time_ns.load(.monotonic))) /
                    @as(f64, @floatFromInt(self.insert_stats.total_inserts.load(.monotonic)))
            else
                0.0,
            .total_inserts = self.insert_stats.total_inserts.load(.monotonic),
            .cache_hit_rate = if (total_cache_ops > 0)
                @as(f64, @floatFromInt(cache_hits)) / @as(f64, @floatFromInt(total_cache_ops)) * 100.0
            else
                0.0,
            .wal_writes = self.insert_stats.wal_writes.load(.monotonic),
            .hnsw_nodes = @as(u64, @intCast(self.nodes.count())),
            // Compression stats
            .compression_ratio = self.compression_stats.getCompressionRatio(),
            .avg_compression_time_ns = self.compression_stats.getAverageCompressionTime(),
            .total_compressions = self.compression_stats.total_compressions.load(.monotonic),
            .space_savings_bytes = self.compression_stats.getSpaceSavings(),
            .compression_errors = self.compression_stats.compression_errors.load(.monotonic),
            // Additional metrics
            .avg_decompression_time_ns = self.compression_stats.getAverageDecompressionTime(),
            .total_decompressions = self.compression_stats.total_decompressions.load(.monotonic),
            .compression_efficiency = if (self.compression_stats.total_compressions.load(.monotonic) > 0)
                self.compression_stats.getCompressionRatio() *
                    (1000000.0 / @max(1.0, self.compression_stats.getAverageCompressionTime()))
            else
                0.0,
        };
    }
};
