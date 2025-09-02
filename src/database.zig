//! WDBX-AI Vector Database
//!
//! A high-performance, file-based vector database for storing and searching high-dimensional embeddings.
//! Features a custom binary format, efficient memory management, SIMD-accelerated search, and extensible metadata.
//!
//! # File Format
//!
//! The WDBX-AI format consists of:
//! - **Header (4096 bytes)**: Magic bytes, version, row count, dimensionality, and offset pointers
//! - **Records Section**: Densely packed float32 vectors, each record is `dim * sizeof(f32)` bytes
//! - **(Future)**: Index and schema blocks for ANN search and metadata
//!
//! # Usage Example
//!
//! ```zig
//! var db = try Db.open("vectors.wdbx", true);
//! defer db.close();
//! try db.init(384);
//! const embedding = [_]f32{0.1, 0.2, 0.3, ...};
//! const row_id = try db.addEmbedding(&embedding);
//! const query = [_]f32{0.15, 0.25, 0.35, ...};
//! const results = try db.search(&query, 10, allocator);
//! defer allocator.free(results);
//! ```
//!
//! # Features
//! - File-based, portable, and embeddable
//! - SIMD-accelerated brute-force search (O(n)), with future support for ANN (HNSW/IVF)
//! - Explicit memory management using Zig allocators
//! - Robust error handling and validation
//! - Extensible header for future schema/index support
//! - Comprehensive documentation and testability
//!
//! # TODO
//! - Add IVF (Inverted File) index support for large-scale search
//! - Import/export compatibility with Milvus/Faiss
//! - Add distributed sharding support
//! - Implement vector quantization for memory efficiency

const std = @import("std");

/// Database-specific error types
pub const DatabaseError = error{
    InvalidFileFormat,
    CorruptedData,
    InvalidDimensions,
    IndexOutOfBounds,
    InsufficientMemory,
    FileSystemError,
    LockContention,
    InvalidOperation,
    VersionMismatch,
    ChecksumMismatch,
};

/// Magic identifier for WDBX-AI files (7 bytes + NUL)
pub const MAGIC: [8]u8 = "WDBXAI\x00\x00".*;

/// Current file format version
pub const FORMAT_VERSION: u16 = 1;

/// Default page size for file operations (4 KiB)
pub const DEFAULT_PAGE_SIZE: u32 = 4096;

/// File-header fixed at 4 KiB (4096 bytes)
pub const WdbxHeader = struct {
    magic0: u8, // 'W'
    magic1: u8, // 'D'
    magic2: u8, // 'B'
    magic3: u8, // 'X'
    magic4: u8, // 'A'
    magic5: u8, // 'I'
    magic6: u8, // '\0'
    version: u16, // Format version number
    row_count: u64, // Number of records in the database
    dim: u16, // Dimensionality of each vector
    page_size: u32, // Page size used for file operations
    schema_off: u64, // Offset to schema information
    index_off: u64, // Offset to index data
    records_off: u64, // Offset to records section
    freelist_off: u64, // Offset to freelist for deleted records
    _reserved: [4072]u8, // Reserved space for future use

    pub fn validateMagic(self: *const WdbxHeader) bool {
        return self.magic0 == 'W' and
            self.magic1 == 'D' and
            self.magic2 == 'B' and
            self.magic3 == 'X' and
            self.magic4 == 'A' and
            self.magic5 == 'I' and
            self.magic6 == 0;
    }

    pub fn createDefault() WdbxHeader {
        return .{
            .magic0 = 'W',
            .magic1 = 'D',
            .magic2 = 'B',
            .magic3 = 'X',
            .magic4 = 'A',
            .magic5 = 'I',
            .magic6 = 0,
            .version = FORMAT_VERSION,
            .row_count = 0,
            .dim = 0,
            .page_size = DEFAULT_PAGE_SIZE,
            .schema_off = DEFAULT_PAGE_SIZE,
            .index_off = 0,
            .records_off = 0,
            .freelist_off = 0,
            ._reserved = [_]u8{0} ** 4072,
        };
    }
};

pub const Db = struct {
    file: std.fs.File,
    header: WdbxHeader,
    allocator: std.mem.Allocator,
    read_buffer: []u8,
    stats: DbStats = .{},
    /// HNSW index instance for fast approximate search
    hnsw_index: ?*HNSWIndex = null,

    pub const DbError = error{
        AlreadyInitialized,
        DimensionMismatch,
        InvalidState,
        OutOfMemory,
        FileBusy,
        EndOfStream,
        InvalidMagic,
        UnsupportedVersion,
        CorruptedDatabase,
    } || std.fs.File.SeekError || std.fs.File.WriteError ||
        std.fs.File.ReadError || std.fs.File.OpenError;

    pub fn isInitialized(self: *const Db) bool {
        return self.header.dim != 0 and self.header.row_count >= 0;
    }

    pub fn init(self: *Db, dim: u16) DbError!void {
        // Allow re-initialization with the same dimensions
        if (self.isInitialized()) {
            if (self.header.dim == dim) {
                // Already initialized with same dimensions, that's OK
                return;
            }
            return DbError.AlreadyInitialized;
        }
        if (dim == 0 or dim > 4096)
            return DbError.DimensionMismatch;

        self.header.dim = dim;
        self.header.records_off = DEFAULT_PAGE_SIZE;

        self.file.setEndPos(DEFAULT_PAGE_SIZE) catch |err| {
            return switch (err) {
                error.NonResizable => DbError.InvalidState,
                else => |e| e,
            };
        };

        try self.writeHeader();
        self.stats.initialization_count += 1;
    }

    fn writeHeader(self: *Db) DbError!void {
        try self.file.seekTo(0);
        const header_bytes = std.mem.asBytes(&self.header);
        try self.file.writeAll(header_bytes);
        try self.file.sync();
    }

    fn readHeader(self: *Db) DbError!void {
        try self.file.seekTo(0);
        const header_bytes = std.mem.asBytes(&self.header);
        _ = try self.file.readAll(header_bytes);
    }

    pub fn addEmbedding(self: *Db, embedding: []const f32) DbError!u64 {
        if (embedding.len != self.header.dim)
            return DbError.DimensionMismatch;
        if (self.header.records_off == 0)
            return DbError.InvalidState;

        const row_index = self.header.row_count;
        const record_size: u64 = @as(u64, self.header.dim) * @sizeOf(f32);
        const offset: u64 = self.header.records_off + row_index * record_size;

        const needed_size = offset + record_size;
        const current_size = try self.file.getEndPos();
        if (needed_size > current_size) {
            const page_size: u64 = self.header.page_size;
            const new_size = ((needed_size + page_size - 1) / page_size) * page_size;
            self.file.setEndPos(new_size) catch |err| {
                return switch (err) {
                    error.NonResizable => DbError.InvalidState,
                    else => |e| e,
                };
            };
        }

        try self.file.seekTo(offset);
        try self.file.writeAll(std.mem.sliceAsBytes(embedding));

        self.header.row_count += 1;
        try self.writeHeader();

        // Add to HNSW index if available
        if (self.hnsw_index != null) {
            self.addToHNSW(row_index, embedding) catch |err| {
                // Log error but don't fail the operation
                std.debug.print("Warning: Failed to add to HNSW index: {}\n", .{err});
            };
        }

        self.stats.write_count += 1;
        return row_index;
    }

    pub fn addEmbeddingsBatch(self: *Db, embeddings: []const []const f32) DbError![]u64 {
        const indices = try self.allocator.alloc(u64, embeddings.len);
        errdefer self.allocator.free(indices);

        for (embeddings, 0..) |embedding, i| {
            indices[i] = try self.addEmbedding(embedding);
        }
        return indices;
    }

    pub fn search(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator) DbError![]Result {
        const start_time = std.time.microTimestamp();
        defer {
            self.stats.search_count += 1;
            self.stats.total_search_time_us += @intCast(std.time.microTimestamp() - start_time);
        }

        if (query.len != self.header.dim)
            return DbError.DimensionMismatch;

        const row_count = self.header.row_count;
        if (row_count == 0)
            return allocator.alloc(Result, 0);

        const row_count_usize: usize = @intCast(row_count);
        const record_size: u64 = @as(u64, self.header.dim) * @sizeOf(f32);

        var all = try allocator.alloc(Result, row_count_usize);
        defer allocator.free(all);

        const buf_size = @min(record_size, self.read_buffer.len);
        const buf = self.read_buffer[0..buf_size];

        for (0..row_count_usize) |row| {
            const offset: u64 = self.header.records_off + @as(u64, row) * record_size;
            try self.file.seekTo(offset);
            _ = try self.file.readAll(buf);

            var dist: f32 = 0;
            const row_data = std.mem.bytesAsSlice(f32, buf);

            if (self.header.dim >= 16 and @hasDecl(std.simd, "f32x16")) {
                var i: usize = 0;
                while (i + 16 <= row_data.len) : (i += 16) {
                    const a = std.simd.f32x16.load(row_data[i..][0..16]);
                    const b = std.simd.f32x16.load(query[i..][0..16]);
                    const diff = a - b;
                    dist += std.simd.f32x16.reduce_add(diff * diff);
                }
                while (i < row_data.len) : (i += 1) {
                    const d = row_data[i] - query[i];
                    dist += d * d;
                }
            } else if (self.header.dim > 8) {
                var i: usize = 0;
                while (i + 4 <= row_data.len) : (i += 4) {
                    const diff0 = row_data[i] - query[i];
                    const diff1 = row_data[i + 1] - query[i + 1];
                    const diff2 = row_data[i + 2] - query[i + 2];
                    const diff3 = row_data[i + 3] - query[i + 3];
                    dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                }
                while (i < row_data.len) : (i += 1) {
                    const diff = row_data[i] - query[i];
                    dist += diff * diff;
                }
            } else {
                for (row_data, query) |val, q| {
                    const diff = val - q;
                    dist += diff * diff;
                }
            }

            all[row] = .{ .index = @intCast(row), .score = dist };
        }

        std.mem.sort(Result, all, {}, comptime Result.lessThanAsc);

        const result_len = @min(top_k, all.len);
        const out = try allocator.alloc(Result, result_len);
        @memcpy(out, all[0..result_len]);
        return out;
    }

    pub const Result = struct {
        index: u64,
        score: f32,
        pub fn lessThanAsc(_: void, a: Result, b: Result) bool {
            return a.score < b.score;
        }
    };

    pub const DbStats = struct {
        initialization_count: u64 = 0,
        write_count: u64 = 0,
        search_count: u64 = 0,
        total_search_time_us: u64 = 0,

        pub fn getAverageSearchTime(self: *const DbStats) u64 {
            if (self.search_count == 0) return 0;
            return self.total_search_time_us / self.search_count;
        }
    };

    pub fn open(path: []const u8, create_if_missing: bool) DbError!*Db {
        const allocator = std.heap.page_allocator;

        const file = std.fs.cwd().openFile(path, .{ .mode = .read_write }) catch |err| blk: {
            if (create_if_missing and err == error.FileNotFound) {
                const new_file = try std.fs.cwd().createFile(path, .{ .read = true, .truncate = true });
                const hdr = WdbxHeader.createDefault();
                try new_file.writeAll(std.mem.asBytes(&hdr));
                new_file.setEndPos(DEFAULT_PAGE_SIZE) catch |set_end_pos_err| {
                    return switch (set_end_pos_err) {
                        error.NonResizable => DbError.InvalidState,
                        else => |e| e,
                    };
                };
                break :blk new_file;
            } else {
                return err;
            }
        };

        const self = try allocator.create(Db);
        errdefer allocator.destroy(self);

        const read_buffer = try allocator.alloc(u8, DEFAULT_PAGE_SIZE);
        errdefer allocator.free(read_buffer);

        self.* = .{
            .file = file,
            .header = undefined,
            .allocator = allocator,
            .read_buffer = read_buffer,
        };

        try self.readHeader();

        if (!self.header.validateMagic()) {
            self.file.close();
            allocator.free(self.read_buffer);
            allocator.destroy(self);
            return DbError.InvalidMagic;
        }
        if (self.header.version != FORMAT_VERSION) {
            self.file.close();
            allocator.free(self.read_buffer);
            allocator.destroy(self);
            return DbError.UnsupportedVersion;
        }
        return self;
    }

    pub fn close(self: *Db) void {
        if (self.hnsw_index) |index| {
            index.deinit();
        }
        self.file.close();
        self.allocator.free(self.read_buffer);
        self.allocator.destroy(self);
    }

    pub fn getStats(self: *const Db) DbStats {
        return self.stats;
    }

    pub fn getRowCount(self: *const Db) u64 {
        return self.header.row_count;
    }

    pub fn getDimension(self: *const Db) u16 {
        return self.header.dim;
    }

    // ---------------------------------------------------------------
    // HNSW Index Implementation for Sub-linear Search
    // ---------------------------------------------------------------

    /// HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search
    pub const HNSWIndex = struct {
        const Self = @This();

        /// Maximum number of connections per layer
        const MAX_CONNECTIONS = 16;
        /// Construction parameter for search quality
        const EF_CONSTRUCTION = 200;
        /// Search parameter for query quality
        const EF_SEARCH = 100;

        /// Node in the HNSW graph
        const Node = struct {
            id: u64,
            vector: []f32,
            layer: u32,
            connections: std.ArrayListUnmanaged(u64),

            pub fn init(allocator: std.mem.Allocator, id: u64, vector: []const f32, layer: u32) !*Node {
                const self = try allocator.create(Node);
                self.* = .{
                    .id = id,
                    .vector = try allocator.dupe(f32, vector),
                    .layer = layer,
                    .connections = std.ArrayListUnmanaged(u64){},
                };
                return self;
            }

            pub fn deinit(self: *Node, allocator: std.mem.Allocator) void {
                allocator.free(self.vector);
                self.connections.deinit(allocator);
                allocator.destroy(self);
            }

            pub fn addConnection(self: *Node, allocator: std.mem.Allocator, node_id: u64) !void {
                try self.connections.append(allocator, node_id);
            }
        };

        /// Search result with distance
        const SearchResult = struct {
            id: u64,
            distance: f32,

            pub fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
                return a.distance < b.distance;
            }
        };

        /// Priority queue entry for search
        const QueueEntry = struct {
            id: u64,
            distance: f32,

            pub fn lessThan(_: void, a: QueueEntry, b: QueueEntry) bool {
                return a.distance > b.distance; // Max heap for closest neighbors
            }
        };

        allocator: std.mem.Allocator,
        nodes: std.AutoHashMap(u64, *Node),
        entry_point: ?u64 = null,
        max_layer: u32 = 0,
        dimension: u16,

        pub fn init(allocator: std.mem.Allocator, dimension: u16) !*Self {
            const self = try allocator.create(Self);
            self.* = .{
                .allocator = allocator,
                .nodes = std.AutoHashMap(u64, *Node).init(allocator),
                .dimension = dimension,
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            var it = self.nodes.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.*.deinit(self.allocator);
            }
            self.nodes.deinit();
            self.allocator.destroy(self);
        }

        /// Add a vector to the HNSW index
        pub fn addVector(self: *Self, id: u64, vector: []const f32) !void {
            if (vector.len != self.dimension) return error.DimensionMismatch;

            // Determine layer for this node (logarithmic distribution)
            const layer = self.getRandomLayer();

            // Create node
            const node = try Node.init(self.allocator, id, vector, layer);
            try self.nodes.put(id, node);

            if (self.entry_point == null) {
                self.entry_point = id;
                self.max_layer = layer;
            } else if (layer > self.max_layer) {
                self.entry_point = id;
                self.max_layer = layer;
            }

            // Connect to existing nodes
            try self.connectNode(node);
        }

        /// Search for approximate nearest neighbors
        pub fn search(self: *Self, query: []const f32, top_k: usize) ![]SearchResult {
            if (query.len != self.dimension) return error.DimensionMismatch;
            if (self.entry_point == null) return try self.allocator.alloc(SearchResult, 0);

            var visited = std.AutoHashMap(u64, void).init(self.allocator);
            defer visited.deinit();

            var candidates = std.PriorityQueue(QueueEntry, void, QueueEntry.lessThan).init(self.allocator);
            defer candidates.deinit();

            var results = std.PriorityQueue(SearchResult, void, SearchResult.lessThan).init(self.allocator);
            defer results.deinit();

            // Start from entry point
            const entry_node = self.nodes.get(self.entry_point.?) orelse return error.EntryPointNotFound;
            const entry_dist = self.calculateDistance(query, entry_node.vector);

            try candidates.add(.{ .id = self.entry_point.?, .distance = entry_dist });
            try visited.put(self.entry_point.?, {});

            // Search through layers
            var current_layer = self.max_layer;
            while (true) : (current_layer = if (current_layer > 0) current_layer - 1 else break) {
                try self.searchLayer(query, current_layer, &candidates, &visited, &results, top_k);
            }

            // Convert results to array
            const result_count = @min(top_k, results.count());
            const search_results = try self.allocator.alloc(SearchResult, result_count);

            var i: usize = 0;
            while (i < result_count) : (i += 1) {
                const entry = results.remove();
                search_results[i] = .{ .id = entry.id, .distance = entry.distance };
            }

            return search_results;
        }

        /// Search within a specific layer
        fn searchLayer(
            self: *Self,
            query: []const f32,
            layer: u32,
            candidates: *std.PriorityQueue(QueueEntry, void, QueueEntry.lessThan),
            visited: *std.AutoHashMap(u64, void),
            results: *std.PriorityQueue(SearchResult, void, SearchResult.lessThan),
            top_k: usize,
        ) !void {
            var layer_candidates = std.PriorityQueue(QueueEntry, void, QueueEntry.lessThan).init(self.allocator);
            defer layer_candidates.deinit();

            // Process candidates at this layer
            while (candidates.count() > 0) {
                const candidate = candidates.remove();
                const node = self.nodes.get(candidate.id) orelse continue;

                if (node.layer != layer) continue;

                // Add to results if better than current worst
                if (results.count() < top_k or candidate.distance < results.peek().?.distance) {
                    try results.add(.{ .id = candidate.id, .distance = candidate.distance });
                    if (results.count() > top_k) {
                        _ = results.remove();
                    }
                }

                // Explore neighbors
                for (node.connections.items) |neighbor_id| {
                    if (visited.contains(neighbor_id)) continue;

                    const neighbor = self.nodes.get(neighbor_id) orelse continue;
                    const dist = self.calculateDistance(query, neighbor.vector);

                    try visited.put(neighbor_id, {});
                    try layer_candidates.add(.{ .id = neighbor_id, .distance = dist });
                }
            }

            // Update main candidates for next layer
            while (layer_candidates.count() > 0) {
                try candidates.add(layer_candidates.remove());
            }
        }

        /// Distance entry for sorting
        const DistanceEntry = struct {
            id: u64,
            distance: f32,
        };

        /// Connect a new node to existing nodes
        fn connectNode(self: *Self, node: *Node) !void {
            const max_connections = @min(MAX_CONNECTIONS, self.nodes.count());
            if (max_connections == 0) return;

            var distances = std.ArrayListUnmanaged(DistanceEntry){};
            defer distances.deinit(self.allocator);

            // Calculate distances to all existing nodes
            var it = self.nodes.iterator();
            while (it.next()) |entry| {
                const existing_node = entry.value_ptr.*;
                if (existing_node.id == node.id) continue;

                const distance = self.calculateDistance(node.vector, existing_node.vector);
                try distances.append(self.allocator, .{ .id = existing_node.id, .distance = distance });
            }

            // Sort by distance and connect to closest
            std.sort.insertion(DistanceEntry, distances.items, {}, struct {
                fn lessThan(_: void, a: DistanceEntry, b: DistanceEntry) bool {
                    return a.distance < b.distance;
                }
            }.lessThan);

            // Connect to closest nodes
            const connect_count = @min(max_connections, distances.items.len);
            for (distances.items[0..connect_count]) |item| {
                try node.addConnection(self.allocator, item.id);

                // Bidirectional connection
                const neighbor = self.nodes.get(item.id) orelse continue;
                try neighbor.addConnection(self.allocator, node.id);
            }
        }

        /// Calculate Euclidean distance between vectors
        fn calculateDistance(self: *Self, a: []const f32, b: []const f32) f32 {
            _ = self;
            var sum: f32 = 0;

            // SIMD-optimized distance calculation
            if (a.len >= 16 and @hasDecl(std.simd, "f32x16")) {
                var i: usize = 0;
                while (i + 16 <= a.len) : (i += 16) {
                    const va = std.simd.f32x16.load(a[i..][0..16]);
                    const vb = std.simd.f32x16.load(b[i..][0..16]);
                    const diff = va - vb;
                    const sq = diff * diff;
                    sum += std.simd.f32x16.reduce_add(sq);
                }
                while (i < a.len) : (i += 1) {
                    const diff = a[i] - b[i];
                    sum += diff * diff;
                }
            } else if (a.len >= 8 and @hasDecl(std.simd, "f32x8")) {
                var i: usize = 0;
                while (i + 8 <= a.len) : (i += 8) {
                    const va = std.simd.f32x8.load(a[i..][0..8]);
                    const vb = std.simd.f32x8.load(b[i..][0..8]);
                    const diff = va - vb;
                    const sq = diff * diff;
                    sum += std.simd.f32x8.reduce_add(sq);
                }
                while (i < a.len) : (i += 1) {
                    const diff = a[i] - b[i];
                    sum += diff * diff;
                }
            } else {
                for (a, b) |va, vb| {
                    const diff = va - vb;
                    sum += diff * diff;
                }
            }

            return @sqrt(sum);
        }

        /// Get random layer (logarithmic distribution)
        fn getRandomLayer(self: *Self) u32 {
            _ = self;
            // Simple logarithmic distribution
            var layer: u32 = 0;
            while (layer < 10 and std.crypto.random.int(u32) % 2 == 0) : (layer += 1) {}
            return layer;
        }
    };

    /// Initialize HNSW index for faster search
    pub fn initHNSW(self: *Db) !void {
        if (self.hnsw_index != null) return error.AlreadyInitialized;

        self.hnsw_index = try HNSWIndex.init(self.allocator, self.header.dim);
    }

    /// Add vector to HNSW index
    pub fn addToHNSW(self: *Db, id: u64, vector: []const f32) !void {
        if (self.hnsw_index) |index| {
            try index.addVector(id, vector);
        }
    }

    /// Search using HNSW index (fallback to brute force if not available)
    pub fn searchHNSW(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator) ![]Result {
        if (self.hnsw_index) |index| {
            const hnsw_results = try index.search(query, top_k);
            defer self.allocator.free(hnsw_results);

            // Convert HNSW results to database results
            const results = try allocator.alloc(Result, hnsw_results.len);
            for (hnsw_results, 0..) |hnsw_result, i| {
                results[i] = .{ .index = hnsw_result.id, .score = hnsw_result.distance };
            }
            return results;
        } else {
            // Fallback to brute force search
            return self.search(query, top_k, allocator);
        }
    }

    /// Parallel search using multiple threads for brute force search
    pub fn searchParallel(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator, num_threads: u32) ![]Result {
        if (self.hnsw_index != null) {
            // Use HNSW if available (already fast)
            return self.searchHNSW(query, top_k, allocator);
        }

        if (num_threads <= 1) {
            return self.search(query, top_k, allocator);
        }

        const row_count = self.header.row_count;
        if (row_count == 0) return try allocator.alloc(Result, 0);

        const chunk_size = (row_count + num_threads - 1) / num_threads;
        var threads = try allocator.alloc(std.Thread, num_threads);
        defer allocator.free(threads);

        var thread_results = try allocator.alloc([]Result, num_threads);
        defer {
            for (thread_results) |result| allocator.free(result);
            allocator.free(thread_results);
        }

        // Spawn threads for parallel search
        for (0..num_threads) |i| {
            const start_row = i * chunk_size;
            const end_row = @min(start_row + chunk_size, row_count);

            threads[i] = try std.Thread.spawn(.{}, searchChunk, .{ self, query, start_row, end_row, &thread_results[i] });
        }

        // Wait for all threads to complete
        for (threads) |thread| {
            thread.join();
        }

        // Merge results from all threads
        return self.mergeSearchResults(thread_results, top_k, allocator);
    }

    /// Search a chunk of vectors (used by parallel search)
    fn searchChunk(self: *Db, query: []const f32, start_row: u64, end_row: u64, results: *[]Result) !void {
        if (start_row >= end_row) {
            results.* = try self.allocator.alloc(Result, 0);
            return;
        }

        const chunk_size = @as(usize, @intCast(end_row - start_row));
        var chunk_results = try self.allocator.alloc(Result, chunk_size);
        defer self.allocator.free(chunk_results);

        const record_size: u64 = @as(u64, self.header.dim) * @sizeOf(f32);
        const buf_size = @min(record_size, self.read_buffer.len);
        const buf = self.read_buffer[0..buf_size];

        var result_index: usize = 0;
        var row: u64 = start_row;
        while (row < end_row) : (row += 1) {
            const offset: u64 = self.header.records_off + row * record_size;
            try self.file.seekTo(offset);
            _ = try self.file.readAll(buf);

            var dist: f32 = 0;
            const row_data = std.mem.bytesAsSlice(f32, buf);

            // SIMD-optimized distance calculation
            if (self.header.dim >= 16 and @hasDecl(std.simd, "f32x16")) {
                var i: usize = 0;
                while (i + 16 <= row_data.len) : (i += 16) {
                    const va = std.simd.f32x16.load(row_data[i..][0..16]);
                    const vb = std.simd.f32x16.load(query[i..][0..16]);
                    const diff = va - vb;
                    const sq = diff * diff;
                    dist += std.simd.f32x16.reduce_add(sq);
                }
                while (i < row_data.len) : (i += 1) {
                    const diff = row_data[i] - query[i];
                    dist += diff * diff;
                }
            } else if (self.header.dim >= 8 and @hasDecl(std.simd, "f32x8")) {
                var i: usize = 0;
                while (i + 8 <= row_data.len) : (i += 8) {
                    const va = std.simd.f32x8.load(row_data[i..][0..8]);
                    const vb = std.simd.f32x8.load(query[i..][0..8]);
                    const diff = va - vb;
                    const sq = diff * diff;
                    dist += std.simd.f32x8.reduce_add(sq);
                }
                while (i < row_data.len) : (i += 1) {
                    const diff = row_data[i] - query[i];
                    dist += diff * diff;
                }
            } else {
                for (row_data, query) |va, vb| {
                    const diff = va - vb;
                    dist += diff * diff;
                }
            }

            chunk_results[result_index] = .{ .index = row, .score = dist };
            result_index += 1;
        }

        // Return only the valid results
        results.* = try self.allocator.dupe(Result, chunk_results[0..result_index]);
    }

    /// Merge search results from multiple threads
    fn mergeSearchResults(thread_results: [][]Result, top_k: usize, allocator: std.mem.Allocator) ![]Result {
        var all_results = std.ArrayListUnmanaged(Result){};
        defer all_results.deinit(allocator);

        // Collect all results
        for (thread_results) |results| {
            try all_results.appendSlice(allocator, results);
        }

        // Sort by score
        std.sort.insertion(Result, all_results.items, {}, Result.lessThanAsc);

        // Return top_k results
        const result_count = @min(top_k, all_results.items.len);
        const final_results = try allocator.alloc(Result, result_count);
        @memcpy(final_results, all_results.items[0..result_count]);

        return final_results;
    }
};
