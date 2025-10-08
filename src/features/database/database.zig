//! WDBX-AI Vector Database - Production Implementation
//!
//! A high-performance, production-ready vector database with advanced features:
//! - SIMD-accelerated search with automatic CPU capability detection
//! - HNSW indexing for approximate nearest neighbor search
//! - Multi-level caching system (L1/L2/L3)
//! - Advanced compression with multiple algorithms
//! - Comprehensive monitoring and metrics
//! - Health monitoring and automatic recovery
//! - Distributed sharding for horizontal scaling
//! ABI Vector Database
//!
//! A high-performance, file-based vector database for storing and searching high-dimensional embeddings.
//! Features a custom binary format, efficient memory management, SIMD-accelerated search, and extensible metadata.
//!
//! # File Format
//!
//! The ABI format consists of:
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

const std = @import("std");
// Note: core functionality is now imported through module dependencies

/// Database-specific error types
pub const DatabaseError = error{
    // File format errors
    InvalidFileFormat,
    CorruptedData,
    VersionMismatch,
    ChecksumMismatch,

    // Data validation errors
    InvalidDimensions,
    IndexOutOfBounds,
    InvalidOperation,
    InvalidState,

    // Resource errors
    InsufficientMemory,
    FileSystemError,
    LockContention,
    ResourceExhausted,

    // Network errors (for distributed operations)
    NetworkError,
    Timeout,
    ConnectionFailed,

    // Configuration errors
    InvalidConfiguration,
    UnsupportedFeature,
};

/// Magic identifier for ABI files (7 bytes + NUL)
pub const MAGIC: [8]u8 = "WDBXAI\x00\x00".*;

/// Current file format version
pub const FORMAT_VERSION: u16 = 1;

/// Default page size for file operations (4 KiB)
pub const DEFAULT_PAGE_SIZE: u32 = 4096;

/// File-header fixed at 4 KiB (4096 bytes)
pub const WdbxHeader = extern struct {
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
    _reserved: [4032]u8, // Reserved space for future use

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
            ._reserved = [_]u8{0} ** 4032,
        };
    }
};

comptime {
    if (@sizeOf(WdbxHeader) != DEFAULT_PAGE_SIZE) {
        @compileError("WdbxHeader must occupy exactly DEFAULT_PAGE_SIZE bytes");
    }
}

pub const Db = struct {
    file: std.fs.File,
    header: WdbxHeader,
    allocator: std.mem.Allocator,
    read_buffer: []f32,
    stats: DbStats = .{},
    /// HNSW index instance for fast approximate search
    hnsw_index: ?*HNSWIndex = null,
    /// Minimal WAL support (append-only, recovery on open)
    db_path: []u8 = &[_]u8{},
    wal_file: ?std.fs.File = null,
    wal_enabled: bool = false,

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

        try self.ensureReadBufferCapacity(@intCast(dim));

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
        const read_len = try self.file.readAll(header_bytes);
        if (read_len != header_bytes.len) {
            return DbError.CorruptedDatabase;
        }
    }

    fn ensureReadBufferCapacity(self: *Db, required_dim: usize) DbError!void {
        if (required_dim == 0) return;
        if (required_dim <= self.read_buffer.len) return;

        const new_capacity = @max(required_dim, self.read_buffer.len * 2);
        const new_buffer = try self.allocator.alloc(f32, new_capacity);
        errdefer self.allocator.free(new_buffer);
        self.allocator.free(self.read_buffer);
        self.read_buffer = new_buffer;
    }

    // --- Minimal WAL helpers (Zig 0.16-compatible) ---
    fn walRecordSize(self: *const Db) u64 {
        return @as(u64, self.header.dim) * @sizeOf(f32);
    }

    fn initWAL(self: *Db) DbError!void {
        if (self.db_path.len == 0) return; // nothing to do
        const wal_path = std.fmt.allocPrint(self.allocator, "{s}.wal", .{self.db_path}) catch |err| return err;
        defer self.allocator.free(wal_path);
        // Open or create WAL file
        const wal_f = try (std.fs.cwd().createFile(wal_path, .{ .read = true, .truncate = false }) catch |e| switch (e) {
            error.PathAlreadyExists => std.fs.cwd().openFile(wal_path, .{ .mode = .read_write }),
            else => e,
        });
        self.wal_file = wal_f;
        self.wal_enabled = true;
    }

    fn walAppendEmbedding(self: *Db, embedding: []const f32) DbError!void {
        if (!self.wal_enabled or self.wal_file == null) return;
        const bytes = std.mem.sliceAsBytes(embedding);
        try self.wal_file.?.seekFromEnd(0);
        try self.wal_file.?.writeAll(bytes);
        try self.wal_file.?.sync();
    }

    fn walTruncate(self: *Db) DbError!void {
        if (!self.wal_enabled or self.wal_file == null) return;
        try self.wal_file.?.seekTo(0);
        self.wal_file.?.setEndPos(0) catch return DbError.InvalidState;
        try self.wal_file.?.sync();
    }

    fn recoverFromWAL(self: *Db) DbError!void {
        if (!self.wal_enabled or self.wal_file == null) return;
        const record_size = self.walRecordSize();
        const wal_len = try self.wal_file.?.getEndPos();
        if (wal_len == 0 or record_size == 0) return;
        if (wal_len % record_size != 0) {
            // Corrupted WAL; truncate to be safe
            try self.walTruncate();
            return;
        }
        const num = wal_len / record_size;
        const dim: usize = @intCast(self.header.dim);
        const tmp = try self.allocator.alloc(f32, dim);
        errdefer self.allocator.free(tmp);
        const tmp_bytes = std.mem.sliceAsBytes(tmp);
        try self.wal_file.?.seekTo(0);
        var i: u64 = 0;
        while (i < num) : (i += 1) {
            _ = try self.wal_file.?.readAll(tmp_bytes);
            const embedding: []const f32 = tmp;
            // Append to main file at current row_count
            const row_index = self.header.row_count;
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
        }
        try self.writeHeader();
        // Clear WAL after successful recovery
        try self.walTruncate();
    }

    pub fn addEmbedding(self: *Db, embedding: []const f32) DbError!u64 {
        // Must be initialized before accepting embeddings
        if (self.header.records_off == 0)
            return DbError.InvalidState;
        if (embedding.len != self.header.dim)
            return DbError.DimensionMismatch;

        // Write-ahead log before committing to main file
        self.walAppendEmbedding(embedding) catch |err| {
            // Non-fatal: keep going but warn
            std.log.warn("WAL append failed (continuing): {any}", .{err});
        };

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

        // Commit succeeded; truncate WAL
        self.walTruncate() catch |err| {
            std.log.warn("WAL truncate failed: {any}", .{err});
        };

        // Add to HNSW index if available
        if (self.hnsw_index != null) {
            self.addToHNSW(row_index, embedding) catch |err| {
                // Log error but don't fail the operation
                std.log.warn("Failed to add to HNSW index: {any}", .{err});
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
        const dim_usize: usize = @intCast(self.header.dim);

        try self.ensureReadBufferCapacity(dim_usize);
        const row_values = self.read_buffer[0..dim_usize];
        const row_bytes = std.mem.sliceAsBytes(row_values);

        var all = try allocator.alloc(Result, row_count_usize);
        defer allocator.free(all);

        for (0..row_count_usize) |row| {
            const offset: u64 = self.header.records_off + @as(u64, row) * record_size;
            _ = try self.file.preadAll(row_bytes, offset);

            var dist: f32 = 0;
            const row_data = row_values;

            if (self.header.dim >= 16 and @hasDecl(std.simd, "f32x16")) {
                var i: usize = 0;
                const Vec = std.simd.f32x16;
                while (i + 16 <= row_data.len) : (i += 16) {
                    const a: Vec = row_data[i .. i + 16][0..16].*;
                    const b: Vec = query[i .. i + 16][0..16].*;
                    const diff = a - b;
                    dist += @reduce(.Add, diff * diff);
                }
                while (i < row_data.len) : (i += 1) {
                    const d = row_data[i] - query[i];
                    dist += d * d;
                }
            } else if (self.header.dim >= 8 and @hasDecl(std.simd, "f32x8")) {
                var i: usize = 0;
                const Vec = std.simd.f32x8;
                while (i + 8 <= row_data.len) : (i += 8) {
                    const a: Vec = row_data[i .. i + 8][0..8].*;
                    const b: Vec = query[i .. i + 8][0..8].*;
                    const diff = a - b;
                    dist += @reduce(.Add, diff * diff);
                }
                while (i < row_data.len) : (i += 1) {
                    const diff = row_data[i] - query[i];
                    dist += diff * diff;
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
        errdefer allocator.free(out);
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

        var file: std.fs.File = undefined;
        if (create_if_missing) {
            // Always create/truncate when requested to ensure a clean slate
            file = try std.fs.cwd().createFile(path, .{ .read = true, .truncate = true });
            const hdr = WdbxHeader.createDefault();
            try file.writeAll(std.mem.asBytes(&hdr));
            file.setEndPos(DEFAULT_PAGE_SIZE) catch |set_end_pos_err| {
                return switch (set_end_pos_err) {
                    error.NonResizable => DbError.InvalidState,
                    else => |e| e,
                };
            };
        } else {
            file = try std.fs.cwd().openFile(path, .{ .mode = .read_write });
        }

        const self = try allocator.create(Db);
        errdefer allocator.destroy(self);

        const read_buffer = try allocator.alloc(f32, DEFAULT_PAGE_SIZE / @sizeOf(f32));
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
        // Setup WAL (best-effort) and recover if needed
        self.db_path = allocator.dupe(u8, path) catch |err| {
            self.file.close();
            allocator.free(self.read_buffer);
            allocator.destroy(self);
            return switch (err) {
                error.OutOfMemory => DbError.InsufficientMemory,
                else => DbError.FileSystemError,
            };
        };
        self.initWAL() catch |err| {
            std.log.warn("WAL init failed: {any}", .{err});
        };
        self.recoverFromWAL() catch |err| {
            std.log.warn("WAL recovery failed: {any}", .{err});
        };
        try self.ensureReadBufferCapacity(@intCast(self.header.dim));
        return self;
    }

    pub fn close(self: *Db) void {
        if (self.hnsw_index) |index| {
            index.deinit();
        }
        if (self.wal_file) |*wal| {
            wal.close();
        }
        if (self.db_path.len != 0) self.allocator.free(self.db_path);
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

        /// Tunable parameters
        max_connections: u32 = 16,
        ef_construction: u32 = 200,
        ef_search: u32 = 100,

        // (type declarations moved after fields)

        allocator: std.mem.Allocator,
        nodes: std.AutoHashMap(u64, *Node),
        entry_point: ?u64 = null,
        max_layer: u32 = 0,
        dimension: u16,

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

            pub fn compare(_: void, a: SearchResult, b: SearchResult) std.math.Order {
                // Maintain a max-heap by considering larger distance as "less"
                if (a.distance > b.distance) return .lt;
                if (a.distance < b.distance) return .gt;
                return .eq;
            }
        };

        /// Priority queue entry for search
        const QueueEntry = struct {
            id: u64,
            distance: f32,

            pub fn compare(_: void, a: QueueEntry, b: QueueEntry) std.math.Order {
                // Maintain a max-heap by considering larger distance as "less"
                if (a.distance > b.distance) return .lt;
                if (a.distance < b.distance) return .gt;
                return .eq;
            }
        };

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

            var candidates = std.PriorityQueue(QueueEntry, void, QueueEntry.compare).init(self.allocator, {});
            defer candidates.deinit();

            var results = std.PriorityQueue(SearchResult, void, SearchResult.compare).init(self.allocator, {});
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

            // Convert results to array (in ascending distance order)
            const result_count = @min(top_k, results.count());
            const search_results = try self.allocator.alloc(SearchResult, result_count);

            // Extract results in reverse order since we're using a max-heap
            var i: usize = result_count;
            while (i > 0) : (i -= 1) {
                const entry = results.remove();
                search_results[i - 1] = .{ .id = entry.id, .distance = entry.distance };
            }

            return search_results;
        }

        /// Search within a specific layer
        fn searchLayer(
            self: *Self,
            query: []const f32,
            layer: u32,
            candidates: *std.PriorityQueue(QueueEntry, void, QueueEntry.compare),
            visited: *std.AutoHashMap(u64, void),
            results: *std.PriorityQueue(SearchResult, void, SearchResult.compare),
            top_k: usize,
        ) !void {
            var layer_candidates = std.PriorityQueue(QueueEntry, void, QueueEntry.compare).init(self.allocator, {});
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
            const max_connections = @min(self.max_connections, self.nodes.count());
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
                const Vec = std.simd.f32x16;
                while (i + 16 <= a.len) : (i += 16) {
                    const va: Vec = a[i .. i + 16][0..16].*;
                    const vb: Vec = b[i .. i + 16][0..16].*;
                    const diff = va - vb;
                    const sq = diff * diff;
                    sum += @reduce(.Add, sq);
                }
                while (i < a.len) : (i += 1) {
                    const diff = a[i] - b[i];
                    sum += diff * diff;
                }
            } else if (a.len >= 8 and @hasDecl(std.simd, "f32x8")) {
                var i: usize = 0;
                const Vec = std.simd.f32x8;
                while (i + 8 <= a.len) : (i += 8) {
                    const va: Vec = a[i .. i + 8][0..8].*;
                    const vb: Vec = b[i .. i + 8][0..8].*;
                    const diff = va - vb;
                    const sq = diff * diff;
                    sum += @reduce(.Add, sq);
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

    /// Adjust HNSW parameters (useful for tests/benchmarks)
    pub fn setHNSWParams(self: *Db, params: struct { max_connections: u32 = 16, ef_construction: u32 = 200, ef_search: u32 = 100 }) void {
        if (self.hnsw_index) |index| {
            index.max_connections = params.max_connections;
            index.ef_construction = params.ef_construction;
            index.ef_search = params.ef_search;
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
            for (thread_results) |result| self.allocator.free(result);
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
        return mergeSearchResults(thread_results, top_k, allocator);
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
        const dim_usize: usize = @intCast(self.header.dim);
        // Use a per-thread buffer to avoid concurrent access to self.read_buffer
        const row_values = try self.allocator.alloc(f32, dim_usize);
        defer self.allocator.free(row_values);
        const row_bytes = std.mem.sliceAsBytes(row_values);

        var result_index: usize = 0;
        var row: u64 = start_row;
        while (row < end_row) : (row += 1) {
            const offset: u64 = self.header.records_off + row * record_size;
            _ = try self.file.preadAll(row_bytes, offset);

            var dist: f32 = 0;
            const row_data = row_values;

            // SIMD-optimized distance calculation
            if (self.header.dim >= 16 and @hasDecl(std.simd, "f32x16")) {
                var i: usize = 0;
                const Vec = std.simd.f32x16;
                while (i + 16 <= row_data.len) : (i += 16) {
                    const va: Vec = row_data[i .. i + 16][0..16].*;
                    const vb: Vec = query[i .. i + 16][0..16].*;
                    const diff = va - vb;
                    const sq = diff * diff;
                    dist += @reduce(.Add, sq);
                }
                while (i < row_data.len) : (i += 1) {
                    const diff = row_data[i] - query[i];
                    dist += diff * diff;
                }
            } else if (self.header.dim >= 8 and @hasDecl(std.simd, "f32x8")) {
                var i: usize = 0;
                const Vec = std.simd.f32x8;
                while (i + 8 <= row_data.len) : (i += 8) {
                    const va: Vec = row_data[i .. i + 8][0..8].*;
                    const vb: Vec = query[i .. i + 8][0..8].*;
                    const diff = va - vb;
                    const sq = diff * diff;
                    dist += @reduce(.Add, sq);
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
        errdefer allocator.free(final_results);
        @memcpy(final_results, all_results.items[0..result_count]);

        return final_results;
    }
};

test "Db add/search round trip" {
    const path = "test_db_temp.wdbx";
    std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile(path) catch {};

    {
        var db = try Db.open(path, true);
        defer db.close();
        try db.init(4);

        const vec_a = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
        const vec_b = [_]f32{ 0.2, 0.1, 0.4, 0.3 };

        const id_a = try db.addEmbedding(&vec_a);
        const id_b = try db.addEmbedding(&vec_b);
        try std.testing.expectEqual(@as(u64, 0), id_a);
        try std.testing.expectEqual(@as(u64, 1), id_b);

        const allocator = std.testing.allocator;
        const results = try db.search(&vec_a, 2, allocator);
        defer allocator.free(results);

        try std.testing.expect(results.len >= 1);
        try std.testing.expectEqual(@as(u64, 0), results[0].index);
        if (results.len > 1) {
            try std.testing.expect(results[0].score <= results[1].score);
        }
    }

    {
        var reopened = try Db.open(path, false);
        defer reopened.close();

        try std.testing.expectEqual(@as(u16, 4), reopened.getDimension());
        try std.testing.expectEqual(@as(u64, 2), reopened.getRowCount());
    }
}

test "Db handles large dimensional embeddings" {
    const path = "test_db_large_temp.wdbx";
    std.fs.cwd().deleteFile(path) catch {};
    defer std.fs.cwd().deleteFile(path) catch {};

    var db = try Db.open(path, true);
    defer db.close();

    const dim: u16 = 1536;
    try db.init(dim);

    const allocator = std.testing.allocator;
    const vector = try allocator.alloc(f32, dim);
    defer allocator.free(vector);

    for (vector, 0..) |*val, idx| {
        val.* = @as(f32, @floatFromInt(idx % 11)) / 11.0;
    }

    _ = try db.addEmbedding(vector);

    const results = try db.search(vector, 1, allocator);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqual(@as(u64, 0), results[0].index);
    try std.testing.expect(results[0].score <= 1e-5);
}
