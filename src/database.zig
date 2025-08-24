//! WDBX-AI Vector Database
//!
//! This module provides a simple file-based vector database for storing and
//! searching high-dimensional embeddings. The database uses a custom binary
//! format with a fixed 4KB header followed by densely packed float32 vectors.
//!
//! ## File Format
//!
//! The WDBX-AI format consists of:
//! - **Header (4096 bytes)**: Contains metadata including magic bytes, version,
//!   row count, dimensionality, and various offset pointers
//! - **Records Section**: Densely packed float32 vectors, each record is
//!   `dim * sizeof(f32)` bytes
//!
//! ## Usage
//!
//! ```zig
//! // Create or open a database
//! var db = try Db.open("vectors.wdbx", true);
//! defer db.close();
//!
//! // Initialize with embedding dimension
//! try db.init(384);
//!
//! // Add vectors
//! const embedding = [_]f32{0.1, 0.2, 0.3, ...};
//! const row_id = try db.addEmbedding(&embedding);
//!
//! // Search for similar vectors
//! const query = [_]f32{0.15, 0.25, 0.35, ...};
//! const results = try db.search(&query, 10, allocator);
//! defer allocator.free(results);
//! ```

const std = @import("std");
const abi = @import("root.zig");
const core = @import("core/mod.zig");

/// Re-export core types for convenience
pub const Allocator = core.Allocator;

/// Database specific error types
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
} || core.Error;

/// Magic identifier for WDBX-AI files (7 bytes + NUL)
pub const MAGIC: [8]u8 = "WDBXAI\x00\x00".*;

/// Current file format version
pub const FORMAT_VERSION: u16 = 1;

/// Default page size for file operations
pub const DEFAULT_PAGE_SIZE: u32 = 4096;

/// File-header fixed at 4 KiB (4096 bytes)
///
/// The header contains all metadata needed to interpret the database file.
/// It uses individual magic bytes for easier debugging and validation.
pub const WdbxHeader = struct {
    /// Magic byte 0: 'W'
    magic0: u8,
    /// Magic byte 1: 'D'
    magic1: u8,
    /// Magic byte 2: 'B'
    magic2: u8,
    /// Magic byte 3: 'X'
    magic3: u8,
    /// Magic byte 4: 'A'
    magic4: u8,
    /// Magic byte 5: 'I'
    magic5: u8,
    /// Magic byte 6: '\0'
    magic6: u8,
    /// Format version number
    version: u16,
    /// Number of active rows in the database
    row_count: u64,
    /// Embedding dimensionality (must be consistent across all vectors)
    dim: u16,
    /// Logical page size for file operations (default 4096)
    page_size: u32,
    /// Offset to UTF-8 JSON schema block (reserved for future use)
    schema_off: u64,
    /// Offset to vector index block (reserved for future use)
    index_off: u64,
    /// Offset to first record row
    records_off: u64,
    /// Offset to free-list block (reserved for future use)
    freelist_off: u64,
    /// Reserved space to pad header to exactly 4096 bytes
    _reserved: [4072]u8,

    /// Validate header magic bytes
    pub fn validateMagic(self: *const WdbxHeader) bool {
        return self.magic0 == 'W' and
            self.magic1 == 'D' and
            self.magic2 == 'B' and
            self.magic3 == 'X' and
            self.magic4 == 'A' and
            self.magic5 == 'I' and
            self.magic6 == 0;
    }

    /// Create a new header with default values
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
            .schema_off = DEFAULT_PAGE_SIZE, // schema follows header by default
            .index_off = 0,
            .records_off = 0,
            .freelist_off = 0,
            ._reserved = [_]u8{0} ** 4072,
        };
    }
};

/// Database handle for WDBX-AI vector database operations
///
/// This structure holds the file handle and cached header for efficient
/// database operations. The database supports basic CRUD operations on
/// high-dimensional float32 vectors with linear search capabilities.
///
/// **Lifetime**: Allocate with `open()`, release with `close()`
pub const Db = struct {
    /// File handle for the database file
    file: std.fs.File,
    /// Cached header for fast access to metadata
    header: WdbxHeader,
    /// Allocator used for database instance management
    allocator: std.mem.Allocator,
    /// Read buffer for optimized file operations
    read_buffer: []u8,
    /// Statistics tracking
    stats: DbStats = .{},

    /// Comprehensive error set for database operations - improved based on Zig best practices
    pub const Error = error{
        /// Database has already been initialized with a dimension
        AlreadyInitialized,
        /// Vector dimension doesn't match database dimension
        DimensionMismatch,
        /// Database is in an invalid state (e.g., not initialized)
        InvalidState,
        /// Out of memory during allocation
        OutOfMemory,
        /// File is busy or locked
        FileBusy,
        /// Unexpected end of stream during read
        EndOfStream,
        /// Invalid magic bytes in file header
        InvalidMagic,
        /// Unsupported file format version
        UnsupportedVersion,
        /// Corrupted database file
        CorruptedDatabase,
    } || std.fs.File.SeekError || std.fs.File.WriteError ||
        std.fs.File.ReadError || std.fs.File.OpenError;

    /// Initialize a brand-new database after creation
    ///
    /// Sets the embedding dimensionality and prepares the records section.
    /// This must be called exactly once on an empty database where both
    /// `row_count == 0` and `dim == 0`.
    ///
    /// **Parameters:**
    /// - `dim`: The dimensionality for all vectors (must be > 0)
    ///
    /// **Returns:** `Error!void`
    ///
    /// **Errors:**
    /// - `AlreadyInitialized`: Database already has a dimension set
    /// - `DimensionMismatch`: Dimension is zero or exceeds maximum
    pub fn init(self: *Db, dim: u16) Error!void {
        // Fast path checks
        if (self.header.dim != 0 or self.header.row_count != 0) {
            return Error.AlreadyInitialized;
        }
        if (dim == 0 or dim > 4096) { // Reasonable upper limit
            return Error.DimensionMismatch;
        }

        self.header.dim = dim;
        self.header.records_off = DEFAULT_PAGE_SIZE; // first page after header

        // Ensure the file is at least one empty page so the first append works
        self.file.setEndPos(DEFAULT_PAGE_SIZE) catch |err| {
            return switch (err) {
                error.NonResizable => Error.InvalidState,
                else => |e| e,
            };
        };

        // Write updated header back to file
        try self.writeHeader();

        self.stats.initialization_count += 1;
    }

    /// Write header to file at offset 0
    fn writeHeader(self: *Db) Error!void {
        try self.file.seekTo(0);
        const header_bytes = std.mem.asBytes(&self.header);
        try self.file.writeAll(header_bytes);
        try self.file.sync(); // Ensure header is written to disk
    }

    /// Read header from file at offset 0
    fn readHeader(self: *Db) Error!void {
        try self.file.seekTo(0);
        const header_bytes = std.mem.asBytes(&self.header);
        _ = try self.file.readAll(header_bytes);
    }

    /// Append a single embedding vector to the database
    ///
    /// The vector is appended to the end of the records section and the
    /// row count is incremented. The file is automatically grown as needed,
    /// extending to the next page boundary for efficiency.
    ///
    /// **Parameters:**
    /// - `embedding`: Slice of f32 values representing the vector
    ///
    /// **Returns:** The 0-based row index of the inserted vector
    ///
    /// **Errors:**
    /// - `DimensionMismatch`: Vector length doesn't match database dimension
    /// - `InvalidState`: Database not initialized (call `init()` first)
    pub fn addEmbedding(self: *Db, embedding: []const f32) Error!u64 {
        // Fast path validation
        if (embedding.len != self.header.dim) {
            return Error.DimensionMismatch;
        }
        if (self.header.records_off == 0) {
            return Error.InvalidState; // not initialized
        }

        const row_index = self.header.row_count;
        const record_size: u64 = @as(u64, self.header.dim) * @sizeOf(f32);
        const offset: u64 = self.header.records_off + row_index * record_size;

        // Grow the file as needed with optimized allocation
        const needed_size = offset + record_size;
        const current_size = try self.file.getEndPos();
        if (needed_size > current_size) {
            // Extend to next page multiple for efficiency
            const page_size: u64 = self.header.page_size;
            const new_size = ((needed_size + page_size - 1) / page_size) * page_size;
            self.file.setEndPos(new_size) catch |err| {
                return switch (err) {
                    error.NonResizable => Error.InvalidState,
                    else => |e| e,
                };
            };
        }

        // Write the embedding
        try self.file.seekTo(offset);
        try self.file.writeAll(std.mem.sliceAsBytes(embedding));

        // Update row_count in header
        self.header.row_count += 1;

        // Write updated header back to file
        try self.writeHeader();

        self.stats.write_count += 1;
        return row_index;
    }

    /// Batch add multiple embeddings for better performance
    pub fn addEmbeddingsBatch(self: *Db, embeddings: []const []const f32) Error![]u64 {
        const indices = try self.allocator.alloc(u64, embeddings.len);
        errdefer self.allocator.free(indices);

        for (embeddings, 0..) |embedding, i| {
            indices[i] = try self.addEmbedding(embedding);
        }

        return indices;
    }

    /// Perform naive linear search for nearest vectors
    ///
    /// Searches all vectors in the database using squared Euclidean distance
    /// and returns the top-k closest matches sorted by ascending distance.
    /// This is a brute-force O(n) search suitable for small to medium datasets.
    ///
    /// **Parameters:**
    /// - `query`: Query vector to search for
    /// - `top_k`: Maximum number of results to return
    /// - `allocator`: Allocator for result array
    ///
    /// **Returns:** Owned slice of `Result` structs, sorted by ascending score
    ///
    /// **Errors:**
    /// - `DimensionMismatch`: Query vector length doesn't match database dimension
    /// - `OutOfMemory`: Failed to allocate memory for results
    ///
    /// **Note:** Caller owns the returned slice and must free it
    pub fn search(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator) Error![]Result {
        const start_time = std.time.microTimestamp();
        defer {
            self.stats.search_count += 1;
            self.stats.total_search_time_us += @intCast(std.time.microTimestamp() - start_time);
        }

        // Fast path validation
        if (query.len != self.header.dim) {
            return Error.DimensionMismatch;
        }

        const row_count = self.header.row_count;
        if (row_count == 0) {
            return allocator.alloc(Result, 0);
        }

        const row_count_usize: usize = @intCast(row_count);
        const record_size: u64 = @as(u64, self.header.dim) * @sizeOf(f32);

        // Allocate all results upfront
        var all = try allocator.alloc(Result, row_count_usize);
        defer allocator.free(all);

        // Use read buffer for better performance
        const buf_size = @min(record_size, self.read_buffer.len);
        const buf = self.read_buffer[0..buf_size];

        for (0..row_count_usize) |row| {
            const offset: u64 = self.header.records_off + @as(u64, row) * record_size;
            try self.file.seekTo(offset);
            _ = try self.file.readAll(buf);

            // Calculate squared Euclidean distance
            var dist: f32 = 0;
            const row_data = std.mem.bytesAsSlice(f32, buf);

            // Unroll loop for better performance on small dimensions
            if (self.header.dim <= 8) {
                for (row_data, query) |val, q| {
                    const diff = val - q;
                    dist += diff * diff;
                }
            } else {
                // Use SIMD-friendly loop for larger dimensions
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
            }

            all[row] = .{ .index = @intCast(row), .score = dist };
        }

        // Sort ascending by score
        std.mem.sort(Result, all, {}, comptime Result.lessThanAsc);

        // Return top-k results
        const result_len = @min(top_k, all.len);
        const out = try allocator.alloc(Result, result_len);
        @memcpy(out, all[0..result_len]);

        return out;
    }

    /// Search result containing vector index and similarity score
    pub const Result = struct {
        /// 0-based row index of the vector in the database
        index: u64,
        /// Squared Euclidean distance (lower is more similar)
        score: f32,

        /// Comparison function for sorting results by ascending score
        pub fn lessThanAsc(_: void, a: Result, b: Result) bool {
            return a.score < b.score;
        }
    };

    /// Database statistics
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

    /// Open an existing WDBX-AI file or create a new one
    ///
    /// Opens a database file for read/write operations. If the file doesn't
    /// exist and `create_if_missing` is true, creates a new empty database
    /// with default header values.
    ///
    /// **Parameters:**
    /// - `path`: File system path to the .wdbx database file
    /// - `create_if_missing`: Whether to create the file if it doesn't exist
    ///
    /// **Returns:** Owned pointer to `Db` instance
    ///
    /// **Errors:**
    /// - `FileNotFound`: File doesn't exist and `create_if_missing` is false
    /// - `InvalidMagic`: File exists but has invalid magic bytes
    /// - `OutOfMemory`: Failed to allocate database instance
    ///
    /// **Note:** Caller must call `close()` to free resources
    pub fn open(path: []const u8, create_if_missing: bool) Error!*Db {
        const allocator = std.heap.page_allocator;

        const file = std.fs.cwd().openFile(path, .{ .mode = .read_write }) catch |err| blk: {
            if (create_if_missing and err == error.FileNotFound) {
                // Create a blank file and write an empty header
                const new_file = try std.fs.cwd().createFile(path, .{ .read = true, .truncate = true });

                // Create default header
                const hdr = WdbxHeader.createDefault();
                try new_file.writeAll(std.mem.asBytes(&hdr));

                // Ensure the file is at least one page
                new_file.setEndPos(DEFAULT_PAGE_SIZE) catch |set_end_pos_err| {
                    return switch (set_end_pos_err) {
                        error.NonResizable => Error.InvalidState,
                        else => |e| e,
                    };
                };
                break :blk new_file;
            } else {
                return err;
            }
        };

        // Allocate Db on the page allocator
        const self = try allocator.create(Db);
        errdefer allocator.destroy(self);

        // Allocate read buffer for optimized file operations
        const read_buffer = try allocator.alloc(u8, DEFAULT_PAGE_SIZE);
        errdefer allocator.free(read_buffer);

        self.* = .{
            .file = file,
            .header = undefined,
            .allocator = allocator,
            .read_buffer = read_buffer,
        };

        // Read and validate header
        try self.readHeader();

        if (!self.header.validateMagic()) {
            self.file.close();
            allocator.free(self.read_buffer);
            allocator.destroy(self);
            return Error.InvalidMagic;
        }

        if (self.header.version != FORMAT_VERSION) {
            self.file.close();
            allocator.free(self.read_buffer);
            allocator.destroy(self);
            return Error.UnsupportedVersion;
        }

        return self;
    }

    /// Flush and close all database resources
    ///
    /// Closes the file handle and deallocates the database instance.
    /// After calling this function, the `Db` pointer is invalid and
    /// must not be used.
    pub fn close(self: *Db) void {
        self.file.close();
        self.allocator.free(self.read_buffer);
        self.allocator.destroy(self);
    }

    /// Get database statistics
    pub fn getStats(self: *const Db) DbStats {
        return self.stats;
    }

    /// Get current row count
    pub fn getRowCount(self: *const Db) u64 {
        return self.header.row_count;
    }

    /// Get vector dimension
    pub fn getDimension(self: *const Db) u16 {
        return self.header.dim;
    }
};
