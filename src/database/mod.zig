//! Database module for WDBX-AI Vector Database
//!
//! This module provides the core database functionality including:
//! - Vector storage and retrieval
//! - HNSW indexing for fast similarity search
//! - Transaction support
//! - Persistence and recovery

const std = @import("std");
const core = @import("../core/mod.zig");
const simd = @import("../simd/mod.zig");

// Sub-modules
pub const enhanced_db = @import("enhanced_db.zig");

// Re-export main database implementation
pub const DatabaseConfig = enhanced_db.DatabaseConfig;
pub const DatabaseRecord = enhanced_db.DatabaseRecord;
pub const DatabaseIndex = enhanced_db.DatabaseIndex;
pub const EnhancedDatabase = enhanced_db.EnhancedDatabase;

// Aliases for compatibility
pub const Db = EnhancedDatabase;
pub const DbError = error{
    DatabaseError,
    NotFound,
    AlreadyExists,
    InvalidOperation,
    CorruptedData,
    OutOfMemory,
};

// Result type for database operations
pub const Result = struct {
    id: []const u8,
    score: f32,
    vector: []const f32,
    metadata: ?std.StringHashMap(MetadataValue),
};

// Metadata value types
pub const MetadataValue = union(enum) {
    string: []const u8,
    integer: i64,
    float: f64,
    boolean: bool,
};

// Vector data structure
pub const VectorData = struct {
    id: []const u8,
    vector: []const f32,
    metadata: ?std.StringHashMap(MetadataValue),
};

// Search options
pub const SearchOptions = struct {
    metric: Metric = .euclidean,
    filter: ?*const fn (metadata: std.StringHashMap(MetadataValue)) bool = null,
};

// Index types
pub const IndexType = enum {
    flat,
    hnsw,
    ivf,
};

// WDBX file header
pub const WdbxHeader = struct {
    magic: [4]u8 = "WDBX".*,
    version: u32 = 0x00020000, // 2.0.0
    vector_dim: u32,
    vector_count: u32,
    index_type: IndexType,
    flags: u32,
};

// Global state
var initialized = false;
var global_allocator: ?std.mem.Allocator = null;

/// Initialize the database module
pub fn init(allocator: std.mem.Allocator) !void {
    if (initialized) return;
    
    global_allocator = allocator;
    initialized = true;
    
    // Initialize sub-modules if needed
    core.getLogger().info(@src(), "Database module initialized", .{});
}

/// Cleanup the database module
pub fn deinit() void {
    if (!initialized) return;
    
    initialized = false;
    global_allocator = null;
    
    core.getLogger().info(@src(), "Database module deinitialized", .{});
}

/// Create a new database instance with default configuration
pub fn create(allocator: std.mem.Allocator, path: []const u8) !*Db {
    const config = core.getConfig();
    
    const db_config = DatabaseConfig{
        .max_file_size = @as(usize, @intCast(config.getInt("database.max_file_size") orelse 1024 * 1024 * 1024)),
        .cache_size = @as(usize, @intCast(config.getInt("database.cache_size") orelse 100 * 1024 * 1024)),
        .enable_compression = config.getBool("database.enable_compression") orelse false,
        .enable_logging = config.getBool("database.enable_logging") orelse true,
    };
    
    const db = try allocator.create(EnhancedDatabase);
    db.* = try EnhancedDatabase.init(allocator, path, db_config);
    return db;
}

/// Open an existing database
pub fn open(allocator: std.mem.Allocator, path: []const u8) !*Db {
    return try create(allocator, path);
}

/// Database statistics
pub const Stats = struct {
    total_vectors: usize,
    total_size: usize,
    index_size: usize,
    cache_hits: usize,
    cache_misses: usize,
    avg_search_time_ms: f64,
    
    pub fn format(self: Stats, writer: anytype) !void {
        try writer.print(
            \\Database Statistics:
            \\  Total vectors: {d}
            \\  Total size: {s}
            \\  Index size: {s}
            \\  Cache hit rate: {d:.1}%
            \\  Avg search time: {d:.2}ms
        , .{
            self.total_vectors,
            formatBytes(self.total_size),
            formatBytes(self.index_size),
            if (self.cache_hits + self.cache_misses > 0)
                @as(f64, @floatFromInt(self.cache_hits)) * 100.0 / @as(f64, @floatFromInt(self.cache_hits + self.cache_misses))
            else
                0.0,
            self.avg_search_time_ms,
        });
    }
    
    fn formatBytes(bytes: usize) []const u8 {
        const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB" };
        var value = @as(f64, @floatFromInt(bytes));
        var unit_idx: usize = 0;
        
        while (value >= 1024 and unit_idx < units.len - 1) : (unit_idx += 1) {
            value /= 1024;
        }
        
        var buf: [32]u8 = undefined;
        return std.fmt.bufPrint(&buf, "{d:.2} {s}", .{ value, units[unit_idx] }) catch "N/A";
    }
};

/// Vector similarity metrics
pub const Metric = enum {
    euclidean,
    cosine,
    dot_product,
    manhattan,
    
    /// Calculate distance between two vectors
    pub fn distance(self: Metric, a: []const f32, b: []const f32) f32 {
        std.debug.assert(a.len == b.len);
        
        return switch (self) {
            .euclidean => simd.VectorOps.euclideanDistance(a, b),
            .cosine => simd.VectorOps.cosineDistance(a, b),
            .dot_product => simd.VectorOps.dotProduct(a, b),
            .manhattan => manhattanDistance(a, b),
        };
    }
    
    fn manhattanDistance(a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0;
        for (a, b) |x, y| {
            sum += @abs(x - y);
        }
        return sum;
    }
};

/// Batch operations for efficiency
pub const BatchOps = struct {
    /// Insert multiple records at once
    pub fn insertBatch(
        db: *Db,
        records: []const []const u8,
    ) !void {
        const logger = core.getLogger();
        var perf = core.logging.PerfLogger.begin(&logger, "batch_insert");
        defer perf.end();
        
        for (records) |record| {
            _ = try db.writeRecord(record);
        }
        
        logger.info(@src(), "Inserted {d} records in batch", .{records.len});
    }
    
    /// Read multiple records at once
    pub fn readBatch(
        db: *Db,
        ids: []const u64,
        allocator: std.mem.Allocator,
    ) ![]?DatabaseRecord {
        var results = try allocator.alloc(?DatabaseRecord, ids.len);
        
        const logger = core.getLogger();
        var perf = core.logging.PerfLogger.begin(&logger, "batch_read");
        defer perf.end();
        
        for (ids, 0..) |id, i| {
            results[i] = try db.readRecord(id);
        }
        
        logger.info(@src(), "Read {d} records in batch", .{ids.len});
        return results;
    }
};

/// Database optimization utilities
pub const Optimizer = struct {
    /// Optimize database for better performance
    pub fn optimize(db: *Db) !void {
        const logger = core.getLogger();
        logger.info(@src(), "Starting database optimization", .{});
        
        // Compact storage
        try db.compact();
        
        logger.info(@src(), "Database optimization complete", .{});
    }
};

/// Import/Export utilities
pub const ImportExport = struct {
    /// Export database records to JSON format
    pub fn exportJson(db: *Db, writer: anytype, record_ids: []const u64) !void {
        try writer.writeAll("{\n");
        try writer.writeAll("  \"version\": \"2.0.0\",\n");
        try writer.writeAll("  \"records\": [\n");
        
        var first = true;
        for (record_ids) |id| {
            if (try db.readRecord(id)) |record| {
                if (!first) try writer.writeAll(",\n");
                first = false;
                
                try writer.writeAll("    {\n");
                try writer.print("      \"id\": {d},\n", .{record.id});
                try writer.print("      \"timestamp\": {d},\n", .{record.timestamp});
                try writer.print("      \"size\": {d},\n", .{record.data.len});
                try writer.print("      \"compressed\": {}", .{record.compressed});
                try writer.writeAll("\n    }");
            }
        }
        
        try writer.writeAll("\n  ]\n}\n");
    }
    
    /// Import database from JSON format
    pub fn importJson(db: *Db, reader: anytype) !void {
        // Implementation would parse JSON and insert vectors
        _ = db;
        _ = reader;
        @panic("Not implemented");
    }
    
    fn writeMetadata(writer: anytype, metadata: std.StringHashMap(MetadataValue)) !void {
        try writer.writeAll("{");
        
        var first = true;
        var iter = metadata.iterator();
        while (iter.next()) |entry| {
            if (!first) try writer.writeAll(", ");
            first = false;
            
            try writer.print("\"{s}\": ", .{entry.key_ptr.*});
            
            switch (entry.value_ptr.*) {
                .string => |s| try writer.print("\"{s}\"", .{s}),
                .integer => |i| try writer.print("{d}", .{i}),
                .float => |f| try writer.print("{d}", .{f}),
                .boolean => |b| try writer.writeAll(if (b) "true" else "false"),
            }
        }
        
        try writer.writeAll("}");
    }
};

test "database module initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    try init(allocator);
    defer deinit();
    
    // Test basic database creation
    const db = try create(allocator, ":memory:");
    defer db.deinit();
    
    try testing.expect(db != null);
}

test "metric calculations" {
    const testing = std.testing;
    
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    
    const euclidean = Metric.euclidean.distance(&a, &b);
    const cosine = Metric.cosine.distance(&a, &b);
    const manhattan = Metric.manhattan.distance(&a, &b);
    
    try testing.expect(euclidean > 0);
    try testing.expect(cosine >= -1 and cosine <= 1);
    try testing.expectEqual(@as(f32, 9), manhattan);
}