//! WDBX Database Storage Format v2
//!
//! Enhanced persistence layer with industry-standard features:
//! - CRC32 checksums for data integrity
//! - Bloom filters for fast negative lookups
//! - LZ4-style compression for vector data
//! - Incremental backup support
//! - Point-in-time recovery (PITR) capabilities
//! - Write-ahead logging (WAL) compatibility
//!
//! ## File Format v2
//!
//! ```
//! +------------------+
//! | File Header      | 64 bytes
//! +------------------+
//! | Metadata Block   | Variable
//! +------------------+
//! | Bloom Filter     | Variable (optional)
//! +------------------+
//! | Vector Index     | Variable
//! +------------------+
//! | Vector Data      | Variable
//! +------------------+
//! | Footer/Checksum  | 32 bytes
//! +------------------+
//! ```

const std = @import("std");
const database = @import("../database.zig");

// Sub-modules
pub const format = @import("format.zig");
pub const integrity = @import("integrity.zig");
pub const compression = @import("compression.zig");
pub const writer = @import("writer.zig");
pub const reader = @import("reader.zig");
pub const wal = @import("wal.zig");

// ============================================================================
// Re-export all public symbols for backward compatibility
// ============================================================================

// --- format.zig ---
pub const MAGIC = format.MAGIC;
pub const FORMAT_VERSION = format.FORMAT_VERSION;
pub const MIN_READ_VERSION = format.MIN_READ_VERSION;
pub const BlockType = format.BlockType;
pub const CompressionType = format.CompressionType;
pub const FileHeader = format.FileHeader;
pub const HeaderFlags = format.HeaderFlags;
pub const DistanceMetric = format.DistanceMetric;
pub const FileFooter = format.FileFooter;

// --- integrity.zig ---
pub const Crc32 = integrity.Crc32;
pub const BloomFilter = integrity.BloomFilter;

// --- compression.zig ---
pub const deltaEncode = compression.deltaEncode;
pub const deltaDecode = compression.deltaDecode;
pub const quantizeVectors = compression.quantizeVectors;
pub const QuantizedVectors = compression.QuantizedVectors;

// --- wal.zig ---
pub const WalEntry = wal.WalEntry;
pub const WalEntryType = wal.WalEntryType;
pub const WalWriter = wal.WalWriter;

// --- writer.zig ---
pub const saveDatabaseV2 = writer.saveDatabaseV2;
pub const saveDatabaseWithIndex = writer.saveDatabaseWithIndex;

// --- reader.zig ---
pub const loadDatabaseV2 = reader.loadDatabaseV2;

// ============================================================================
// Shared types used by both reader and writer
// ============================================================================

pub const StorageV2Error = error{
    InvalidMagic,
    UnsupportedVersion,
    InvalidChecksum,
    CorruptedData,
    InvalidFooter,
    InvalidBloomFilter,
    TruncatedData,
    OutOfMemory,
};

pub const StorageV2Config = struct {
    /// Enable bloom filter for fast ID lookups
    enable_bloom_filter: bool = true,
    /// Expected number of vectors (for bloom filter sizing)
    expected_vectors: usize = 100_000,
    /// Bloom filter false positive rate
    bloom_fp_rate: f64 = 0.01,
    /// Enable vector quantization for compression
    enable_quantization: bool = false,
    /// Write buffer size
    write_buffer_size: usize = 256 * 1024,
    /// Verify checksums on read
    verify_checksums: bool = true,
    /// Include index in file
    include_index: bool = true,
};

// ============================================================================
// HNSW Graph Persistence
// ============================================================================

/// Serialised HNSW graph data passed to the save path.
pub const HnswGraphData = struct {
    /// Entry point node ID.
    entry_point: u32,
    /// Maximum layer in the graph.
    max_layer: u32,
    /// Per-node neighbor lists.  Outer slice is indexed by node id,
    /// inner slice contains neighbor node ids.
    neighbors: []const []const u32,
};

/// Free an HnswGraphData that was returned by readHnswBlock.
pub fn freeHnswGraphData(allocator: std.mem.Allocator, h: HnswGraphData) void {
    for (h.neighbors) |nbrs| allocator.free(@constCast(nbrs));
    allocator.free(@constCast(h.neighbors));
}

// ============================================================================
// Unified Storage API (v2)
// ============================================================================

/// Unified save function - always uses v2 format
pub fn saveDatabase(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
) !void {
    return saveDatabaseV2(allocator, db, path, .{});
}

/// Unified save function with custom config - always uses v2 format
pub fn saveDatabaseWithConfig(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
    config: StorageV2Config,
) !void {
    return saveDatabaseV2(allocator, db, path, config);
}

/// Unified load function - v2 format
pub fn loadDatabase(
    allocator: std.mem.Allocator,
    path: []const u8,
) !database.Database {
    return loadDatabaseWithConfig(allocator, path, .{});
}

/// Unified load function with config - v2 format
pub fn loadDatabaseWithConfig(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: StorageV2Config,
) !database.Database {
    return loadDatabaseV2(allocator, path, config);
}

// ============================================================================
// Tests
// ============================================================================

test "crc32 basic" {
    const data = "hello world";
    const crc = Crc32.compute(data);
    try std.testing.expect(crc != 0);

    // Same data should produce same CRC
    const crc2 = Crc32.compute(data);
    try std.testing.expectEqual(crc, crc2);

    // Different data should produce different CRC
    const crc3 = Crc32.compute("hello world!");
    try std.testing.expect(crc != crc3);
}

test "bloom filter" {
    const allocator = std.testing.allocator;

    var bloom = try BloomFilter.init(allocator, 1000, 0.01);
    defer bloom.deinit();

    // Add some IDs
    bloom.add(100);
    bloom.add(200);
    bloom.add(300);

    // Should find added IDs
    try std.testing.expect(bloom.mightContain(100));
    try std.testing.expect(bloom.mightContain(200));
    try std.testing.expect(bloom.mightContain(300));

    // May or may not find non-added IDs (false positives possible)
    // But should have low false positive rate

    // Test serialization roundtrip
    const serialized = try bloom.serialize(allocator);
    defer allocator.free(serialized);

    var bloom2 = try BloomFilter.deserialize(allocator, serialized);
    defer bloom2.deinit();

    try std.testing.expect(bloom2.mightContain(100));
    try std.testing.expect(bloom2.mightContain(200));
}

test "delta encoding" {
    const allocator = std.testing.allocator;

    const ids = [_]u64{ 100, 105, 110, 200, 250 };
    const encoded = try deltaEncode(&ids, allocator);
    defer allocator.free(encoded);

    try std.testing.expectEqual(@as(u64, 100), encoded[0]);
    try std.testing.expectEqual(@as(u64, 5), encoded[1]);
    try std.testing.expectEqual(@as(u64, 5), encoded[2]);
    try std.testing.expectEqual(@as(u64, 90), encoded[3]);

    const decoded = try deltaDecode(encoded, allocator);
    defer allocator.free(decoded);

    try std.testing.expectEqualSlices(u64, &ids, decoded);
}

test "header serialization" {
    const header = FileHeader{
        .version = FORMAT_VERSION,
        .vector_count = 1000,
        .dimension = 128,
        .distance_metric = .cosine,
        .flags = .{ .has_bloom_filter = true, .compressed = true },
        .uuid = .{
            0x00, 0x01, 0x02, 0x03,
            0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0A, 0x0B,
            0x0C, 0x0D, 0x0E, 0x0F,
        },
        .reserved = .{
            0xF0, 0xF1, 0xF2, 0xF3,
            0xF4, 0xF5, 0xF6, 0xF7,
            0xF8, 0xF9, 0xFA, 0xFB,
            0xFC, 0xFD, 0xFE, 0xFF,
        },
    };

    const bytes = header.serialize();
    const restored = try FileHeader.deserialize(bytes);

    try std.testing.expectEqual(header.version, restored.version);
    try std.testing.expectEqual(header.vector_count, restored.vector_count);
    try std.testing.expectEqual(header.dimension, restored.dimension);
    try std.testing.expectEqual(header.distance_metric, restored.distance_metric);
    try std.testing.expect(restored.flags.has_bloom_filter);
    try std.testing.expect(restored.flags.compressed);
    try std.testing.expectEqualSlices(u8, &header.uuid, &restored.uuid);
    try std.testing.expectEqualSlices(
        u8,
        header.reserved[0..format.HEADER_RESERVED_STORED_SIZE],
        restored.reserved[0..format.HEADER_RESERVED_STORED_SIZE],
    );
    try std.testing.expectEqualSlices(
        u8,
        &[_]u8{0} ** (header.reserved.len - format.HEADER_RESERVED_STORED_SIZE),
        restored.reserved[format.HEADER_RESERVED_STORED_SIZE..],
    );
    try std.testing.expectEqualSlices(
        u8,
        &header.uuid,
        bytes[format.HEADER_UUID_OFFSET .. format.HEADER_UUID_OFFSET + format.HEADER_UUID_SIZE],
    );
    try std.testing.expectEqualSlices(
        u8,
        header.reserved[0..format.HEADER_RESERVED_STORED_SIZE],
        bytes[format.HEADER_RESERVED_OFFSET .. format.HEADER_RESERVED_OFFSET + format.HEADER_RESERVED_STORED_SIZE],
    );
}

test "footer serialization" {
    const footer = FileFooter{
        .header_checksum = 0x12345678,
        .data_checksum = 0xDEADBEEF,
        .file_size = 1024 * 1024,
        .num_blocks = 5,
    };

    const bytes = footer.serialize();
    const restored = try FileFooter.deserialize(bytes);

    try std.testing.expectEqual(footer.header_checksum, restored.header_checksum);
    try std.testing.expectEqual(footer.data_checksum, restored.data_checksum);
    try std.testing.expectEqual(footer.file_size, restored.file_size);
    try std.testing.expectEqual(footer.num_blocks, restored.num_blocks);
}

test "vector quantization" {
    const allocator = std.testing.allocator;

    // Create some test vectors
    var vectors: [3][]f32 = undefined;
    vectors[0] = try allocator.alloc(f32, 4);
    defer allocator.free(vectors[0]);
    vectors[1] = try allocator.alloc(f32, 4);
    defer allocator.free(vectors[1]);
    vectors[2] = try allocator.alloc(f32, 4);
    defer allocator.free(vectors[2]);

    vectors[0][0] = 0.0;
    vectors[0][1] = 0.5;
    vectors[0][2] = 1.0;
    vectors[0][3] = -1.0;
    vectors[1][0] = 0.1;
    vectors[1][1] = 0.6;
    vectors[1][2] = 0.9;
    vectors[1][3] = -0.5;
    vectors[2][0] = 0.2;
    vectors[2][1] = 0.4;
    vectors[2][2] = 0.8;
    vectors[2][3] = 0.0;

    const slices: []const []const f32 = &.{ vectors[0], vectors[1], vectors[2] };
    var quantized = try quantizeVectors(slices, allocator);
    defer quantized.deinit();

    try std.testing.expectEqual(@as(usize, 4), quantized.dimension);
    try std.testing.expectEqual(@as(usize, 12), quantized.data.len); // 3 vectors * 4 dims

    // Small vectors include per-dimension scale/offset overhead, so expect a modest gain.
    const ratio = quantized.compressionRatio(3);
    try std.testing.expect(ratio > 1.0 and ratio < 2.0);
}

test "loadDatabaseV2 detects payload checksum corruption" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/corrupt.wdbx",
        .{tmp.sub_path},
    );
    defer allocator.free(path);

    var db = try database.Database.init(allocator, "checksum-db");
    defer db.deinit();
    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, "meta");

    try saveDatabaseV2(allocator, &db, path, .{});

    var bytes = try reader.readFileBytesForTest(allocator, path);
    defer allocator.free(bytes);
    bytes[format.HEADER_SIZE + 2] ^= 0xFF;
    try reader.writeFileBytesForTest(allocator, path, bytes);

    const loaded = loadDatabaseV2(allocator, path, .{ .verify_checksums = true });
    try std.testing.expectError(error.InvalidChecksum, loaded);
}

test "loadDatabaseV2 detects footer file_size mismatch" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/bad-size.wdbx",
        .{tmp.sub_path},
    );
    defer allocator.free(path);

    var db = try database.Database.init(allocator, "size-db");
    defer db.deinit();
    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, null);

    try saveDatabaseV2(allocator, &db, path, .{});

    var bytes = try reader.readFileBytesForTest(allocator, path);
    defer allocator.free(bytes);
    const footer_start = bytes.len - format.FOOTER_SIZE;
    const file_size_bytes: *const [8]u8 = @ptrCast(bytes[footer_start + 16 .. footer_start + 24].ptr);
    const bad_size = std.mem.readInt(u64, file_size_bytes, .little) + 1;
    const file_size_out: *[8]u8 = @ptrCast(bytes[footer_start + 16 .. footer_start + 24].ptr);
    std.mem.writeInt(u64, file_size_out, bad_size, .little);
    try reader.writeFileBytesForTest(allocator, path, bytes);

    const loaded = loadDatabaseV2(allocator, path, .{ .verify_checksums = true });
    try std.testing.expectError(error.CorruptedData, loaded);
}

test "loadDatabaseV2 detects vector count mismatch without checksum validation" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/count-mismatch.wdbx",
        .{tmp.sub_path},
    );
    defer allocator.free(path);

    var db = try database.Database.init(allocator, "count-db");
    defer db.deinit();
    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, null);

    try saveDatabaseV2(allocator, &db, path, .{});

    var bytes = try reader.readFileBytesForTest(allocator, path);
    defer allocator.free(bytes);
    std.mem.writeInt(u64, bytes[24..32], 2, .little);
    try reader.writeFileBytesForTest(allocator, path, bytes);

    const loaded = loadDatabaseV2(allocator, path, .{ .verify_checksums = false });
    try std.testing.expectError(error.CorruptedData, loaded);
}

test "loadDatabaseV2 detects dimension mismatch without checksum validation" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/dim-mismatch.wdbx",
        .{tmp.sub_path},
    );
    defer allocator.free(path);

    var db = try database.Database.init(allocator, "dim-db");
    defer db.deinit();
    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, null);

    try saveDatabaseV2(allocator, &db, path, .{});

    var bytes = try reader.readFileBytesForTest(allocator, path);
    defer allocator.free(bytes);
    std.mem.writeInt(u32, bytes[32..36], 2, .little);
    try reader.writeFileBytesForTest(allocator, path, bytes);

    const loaded = loadDatabaseV2(allocator, path, .{ .verify_checksums = false });
    try std.testing.expectError(error.CorruptedData, loaded);
}

test "loadDatabaseV2 detects missing bloom block when flag is set" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/bloom-missing.wdbx",
        .{tmp.sub_path},
    );
    defer allocator.free(path);

    var db = try database.Database.init(allocator, "bloom-db");
    defer db.deinit();
    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, null);

    try saveDatabaseV2(allocator, &db, path, .{ .enable_bloom_filter = false });

    var bytes = try reader.readFileBytesForTest(allocator, path);
    defer allocator.free(bytes);
    bytes[6] |= 0x01; // force has_bloom_filter flag
    try reader.writeFileBytesForTest(allocator, path, bytes);

    const loaded = loadDatabaseV2(allocator, path, .{ .verify_checksums = false });
    try std.testing.expectError(error.InvalidBloomFilter, loaded);
}

test {
    std.testing.refAllDecls(@This());
}
