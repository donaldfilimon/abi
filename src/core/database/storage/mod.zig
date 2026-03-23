//! WDBX storage subsystem.
//!
//! Unified write path: v3 sectioned format.
//! Unified read path: accepts v2 and v3 during migration.

const std = @import("std");
const database = @import("../database.zig");

pub const format = @import("format.zig");
pub const format_v2 = @import("format_v2.zig");
pub const integrity = @import("integrity.zig");
pub const compression = @import("compression.zig");
pub const writer = @import("writer.zig");
pub const reader = @import("reader.zig");
pub const writer_v3 = @import("writer_v3.zig");
pub const reader_v3 = @import("reader_v3.zig");
pub const wal = @import("wal.zig");

pub const MAGIC = format.MAGIC;
pub const FORMAT_VERSION = format.FORMAT_VERSION;
pub const MIN_READ_VERSION = format.MIN_READ_VERSION;
pub const HEADER_SIZE = format.HEADER_SIZE;
pub const FOOTER_SIZE = format.FOOTER_SIZE;
pub const DIRECTORY_ENTRY_SIZE = format.DIRECTORY_ENTRY_SIZE;
pub const BlockType = format_v2.BlockType;
pub const SectionType = format.SectionType;
pub const FileHeader = format.FileHeader;
pub const HeaderFlags = format.HeaderFlags;
pub const DistanceMetric = format.DistanceMetric;
pub const CompressionType = format.CompressionType;
pub const SectionDirectoryEntry = format.SectionDirectoryEntry;
pub const FileFooter = format.FileFooter;

pub const Crc32 = integrity.Crc32;
pub const BloomFilter = integrity.BloomFilter;
pub const deltaEncode = compression.deltaEncode;
pub const deltaDecode = compression.deltaDecode;
pub const quantizeVectors = compression.quantizeVectors;
pub const QuantizedVectors = compression.QuantizedVectors;
pub const WalEntry = wal.WalEntry;
pub const WalEntryType = wal.WalEntryType;
pub const WalWriter = wal.WalWriter;

pub const StorageError = error{
    InvalidMagic,
    UnsupportedVersion,
    InvalidChecksum,
    CorruptedData,
    InvalidFooter,
    InvalidBloomFilter,
    TruncatedData,
    OutOfMemory,
};
pub const StorageV2Error = StorageError;

pub const StorageConfig = struct {
    enable_bloom_filter: bool = true,
    expected_vectors: usize = 100_000,
    bloom_fp_rate: f64 = 0.01,
    enable_quantization: bool = false,
    write_buffer_size: usize = 256 * 1024,
    verify_checksums: bool = true,
    include_index: bool = true,
    include_lineage: bool = false,
    include_distributed: bool = false,
};

pub const StorageV2Config = StorageConfig;
pub const StorageV3Config = StorageConfig;

pub const HnswGraphData = struct {
    entry_point: u32,
    max_layer: u32,
    neighbors: []const []const u32,
};

pub fn freeHnswGraphData(allocator: std.mem.Allocator, h: HnswGraphData) void {
    for (h.neighbors) |nbrs| allocator.free(@constCast(nbrs));
    allocator.free(@constCast(h.neighbors));
}

pub const saveDatabaseV2 = writer.saveDatabaseV2;
pub const loadDatabaseV2 = reader.loadDatabaseV2;
pub const saveDatabaseV3 = writer_v3.saveDatabaseV3;
pub const loadDatabaseV3 = reader_v3.loadDatabaseV3;
pub const saveDatabaseWithIndex = writer_v3.saveDatabaseV3WithIndex;
pub const readFileBytesForTest = reader_v3.readFileBytesForTest;
pub const writeFileBytesForTest = reader_v3.writeFileBytesForTest;

fn readIntLe(comptime T: type, bytes: []const u8) T {
    const raw: *const [@sizeOf(T)]u8 = @ptrCast(bytes.ptr);
    return std.mem.readInt(T, raw, .little);
}

fn detectFormatVersion(allocator: std.mem.Allocator, path: []const u8) !u16 {
    const data = try reader_v3.readFileBytesForTest(allocator, path);
    defer allocator.free(data);
    if (data.len < 6) return error.TruncatedData;
    if (!std.mem.eql(u8, data[0..4], &MAGIC)) return error.InvalidMagic;
    return readIntLe(u16, data[4..6]);
}

pub fn saveDatabase(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
) !void {
    return saveDatabaseV3(allocator, db, path, .{});
}

pub fn saveDatabaseWithConfig(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
    config: StorageConfig,
) !void {
    return saveDatabaseV3(allocator, db, path, config);
}

pub fn loadDatabase(
    allocator: std.mem.Allocator,
    path: []const u8,
) !database.Database {
    return loadDatabaseWithConfig(allocator, path, .{});
}

pub fn loadDatabaseWithConfig(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: StorageConfig,
) !database.Database {
    return switch (try detectFormatVersion(allocator, path)) {
        2 => loadDatabaseV2(allocator, path, config),
        3 => loadDatabaseV3(allocator, path, config),
        else => error.UnsupportedVersion,
    };
}

test "crc32 basic" {
    const data = "hello world";
    const crc = Crc32.compute(data);
    try std.testing.expect(crc != 0);
    try std.testing.expectEqual(crc, Crc32.compute(data));
    try std.testing.expect(crc != Crc32.compute("hello world!"));
}

test "bloom filter" {
    const allocator = std.testing.allocator;

    var bloom = try BloomFilter.init(allocator, 1000, 0.01);
    defer bloom.deinit();

    bloom.add(100);
    bloom.add(200);
    bloom.add(300);

    try std.testing.expect(bloom.mightContain(100));
    try std.testing.expect(bloom.mightContain(200));
    try std.testing.expect(bloom.mightContain(300));

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
    const decoded = try deltaDecode(encoded, allocator);
    defer allocator.free(decoded);
    try std.testing.expectEqualSlices(u64, &ids, decoded);
}

test "v3 header and section directory serialize" {
    const header = FileHeader{
        .section_count = 2,
        .section_dir_size = 2 * DIRECTORY_ENTRY_SIZE,
        .vector_count = 4,
        .dimension = 384,
        .distance_metric = .cosine,
    };
    const header_bytes = header.serialize();
    const restored = try FileHeader.deserialize(header_bytes);

    try std.testing.expectEqual(@as(u16, FORMAT_VERSION), restored.version);
    try std.testing.expectEqual(@as(u16, 2), restored.section_count);
    try std.testing.expectEqual(@as(u64, 4), restored.vector_count);
    try std.testing.expectEqual(@as(u32, 384), restored.dimension);

    const entry = SectionDirectoryEntry{
        .section_type = .vectors,
        .offset = 128,
        .length = 512,
        .checksum = 0x12345678,
        .item_count = 4,
    };
    const entry_bytes = entry.serialize();
    const restored_entry = try SectionDirectoryEntry.deserialize(entry_bytes);
    try std.testing.expectEqual(SectionType.vectors, restored_entry.section_type);
    try std.testing.expectEqual(@as(u64, 128), restored_entry.offset);
    try std.testing.expectEqual(@as(u64, 512), restored_entry.length);
}

test "v3 save/load roundtrip" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/roundtrip-v3.wdbx", .{tmp.sub_path});
    defer allocator.free(path);

    var db = try database.Database.init(allocator, "roundtrip-v3");
    defer db.deinit();
    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, "meta-1");
    try db.insert(2, &.{ 0.3, 0.2, 0.1 }, "meta-2");

    try saveDatabase(allocator, &db, path);
    const loaded = try loadDatabase(allocator, path);
    defer @constCast(&loaded).deinit();

    try std.testing.expectEqual(@as(usize, 2), loaded.records.items.len);
    try std.testing.expectEqualStrings("roundtrip-v3", loaded.name);
    try std.testing.expectEqual(@as(u64, 1), loaded.records.items[0].id);
    try std.testing.expectEqualStrings("meta-2", loaded.records.items[1].metadata.?);
}

test "v3 load detects section corruption" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/corrupt-v3.wdbx", .{tmp.sub_path});
    defer allocator.free(path);

    var db = try database.Database.init(allocator, "corrupt-v3");
    defer db.deinit();
    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, "meta");
    try saveDatabase(allocator, &db, path);

    var bytes = try readFileBytesForTest(allocator, path);
    defer allocator.free(bytes);

    var header_raw: [HEADER_SIZE]u8 = undefined;
    @memcpy(&header_raw, bytes[0..HEADER_SIZE]);
    const header = try FileHeader.deserialize(header_raw);

    var entry_raw: [DIRECTORY_ENTRY_SIZE]u8 = undefined;
    const vectors_entry_offset = HEADER_SIZE + DIRECTORY_ENTRY_SIZE;
    @memcpy(&entry_raw, bytes[vectors_entry_offset .. vectors_entry_offset + DIRECTORY_ENTRY_SIZE]);
    const entry = try SectionDirectoryEntry.deserialize(entry_raw);
    bytes[entry.offset + 1] ^= 0xFF;
    try writeFileBytesForTest(allocator, path, bytes);

    try std.testing.expectError(error.InvalidChecksum, loadDatabase(allocator, path));
    _ = header;
}

test "v3 load detects missing vectors section" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/missing-vectors-v3.wdbx", .{tmp.sub_path});
    defer allocator.free(path);

    var db = try database.Database.init(allocator, "missing-vectors-v3");
    defer db.deinit();
    try db.insert(1, &.{ 0.1, 0.2, 0.3 }, "meta");
    try saveDatabase(allocator, &db, path);

    var bytes = try readFileBytesForTest(allocator, path);
    defer allocator.free(bytes);

    var entry_raw: [DIRECTORY_ENTRY_SIZE]u8 = undefined;
    const vectors_entry_offset = HEADER_SIZE + DIRECTORY_ENTRY_SIZE;
    @memcpy(&entry_raw, bytes[vectors_entry_offset .. vectors_entry_offset + DIRECTORY_ENTRY_SIZE]);
    var entry = try SectionDirectoryEntry.deserialize(entry_raw);
    entry.section_type = .lineage;
    const entry_bytes = entry.serialize();
    @memcpy(bytes[vectors_entry_offset .. vectors_entry_offset + DIRECTORY_ENTRY_SIZE], &entry_bytes);
    try writeFileBytesForTest(allocator, path, bytes);

    try std.testing.expectError(
        error.CorruptedData,
        loadDatabaseWithConfig(allocator, path, .{ .verify_checksums = false }),
    );
}

test "v2 files still load through the unified API" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/legacy-v2.wdbx", .{tmp.sub_path});
    defer allocator.free(path);

    var db = try database.Database.init(allocator, "legacy-v2");
    defer db.deinit();
    try db.insert(5, &.{ 0.5, 0.6, 0.7 }, "legacy");

    try saveDatabaseV2(allocator, &db, path, .{});
    var loaded = try loadDatabase(allocator, path);
    defer loaded.deinit();

    try std.testing.expectEqual(@as(usize, 1), loaded.records.items.len);
    try std.testing.expectEqualStrings("legacy-v2", loaded.name);
    try std.testing.expectEqual(@as(u64, 5), loaded.records.items[0].id);
}

test {
    std.testing.refAllDecls(@This());
}
