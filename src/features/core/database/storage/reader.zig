//! WDBX v2 read path — deserialises a Database from the on-disk format.

const std = @import("std");
const database = @import("../database.zig");
const fmt = @import("format_v2.zig");
const integrity = @import("integrity.zig");

const Crc32 = integrity.Crc32;
const BlockType = fmt.BlockType;
const FileHeader = fmt.FileHeader;
const FileFooter = fmt.FileFooter;
const HEADER_SIZE = fmt.HEADER_SIZE;
const FOOTER_SIZE = fmt.FOOTER_SIZE;

pub const StorageV2Config = @import("mod.zig").StorageV2Config;
pub const HnswGraphData = @import("mod.zig").HnswGraphData;

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

pub fn verifyV2Checksums(data: []const u8, header_bytes: []const u8, footer: FileFooter, payload_end: usize) !void {
    const header_checksum = Crc32.compute(header_bytes);
    if (header_checksum != footer.header_checksum) return error.InvalidChecksum;

    const data_checksum = Crc32.compute(data[HEADER_SIZE..payload_end]);
    if (data_checksum != footer.data_checksum) return error.InvalidChecksum;

    const file_checksum = Crc32.compute(data[0..payload_end]);
    const legacy_checksum = Crc32.compute(header_bytes);
    if (footer.file_checksum != file_checksum and footer.file_checksum != legacy_checksum) {
        return error.InvalidChecksum;
    }
}

pub fn parseMetadataBlock(data: []const u8, cursor: *usize, payload_end: usize) ![]const u8 {
    if (cursor.* + 2 > payload_end) return error.TruncatedData;
    const name_len = std.mem.readInt(u16, data[cursor.*..][0..2], .little);
    cursor.* += 2;

    if (cursor.* + name_len > payload_end) return error.TruncatedData;
    const db_name = data[cursor.*..][0..name_len];
    cursor.* += name_len;
    return db_name;
}

pub fn skipBloomFilter(data: []const u8, cursor: *usize, payload_end: usize, header: FileHeader) !void {
    const expect_bloom = header.flags.has_bloom_filter and header.vector_count > 0;
    if (!header.flags.has_bloom_filter) return;

    if (cursor.* + 8 > payload_end) return error.TruncatedData;
    const block_type = data[cursor.*];
    if (block_type == @intFromEnum(BlockType.bloom_filter)) {
        cursor.* += 4;
        const bloom_size = std.mem.readInt(u32, data[cursor.*..][0..4], .little);
        cursor.* += 4;
        if (expect_bloom and bloom_size == 0) return error.InvalidBloomFilter;
        const bloom_size_usize: usize = @intCast(bloom_size);
        if (bloom_size_usize > payload_end - cursor.*) return error.TruncatedData;
        cursor.* += bloom_size_usize;
    } else if (expect_bloom) {
        return error.InvalidBloomFilter;
    }
}

pub fn readVectorRecord(
    allocator: std.mem.Allocator,
    data: []const u8,
    cursor: *usize,
    payload_end: usize,
    expected_dimension: u32,
) !struct { id: u64, vector: []f32, metadata: ?[]u8 } {
    if (cursor.* + 16 > payload_end) return error.TruncatedData;

    const id = std.mem.readInt(u64, data[cursor.*..][0..8], .little);
    cursor.* += 8;
    const vec_len = std.mem.readInt(u32, data[cursor.*..][0..4], .little);
    cursor.* += 4;
    const meta_len = std.mem.readInt(u32, data[cursor.*..][0..4], .little);
    cursor.* += 4;

    if (expected_dimension != 0 and vec_len != expected_dimension) {
        return error.CorruptedData;
    }

    const vec_bytes_len = vec_len * @sizeOf(f32);
    if (cursor.* + vec_bytes_len > payload_end) return error.TruncatedData;

    const vector = try allocator.alloc(f32, vec_len);
    errdefer allocator.free(vector);
    @memcpy(std.mem.sliceAsBytes(vector), data[cursor.*..][0..vec_bytes_len]);
    cursor.* += vec_bytes_len;

    var metadata: ?[]u8 = null;
    if (meta_len > 0) {
        if (cursor.* + meta_len > payload_end) return error.TruncatedData;
        metadata = try allocator.alloc(u8, meta_len);
        @memcpy(metadata.?, data[cursor.*..][0..meta_len]);
        cursor.* += meta_len;
    }

    return .{ .id = id, .vector = vector, .metadata = metadata };
}

/// Read a vector_index block from raw file data.  Returns the graph or
/// null if the cursor is not positioned at a vector_index block.
pub fn readHnswBlock(
    allocator: std.mem.Allocator,
    data: []const u8,
    cursor: *usize,
    payload_end: usize,
) !?HnswGraphData {
    if (cursor.* + 16 > payload_end) return null;
    if (data[cursor.*] != @intFromEnum(BlockType.vector_index)) return null;

    cursor.* += 4; // skip type + reserved
    const entry_point = std.mem.readInt(u32, data[cursor.*..][0..4], .little);
    cursor.* += 4;
    const max_layer = std.mem.readInt(u32, data[cursor.*..][0..4], .little);
    cursor.* += 4;
    const node_count = std.mem.readInt(u32, data[cursor.*..][0..4], .little);
    cursor.* += 4;

    const neighbors = try allocator.alloc([]u32, node_count);
    // Track how many entries have been initialised so the errdefer
    // only frees valid slices (avoids reading uninitialised memory).
    var nodes_read: u32 = 0;
    errdefer {
        for (neighbors[0..nodes_read]) |nbrs| allocator.free(nbrs);
        allocator.free(neighbors);
    }

    for (0..node_count) |i| {
        if (cursor.* + 4 > payload_end) return error.TruncatedData;
        const nbr_count = std.mem.readInt(u32, data[cursor.*..][0..4], .little);
        cursor.* += 4;

        const nbrs = try allocator.alloc(u32, nbr_count);
        errdefer allocator.free(nbrs);

        for (0..nbr_count) |j| {
            if (cursor.* + 4 > payload_end) return error.TruncatedData;
            nbrs[j] = std.mem.readInt(u32, data[cursor.*..][0..4], .little);
            cursor.* += 4;
        }
        neighbors[i] = nbrs;
        nodes_read += 1;
    }

    // Cast [][]u32 to []const []const u32 for the return struct.
    return .{
        .entry_point = entry_point,
        .max_layer = max_layer,
        .neighbors = neighbors,
    };
}

/// Load database with v2 format
pub fn loadDatabaseV2(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: StorageV2Config,
) !database.Database {
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const data = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(1024 * 1024 * 1024),
    );
    defer allocator.free(data);

    if (data.len < HEADER_SIZE + FOOTER_SIZE) return error.TruncatedData;
    const payload_end = data.len - FOOTER_SIZE;

    // 1. Parse and verify header/footer
    var header_bytes: [64]u8 = undefined;
    @memcpy(&header_bytes, data[0..HEADER_SIZE]);
    const header = try FileHeader.deserialize(header_bytes);

    var footer_bytes: [32]u8 = undefined;
    @memcpy(&footer_bytes, data[payload_end..]);
    const footer = try FileFooter.deserialize(footer_bytes);

    if (footer.file_size != payload_end) return error.CorruptedData;

    if (config.verify_checksums) {
        try verifyV2Checksums(data, &header_bytes, footer, payload_end);
    }

    // 2. Parse metadata and create database
    var cursor: usize = HEADER_SIZE;
    const db_name = try parseMetadataBlock(data, &cursor, payload_end);

    var db = try database.Database.init(allocator, db_name);
    errdefer db.deinit();

    // 3. Skip bloom filter if present
    try skipBloomFilter(data, &cursor, payload_end, header);

    // 4. Parse vector data block header
    if (cursor + 16 > payload_end) return error.TruncatedData;
    if (data[cursor] != @intFromEnum(BlockType.vector_data)) return error.CorruptedData;
    cursor += 4;
    const vector_count = std.mem.readInt(u32, data[cursor..][0..4], .little);
    cursor += 4;
    const vector_data_size = std.mem.readInt(u64, data[cursor..][0..8], .little);
    cursor += 8;

    if (@as(u64, vector_count) != header.vector_count) return error.CorruptedData;
    if (vector_count > 0 and header.dimension == 0) return error.CorruptedData;
    if (vector_data_size < @as(u64, vector_count) * 16) return error.CorruptedData;
    if (vector_data_size > @as(u64, @intCast(payload_end - cursor))) return error.TruncatedData;

    // 5. Read vector records
    const vector_payload_start = cursor;
    var i: usize = 0;
    while (i < vector_count) : (i += 1) {
        const rec = try readVectorRecord(allocator, data, &cursor, payload_end, header.dimension);
        errdefer {
            allocator.free(rec.vector);
            if (rec.metadata) |m| allocator.free(m);
        }
        try db.insertOwned(rec.id, rec.vector, rec.metadata);
    }

    // 6. Verify consumed payload matches declared size
    const consumed = cursor - vector_payload_start;
    if (@as(u64, @intCast(consumed)) != vector_data_size) return error.CorruptedData;

    // 7. Skip optional trailing blocks (e.g. vector_index / HNSW graph).
    //    The HNSW data is parsed but not stored on the Database struct itself;
    //    callers that need it should use loadDatabaseWithIndex.
    if (cursor < payload_end) {
        if (data[cursor] == @intFromEnum(BlockType.vector_index)) {
            const hnsw = try readHnswBlock(allocator, data, &cursor, payload_end);
            if (hnsw) |h| @import("mod.zig").freeHnswGraphData(allocator, h);
        }
    }

    if (cursor != payload_end) return error.CorruptedData;

    return db;
}

pub fn readFileBytesForTest(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024 * 1024));
}

pub fn writeFileBytesForTest(allocator: std.mem.Allocator, path: []const u8, bytes: []const u8) !void {
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, bytes);
}

test {
    std.testing.refAllDecls(@This());
}
