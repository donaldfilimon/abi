//! WDBX v2 write path — serialises a Database to the on-disk format.

const std = @import("std");
const database = @import("../database.zig");
const fmt = @import("format.zig");
const integrity = @import("integrity.zig");

const Crc32 = integrity.Crc32;
const BloomFilter = integrity.BloomFilter;
const BlockType = fmt.BlockType;
const FileHeader = fmt.FileHeader;
const FileFooter = fmt.FileFooter;
const FORMAT_VERSION = fmt.FORMAT_VERSION;
const HEADER_SIZE = fmt.HEADER_SIZE;

pub const StorageV2Config = @import("mod.zig").StorageV2Config;
pub const HnswGraphData = @import("mod.zig").HnswGraphData;

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

/// Accumulated state during a v2 save operation.
pub const SaveState = struct {
    data_crc: Crc32,
    file_crc: Crc32,
    total_size: u64,
    num_blocks: u32,

    pub fn init() SaveState {
        return .{ .data_crc = Crc32{}, .file_crc = Crc32{}, .total_size = 0, .num_blocks = 0 };
    }

    pub fn updateData(self: *SaveState, bytes: []const u8) void {
        self.data_crc.update(bytes);
        self.file_crc.update(bytes);
        self.total_size += bytes.len;
    }

    pub fn updateFileOnly(self: *SaveState, bytes: []const u8) void {
        self.file_crc.update(bytes);
        self.total_size += bytes.len;
    }
};

pub fn writeHeaderV2(file: anytype, io: anytype, db: *database.Database, config: StorageV2Config) !struct { header_checksum: u32, header: FileHeader } {
    const dimension: u32 = if (db.records.items.len > 0)
        @intCast(db.records.items[0].vector.len)
    else
        0;

    const header = FileHeader{
        .version = FORMAT_VERSION,
        .flags = .{
            .has_bloom_filter = config.enable_bloom_filter,
            .has_metadata = true,
            .compressed = config.enable_quantization,
            .has_index = config.include_index,
        },
        .created_at = 0,
        .modified_at = 0,
        .vector_count = db.records.items.len,
        .dimension = dimension,
        .compression = if (config.enable_quantization) .delta else .none,
    };

    const header_bytes = header.serialize();
    try file.writeStreamingAll(io, &header_bytes);
    return .{ .header_checksum = Crc32.compute(&header_bytes), .header = header };
}

pub fn writeMetadataBlock(
    allocator: std.mem.Allocator,
    file: anytype,
    io: anytype,
    db: *database.Database,
    state: *SaveState,
) !void {
    const name_len: u16 = @intCast(@min(db.name.len, 65535));
    const metadata_len: usize = 2 + name_len;
    const metadata = try allocator.alloc(u8, metadata_len);
    defer allocator.free(metadata);

    std.mem.writeInt(u16, metadata[0..2], name_len, .little);
    @memcpy(metadata[2..][0..name_len], db.name[0..name_len]);

    try file.writeStreamingAll(io, metadata);
    state.updateData(metadata);
    state.num_blocks += 1;
}

pub fn writeBloomBlock(
    allocator: std.mem.Allocator,
    file: anytype,
    io: anytype,
    db: *database.Database,
    config: StorageV2Config,
    state: *SaveState,
) !void {
    if (!config.enable_bloom_filter or db.records.items.len == 0) return;

    var bloom = try BloomFilter.init(allocator, db.records.items.len, config.bloom_fp_rate);
    defer bloom.deinit();

    for (db.records.items) |record| {
        bloom.add(record.id);
    }

    const bloom_data = try bloom.serialize(allocator);
    defer allocator.free(bloom_data);

    var bloom_header: [8]u8 = undefined;
    bloom_header[0] = @intFromEnum(BlockType.bloom_filter);
    bloom_header[1] = 0;
    std.mem.writeInt(u16, bloom_header[2..4], 0, .little);
    std.mem.writeInt(u32, bloom_header[4..8], @intCast(bloom_data.len), .little);

    try file.writeStreamingAll(io, &bloom_header);
    try file.writeStreamingAll(io, bloom_data);
    state.updateData(&bloom_header);
    state.updateData(bloom_data);
    state.num_blocks += 1;
}

/// Flush a write buffer through the file, updating CRCs and counters.
pub fn flushWriteBuffer(
    file: anytype,
    io: anytype,
    write_buffer: []u8,
    buffer_pos: *usize,
    state: *SaveState,
) !void {
    if (buffer_pos.* > 0) {
        const pending = write_buffer[0..buffer_pos.*];
        try file.writeStreamingAll(io, pending);
        state.updateData(pending);
        buffer_pos.* = 0;
    }
}

/// Buffer `src` bytes into `write_buffer`, flushing to disk as needed.
pub fn bufferBytes(
    file: anytype,
    io: anytype,
    write_buffer: []u8,
    buffer_pos: *usize,
    state: *SaveState,
    src: []const u8,
) !void {
    var offset: usize = 0;
    while (offset < src.len) {
        const space = write_buffer.len - buffer_pos.*;
        if (space == 0) {
            try flushWriteBuffer(file, io, write_buffer, buffer_pos, state);
            continue;
        }
        const to_copy = @min(src.len - offset, space);
        @memcpy(write_buffer[buffer_pos.*..][0..to_copy], src[offset..][0..to_copy]);
        buffer_pos.* += to_copy;
        offset += to_copy;
    }
}

pub fn writeVectorBlocks(
    allocator: std.mem.Allocator,
    file: anytype,
    io: anytype,
    db: *database.Database,
    header: FileHeader,
    config: StorageV2Config,
    state: *SaveState,
) !void {
    const write_buffer = try allocator.alloc(u8, config.write_buffer_size);
    defer allocator.free(write_buffer);
    var buffer_pos: usize = 0;

    // Compute total vector data size
    var vector_data_size: u64 = 0;
    for (db.records.items) |record| {
        const metadata_size: u64 = if (record.metadata) |meta| @intCast(meta.len) else 0;
        vector_data_size += 16;
        vector_data_size += @as(u64, @intCast(record.vector.len)) * @sizeOf(f32);
        vector_data_size += metadata_size;
    }

    // Write vector data block header
    var vector_header: [16]u8 = undefined;
    vector_header[0] = @intFromEnum(BlockType.vector_data);
    vector_header[1] = @intFromEnum(header.compression);
    std.mem.writeInt(u16, vector_header[2..4], 0, .little);
    std.mem.writeInt(u32, vector_header[4..8], @intCast(db.records.items.len), .little);
    std.mem.writeInt(u64, vector_header[8..16], vector_data_size, .little);

    try file.writeStreamingAll(io, &vector_header);
    state.updateData(&vector_header);
    state.num_blocks += 1;

    // Write each record using the shared write buffer
    for (db.records.items) |record| {
        const meta_length: u32 = if (record.metadata) |m| @intCast(m.len) else 0;

        var record_header: [16]u8 = undefined;
        std.mem.writeInt(u64, record_header[0..8], record.id, .little);
        std.mem.writeInt(u32, record_header[8..12], @intCast(record.vector.len), .little);
        std.mem.writeInt(u32, record_header[12..16], meta_length, .little);

        try bufferBytes(file, io, write_buffer, &buffer_pos, state, &record_header);
        try bufferBytes(file, io, write_buffer, &buffer_pos, state, std.mem.sliceAsBytes(record.vector));
        if (record.metadata) |meta| {
            try bufferBytes(file, io, write_buffer, &buffer_pos, state, meta);
        }
    }

    try flushWriteBuffer(file, io, write_buffer, &buffer_pos, state);
}

/// Write a vector_index (0x03) block containing a serialised HNSW graph.
///
/// On-disk layout:
///   [block type u8][reserved 3 bytes][entry_point u32 LE][max_layer u32 LE]
///   [node_count u32 LE]
///   for each node:
///     [neighbor_count u32 LE][neighbor_id u32 LE ...]
pub fn writeHnswBlock(
    file: anytype,
    io: anytype,
    hnsw: HnswGraphData,
    state: *SaveState,
) !void {
    // Block header: type + 3 reserved + entry_point + max_layer + node_count = 16 bytes
    var blk_header: [16]u8 = .{0} ** 16;
    blk_header[0] = @intFromEnum(BlockType.vector_index);
    std.mem.writeInt(u32, blk_header[4..8], hnsw.entry_point, .little);
    std.mem.writeInt(u32, blk_header[8..12], hnsw.max_layer, .little);
    std.mem.writeInt(u32, blk_header[12..16], @intCast(hnsw.neighbors.len), .little);

    try file.writeStreamingAll(io, &blk_header);
    state.updateData(&blk_header);

    // Write per-node neighbor lists.
    for (hnsw.neighbors) |nbrs| {
        var count_bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &count_bytes, @intCast(nbrs.len), .little);
        try file.writeStreamingAll(io, &count_bytes);
        state.updateData(&count_bytes);

        for (nbrs) |nid| {
            var nid_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &nid_bytes, nid, .little);
            try file.writeStreamingAll(io, &nid_bytes);
            state.updateData(&nid_bytes);
        }
    }

    state.num_blocks += 1;
}

pub fn writeFooterV2(file: anytype, io: anytype, header_checksum: u32, state: *const SaveState) !void {
    const footer = FileFooter{
        .header_checksum = header_checksum,
        .data_checksum = @constCast(&state.data_crc).finalize(),
        .file_checksum = @constCast(&state.file_crc).finalize(),
        .file_size = state.total_size,
        .num_blocks = state.num_blocks,
    };
    const footer_bytes = footer.serialize();
    try file.writeStreamingAll(io, &footer_bytes);
}

pub fn saveDatabaseV2(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
    config: StorageV2Config,
) !void {
    return saveDatabaseV2Impl(allocator, db, path, config, null);
}

/// Save database including an HNSW graph index block (vector_index 0x03).
pub fn saveDatabaseWithIndex(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
    config: StorageV2Config,
    hnsw: HnswGraphData,
) !void {
    return saveDatabaseV2Impl(allocator, db, path, config, hnsw);
}

fn saveDatabaseV2Impl(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
    config: StorageV2Config,
    hnsw: ?HnswGraphData,
) !void {
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);

    var state = SaveState.init();

    // 1. Write header
    const hdr = try writeHeaderV2(&file, io, db, config);
    const header_bytes = hdr.header.serialize();
    state.updateFileOnly(&header_bytes);

    // 2. Write metadata block
    try writeMetadataBlock(allocator, &file, io, db, &state);

    // 3. Write bloom filter (optional)
    try writeBloomBlock(allocator, &file, io, db, config, &state);

    // 4. Write vector data
    try writeVectorBlocks(allocator, &file, io, db, hdr.header, config, &state);

    // 5. Write HNSW index block (optional)
    if (hnsw) |h| try writeHnswBlock(&file, io, h, &state);

    // 6. Write footer
    try writeFooterV2(&file, io, hdr.header_checksum, &state);
}
