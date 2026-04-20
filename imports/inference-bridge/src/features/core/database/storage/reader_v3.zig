//! WDBX v3 read path — sectioned persistence format.

const std = @import("std");
const database = @import("../database.zig");
const fmt = @import("format.zig");
const integrity = @import("integrity.zig");

const Crc32 = integrity.Crc32;

pub const StorageConfig = @import("mod.zig").StorageConfig;

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

fn readIntLe(comptime T: type, bytes: []const u8) T {
    const raw: *const [@sizeOf(T)]u8 = @ptrCast(bytes.ptr);
    return std.mem.readInt(T, raw, .little);
}

fn readSectionEntries(data: []const u8, header: fmt.FileHeader) ![]fmt.SectionDirectoryEntry {
    if (header.section_dir_offset + header.section_dir_size > data.len - fmt.FOOTER_SIZE) {
        return error.CorruptedData;
    }
    if (header.section_dir_size != @as(u64, header.section_count) * fmt.DIRECTORY_ENTRY_SIZE) {
        return error.CorruptedData;
    }

    const bytes = data[header.section_dir_offset .. header.section_dir_offset + header.section_dir_size];
    const entries = try std.heap.page_allocator.alloc(fmt.SectionDirectoryEntry, header.section_count);
    errdefer std.heap.page_allocator.free(entries);

    for (entries, 0..) |*entry, idx| {
        var raw: [fmt.DIRECTORY_ENTRY_SIZE]u8 = undefined;
        const start = idx * fmt.DIRECTORY_ENTRY_SIZE;
        @memcpy(&raw, bytes[start .. start + fmt.DIRECTORY_ENTRY_SIZE]);
        entry.* = try fmt.SectionDirectoryEntry.deserialize(raw);
    }
    return entries;
}

fn sectionSlice(data: []const u8, payload_end: usize, entry: fmt.SectionDirectoryEntry) ![]const u8 {
    if (entry.offset + entry.length > payload_end) return error.CorruptedData;
    return data[entry.offset .. entry.offset + entry.length];
}

fn verifyChecksums(
    data: []const u8,
    header_bytes: []const u8,
    header: fmt.FileHeader,
    footer: fmt.FileFooter,
    payload_end: usize,
) !void {
    if (Crc32.compute(header_bytes) != footer.header_checksum) return error.InvalidChecksum;

    const directory = data[header.section_dir_offset .. header.section_dir_offset + header.section_dir_size];
    if (Crc32.compute(directory) != footer.directory_checksum) return error.InvalidChecksum;

    var data_crc = Crc32{};
    data_crc.update(data[header.section_dir_offset + header.section_dir_size .. payload_end]);
    if (data_crc.finalize() != footer.data_checksum) return error.InvalidChecksum;

    if (Crc32.compute(data[0..payload_end]) != footer.file_checksum) return error.InvalidChecksum;
}

fn parseDbName(bytes: []const u8) ![]const u8 {
    if (bytes.len < 2) return error.CorruptedData;
    const name_len = readIntLe(u16, bytes[0..2]);
    if (2 + name_len > bytes.len) return error.CorruptedData;
    return bytes[2 .. 2 + name_len];
}

fn parseVectors(
    allocator: std.mem.Allocator,
    db: *database.Database,
    bytes: []const u8,
    expected_dimension: u32,
    expected_count: u32,
) !void {
    var cursor: usize = 0;
    var count: u32 = 0;
    while (cursor < bytes.len) : (count += 1) {
        if (cursor + 16 > bytes.len) return error.CorruptedData;

        const id = readIntLe(u64, bytes[cursor..][0..8]);
        cursor += 8;
        const vec_len = readIntLe(u32, bytes[cursor..][0..4]);
        cursor += 4;
        const meta_len = readIntLe(u32, bytes[cursor..][0..4]);
        cursor += 4;

        if (expected_dimension != 0 and vec_len != expected_dimension) return error.CorruptedData;

        const vec_bytes_len = @as(usize, vec_len) * @sizeOf(f32);
        if (cursor + vec_bytes_len > bytes.len) return error.CorruptedData;
        const vector = try allocator.alloc(f32, vec_len);
        errdefer allocator.free(vector);
        @memcpy(std.mem.sliceAsBytes(vector), bytes[cursor .. cursor + vec_bytes_len]);
        cursor += vec_bytes_len;

        var metadata: ?[]u8 = null;
        if (meta_len > 0) {
            if (cursor + meta_len > bytes.len) return error.CorruptedData;
            metadata = try allocator.alloc(u8, meta_len);
            @memcpy(metadata.?, bytes[cursor .. cursor + meta_len]);
            cursor += meta_len;
        }

        try db.insertOwned(id, vector, metadata);
    }

    if (cursor != bytes.len or count != expected_count) return error.CorruptedData;
}

pub fn loadDatabaseV3(
    allocator: std.mem.Allocator,
    path: []const u8,
    config: StorageConfig,
) !database.Database {
    const data = try readFileBytesForTest(allocator, path);
    defer allocator.free(data);

    if (data.len < fmt.HEADER_SIZE + fmt.FOOTER_SIZE) return error.TruncatedData;
    const payload_end = data.len - fmt.FOOTER_SIZE;

    var header_raw: [fmt.HEADER_SIZE]u8 = undefined;
    @memcpy(&header_raw, data[0..fmt.HEADER_SIZE]);
    const header = try fmt.FileHeader.deserialize(header_raw);

    var footer_raw: [fmt.FOOTER_SIZE]u8 = undefined;
    @memcpy(&footer_raw, data[payload_end..]);
    const footer = try fmt.FileFooter.deserialize(footer_raw);

    if (footer.file_size != payload_end) return error.CorruptedData;
    if (footer.section_count != header.section_count) return error.CorruptedData;

    if (config.verify_checksums) {
        try verifyChecksums(data, &header_raw, header, footer, payload_end);
    }

    const entries = try readSectionEntries(data, header);
    defer std.heap.page_allocator.free(entries);

    var metadata_entry: ?fmt.SectionDirectoryEntry = null;
    var vector_entry: ?fmt.SectionDirectoryEntry = null;

    for (entries) |entry| {
        const bytes = try sectionSlice(data, payload_end, entry);
        if (config.verify_checksums and Crc32.compute(bytes) != entry.checksum) {
            return error.InvalidChecksum;
        }
        switch (entry.section_type) {
            .metadata => metadata_entry = entry,
            .vectors => vector_entry = entry,
            else => {},
        }
    }

    const name_entry = metadata_entry orelse return error.CorruptedData;
    const db_name = try parseDbName(try sectionSlice(data, payload_end, name_entry));

    var db = try database.Database.init(allocator, db_name);
    errdefer db.deinit();

    const vectors = try sectionSlice(data, payload_end, vector_entry orelse return error.CorruptedData);
    try parseVectors(allocator, &db, vectors, header.dimension, @intCast(header.vector_count));

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
