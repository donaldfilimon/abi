//! WDBX v3 write path — sectioned persistence format.

const std = @import("std");
const database = @import("../database.zig");
const fmt = @import("format.zig");
const integrity = @import("integrity.zig");

const Crc32 = integrity.Crc32;
const BloomFilter = integrity.BloomFilter;

pub const StorageConfig = @import("mod.zig").StorageConfig;
pub const HnswGraphData = @import("mod.zig").HnswGraphData;

const SectionBytes = struct {
    section_type: fmt.SectionType,
    bytes: []u8,
    item_count: u32 = 0,
    flags: u16 = 0,
};

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

fn ensureParentDirExists(allocator: std.mem.Allocator, path: []const u8) !void {
    const dir_path = std.fs.path.dirname(path) orelse return;
    if (dir_path.len == 0) return;

    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();
    std.Io.Dir.cwd().createDirPath(io, dir_path) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

fn appendInt(comptime T: type, out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: T) !void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    try out.appendSlice(allocator, &bytes);
}

fn buildMetadataSection(allocator: std.mem.Allocator, db: *database.Database) !SectionBytes {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    const name_len: u16 = @intCast(@min(db.name.len, std.math.maxInt(u16)));
    try appendInt(u16, &out, allocator, name_len);
    try out.appendSlice(allocator, db.name[0..name_len]);

    return .{
        .section_type = .metadata,
        .bytes = try out.toOwnedSlice(allocator),
        .item_count = 1,
    };
}

fn buildVectorSection(allocator: std.mem.Allocator, db: *database.Database) !SectionBytes {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    for (db.records.items) |record| {
        const meta_len: u32 = if (record.metadata) |meta| @intCast(meta.len) else 0;
        try appendInt(u64, &out, allocator, record.id);
        try appendInt(u32, &out, allocator, @intCast(record.vector.len));
        try appendInt(u32, &out, allocator, meta_len);
        try out.appendSlice(allocator, std.mem.sliceAsBytes(record.vector));
        if (record.metadata) |meta| {
            try out.appendSlice(allocator, meta);
        }
    }

    return .{
        .section_type = .vectors,
        .bytes = try out.toOwnedSlice(allocator),
        .item_count = @intCast(db.records.items.len),
    };
}

fn buildBloomSection(
    allocator: std.mem.Allocator,
    db: *database.Database,
    config: StorageConfig,
) !?SectionBytes {
    if (!config.enable_bloom_filter or db.records.items.len == 0) return null;

    var bloom = try BloomFilter.init(allocator, db.records.items.len, config.bloom_fp_rate);
    defer bloom.deinit();

    for (db.records.items) |record| bloom.add(record.id);
    const bytes = try bloom.serialize(allocator);

    return .{
        .section_type = .bloom_filter,
        .bytes = bytes,
        .item_count = @intCast(db.records.items.len),
    };
}

fn buildIndexSection(allocator: std.mem.Allocator, hnsw: HnswGraphData) !SectionBytes {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try appendInt(u32, &out, allocator, hnsw.entry_point);
    try appendInt(u32, &out, allocator, hnsw.max_layer);
    try appendInt(u32, &out, allocator, @intCast(hnsw.neighbors.len));
    for (hnsw.neighbors) |nbrs| {
        try appendInt(u32, &out, allocator, @intCast(nbrs.len));
        for (nbrs) |nid| try appendInt(u32, &out, allocator, nid);
    }

    return .{
        .section_type = .vector_index,
        .bytes = try out.toOwnedSlice(allocator),
        .item_count = @intCast(hnsw.neighbors.len),
    };
}

fn collectSections(
    allocator: std.mem.Allocator,
    db: *database.Database,
    config: StorageConfig,
    hnsw: ?HnswGraphData,
) !std.ArrayListUnmanaged(SectionBytes) {
    var sections = std.ArrayListUnmanaged(SectionBytes).empty;
    errdefer {
        for (sections.items) |section| allocator.free(section.bytes);
        sections.deinit(allocator);
    }

    try sections.append(allocator, try buildMetadataSection(allocator, db));
    try sections.append(allocator, try buildVectorSection(allocator, db));
    if (try buildBloomSection(allocator, db, config)) |section| {
        try sections.append(allocator, section);
    }
    if (config.include_index and hnsw != null) {
        try sections.append(allocator, try buildIndexSection(allocator, hnsw.?));
    }

    return sections;
}

fn freeSections(allocator: std.mem.Allocator, sections: *std.ArrayListUnmanaged(SectionBytes)) void {
    for (sections.items) |section| allocator.free(section.bytes);
    sections.deinit(allocator);
}

fn computeDimension(db: *database.Database) u32 {
    return if (db.records.items.len > 0)
        @intCast(db.records.items[0].vector.len)
    else
        0;
}

pub fn saveDatabaseV3(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
    config: StorageConfig,
) !void {
    return saveDatabaseV3WithIndex(allocator, db, path, config, null);
}

pub fn saveDatabaseV3WithIndex(
    allocator: std.mem.Allocator,
    db: *database.Database,
    path: []const u8,
    config: StorageConfig,
    hnsw: ?HnswGraphData,
) !void {
    var sections = try collectSections(allocator, db, config, hnsw);
    defer freeSections(allocator, &sections);

    const directory_size = sections.items.len * fmt.DIRECTORY_ENTRY_SIZE;
    const section_count: u16 = @intCast(sections.items.len);

    var header = fmt.FileHeader{
        .flags = .{
            .has_directory = true,
            .has_metadata = true,
            .has_vectors = true,
            .has_index = config.include_index and hnsw != null,
            .compressed = config.enable_quantization,
        },
        .section_count = section_count,
        .section_dir_offset = fmt.HEADER_SIZE,
        .section_dir_size = directory_size,
        .vector_count = db.records.items.len,
        .dimension = computeDimension(db),
        .compression = if (config.enable_quantization) .delta else .none,
    };

    const directory = try allocator.alloc(u8, directory_size);
    defer allocator.free(directory);

    var offset: u64 = fmt.HEADER_SIZE + directory_size;
    var data_crc = Crc32{};
    for (sections.items, 0..) |section, idx| {
        const checksum = Crc32.compute(section.bytes);
        const entry = fmt.SectionDirectoryEntry{
            .section_type = section.section_type,
            .flags = section.flags,
            .offset = offset,
            .length = section.bytes.len,
            .checksum = checksum,
            .item_count = section.item_count,
        };
        const entry_bytes = entry.serialize();
        @memcpy(
            directory[idx * fmt.DIRECTORY_ENTRY_SIZE .. (idx + 1) * fmt.DIRECTORY_ENTRY_SIZE],
            &entry_bytes,
        );
        offset += section.bytes.len;
        data_crc.update(section.bytes);
    }

    const header_bytes = header.serialize();
    const header_checksum = Crc32.compute(&header_bytes);
    const directory_checksum = Crc32.compute(directory);

    var file_crc = Crc32{};
    file_crc.update(&header_bytes);
    file_crc.update(directory);
    for (sections.items) |section| file_crc.update(section.bytes);

    const footer = fmt.FileFooter{
        .header_checksum = header_checksum,
        .directory_checksum = directory_checksum,
        .data_checksum = data_crc.finalize(),
        .file_checksum = file_crc.finalize(),
        .file_size = offset,
        .section_count = section_count,
    };
    const footer_bytes = footer.serialize();

    try ensureParentDirExists(allocator, path);

    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);

    try file.writeStreamingAll(io, &header_bytes);
    try file.writeStreamingAll(io, directory);
    for (sections.items) |section| {
        try file.writeStreamingAll(io, section.bytes);
    }
    try file.writeStreamingAll(io, &footer_bytes);
}
