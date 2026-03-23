//! WDBX Storage Format v3 — sectioned on-disk format types.

const std = @import("std");

pub const MAGIC: [4]u8 = .{ 'W', 'D', 'B', 'X' };
pub const FORMAT_VERSION: u16 = 3;
pub const MIN_READ_VERSION: u16 = 2;

pub const HEADER_SIZE: usize = 80;
pub const FOOTER_SIZE: usize = 40;
pub const DIRECTORY_ENTRY_SIZE: usize = 32;

pub const SectionType = enum(u16) {
    metadata = 0x0001,
    vectors = 0x0002,
    bloom_filter = 0x0003,
    vector_index = 0x0004,
    lineage = 0x0005,
    distributed = 0x0006,
};

pub const CompressionType = enum(u8) {
    none = 0x00,
    lz4 = 0x01,
    zstd = 0x02,
    delta = 0x03,
};

pub const DistanceMetric = enum(u8) {
    euclidean = 0,
    cosine = 1,
    dot_product = 2,
    manhattan = 3,
};

pub const HeaderFlags = packed struct(u16) {
    has_directory: bool = true,
    has_metadata: bool = true,
    has_vectors: bool = true,
    has_index: bool = false,
    has_lineage: bool = false,
    has_distributed: bool = false,
    compressed: bool = false,
    encrypted: bool = false,
    reserved: u8 = 0,
};

pub const FileHeader = struct {
    magic: [4]u8 = MAGIC,
    version: u16 = FORMAT_VERSION,
    flags: HeaderFlags = .{},
    header_size: u16 = HEADER_SIZE,
    section_count: u16 = 0,
    reserved0: u16 = 0,
    section_dir_offset: u64 = HEADER_SIZE,
    section_dir_size: u64 = 0,
    created_at: i64 = 0,
    modified_at: i64 = 0,
    vector_count: u64 = 0,
    dimension: u32 = 0,
    distance_metric: DistanceMetric = .euclidean,
    compression: CompressionType = .none,
    reserved1: [2]u8 = .{0} ** 2,
    uuid: [16]u8 = .{0} ** 16,
    reserved2: [16]u8 = .{0} ** 16,

    pub fn serialize(self: FileHeader) [HEADER_SIZE]u8 {
        var buf: [HEADER_SIZE]u8 = .{0} ** HEADER_SIZE;
        @memcpy(buf[0..4], &self.magic);
        std.mem.writeInt(u16, buf[4..6], self.version, .little);
        std.mem.writeInt(u16, buf[6..8], @bitCast(self.flags), .little);
        std.mem.writeInt(u16, buf[8..10], self.header_size, .little);
        std.mem.writeInt(u16, buf[10..12], self.section_count, .little);
        std.mem.writeInt(u16, buf[12..14], self.reserved0, .little);
        std.mem.writeInt(u64, buf[16..24], self.section_dir_offset, .little);
        std.mem.writeInt(u64, buf[24..32], self.section_dir_size, .little);
        std.mem.writeInt(i64, buf[32..40], self.created_at, .little);
        std.mem.writeInt(i64, buf[40..48], self.modified_at, .little);
        std.mem.writeInt(u64, buf[48..56], self.vector_count, .little);
        std.mem.writeInt(u32, buf[56..60], self.dimension, .little);
        buf[60] = @intFromEnum(self.distance_metric);
        buf[61] = @intFromEnum(self.compression);
        @memcpy(buf[62..64], &self.reserved1);
        @memcpy(buf[64..80], &self.uuid);
        return buf;
    }

    pub fn deserialize(buf: [HEADER_SIZE]u8) !FileHeader {
        if (!std.mem.eql(u8, buf[0..4], &MAGIC)) return error.InvalidMagic;
        const version = std.mem.readInt(u16, buf[4..6], .little);
        if (version != FORMAT_VERSION) return error.UnsupportedVersion;
        return .{
            .magic = buf[0..4].*,
            .version = version,
            .flags = @bitCast(std.mem.readInt(u16, buf[6..8], .little)),
            .header_size = std.mem.readInt(u16, buf[8..10], .little),
            .section_count = std.mem.readInt(u16, buf[10..12], .little),
            .reserved0 = std.mem.readInt(u16, buf[12..14], .little),
            .section_dir_offset = std.mem.readInt(u64, buf[16..24], .little),
            .section_dir_size = std.mem.readInt(u64, buf[24..32], .little),
            .created_at = std.mem.readInt(i64, buf[32..40], .little),
            .modified_at = std.mem.readInt(i64, buf[40..48], .little),
            .vector_count = std.mem.readInt(u64, buf[48..56], .little),
            .dimension = std.mem.readInt(u32, buf[56..60], .little),
            .distance_metric = @enumFromInt(buf[60]),
            .compression = @enumFromInt(buf[61]),
            .reserved1 = buf[62..64].*,
            .uuid = buf[64..80].*,
        };
    }
};

pub const SectionDirectoryEntry = struct {
    section_type: SectionType,
    flags: u16 = 0,
    offset: u64,
    length: u64,
    checksum: u32 = 0,
    item_count: u32 = 0,
    reserved: u32 = 0,

    pub fn serialize(self: SectionDirectoryEntry) [DIRECTORY_ENTRY_SIZE]u8 {
        var buf: [DIRECTORY_ENTRY_SIZE]u8 = .{0} ** DIRECTORY_ENTRY_SIZE;
        std.mem.writeInt(u16, buf[0..2], @intFromEnum(self.section_type), .little);
        std.mem.writeInt(u16, buf[2..4], self.flags, .little);
        std.mem.writeInt(u64, buf[4..12], self.offset, .little);
        std.mem.writeInt(u64, buf[12..20], self.length, .little);
        std.mem.writeInt(u32, buf[20..24], self.checksum, .little);
        std.mem.writeInt(u32, buf[24..28], self.item_count, .little);
        std.mem.writeInt(u32, buf[28..32], self.reserved, .little);
        return buf;
    }

    pub fn deserialize(buf: [DIRECTORY_ENTRY_SIZE]u8) !SectionDirectoryEntry {
        return .{
            .section_type = @enumFromInt(std.mem.readInt(u16, buf[0..2], .little)),
            .flags = std.mem.readInt(u16, buf[2..4], .little),
            .offset = std.mem.readInt(u64, buf[4..12], .little),
            .length = std.mem.readInt(u64, buf[12..20], .little),
            .checksum = std.mem.readInt(u32, buf[20..24], .little),
            .item_count = std.mem.readInt(u32, buf[24..28], .little),
            .reserved = std.mem.readInt(u32, buf[28..32], .little),
        };
    }
};

pub const FileFooter = struct {
    marker: [4]u8 = .{ 'E', 'N', 'D', '3' },
    header_checksum: u32 = 0,
    directory_checksum: u32 = 0,
    data_checksum: u32 = 0,
    file_checksum: u32 = 0,
    file_size: u64 = 0,
    section_count: u32 = 0,
    reserved: [8]u8 = .{0} ** 8,

    pub fn serialize(self: FileFooter) [FOOTER_SIZE]u8 {
        var buf: [FOOTER_SIZE]u8 = .{0} ** FOOTER_SIZE;
        @memcpy(buf[0..4], &self.marker);
        std.mem.writeInt(u32, buf[4..8], self.header_checksum, .little);
        std.mem.writeInt(u32, buf[8..12], self.directory_checksum, .little);
        std.mem.writeInt(u32, buf[12..16], self.data_checksum, .little);
        std.mem.writeInt(u32, buf[16..20], self.file_checksum, .little);
        std.mem.writeInt(u64, buf[20..28], self.file_size, .little);
        std.mem.writeInt(u32, buf[28..32], self.section_count, .little);
        @memcpy(buf[32..40], &self.reserved);
        return buf;
    }

    pub fn deserialize(buf: [FOOTER_SIZE]u8) !FileFooter {
        if (!std.mem.eql(u8, buf[0..4], &.{ 'E', 'N', 'D', '3' })) return error.InvalidFooter;
        return .{
            .marker = buf[0..4].*,
            .header_checksum = std.mem.readInt(u32, buf[4..8], .little),
            .directory_checksum = std.mem.readInt(u32, buf[8..12], .little),
            .data_checksum = std.mem.readInt(u32, buf[12..16], .little),
            .file_checksum = std.mem.readInt(u32, buf[16..20], .little),
            .file_size = std.mem.readInt(u64, buf[20..28], .little),
            .section_count = std.mem.readInt(u32, buf[28..32], .little),
            .reserved = buf[32..40].*,
        };
    }
};
