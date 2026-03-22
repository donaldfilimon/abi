//! WDBX Storage Format v2 — Type definitions for the on-disk format.
//!
//! Contains constants, header, footer, and block-type enums shared by
//! the reader and writer sub-modules.

const std = @import("std");

// ============================================================================
// Constants and Magic Numbers
// ============================================================================

/// File magic: "WDBX" in ASCII
pub const MAGIC: [4]u8 = .{ 'W', 'D', 'B', 'X' };

/// Current format version
pub const FORMAT_VERSION: u16 = 2;

/// Minimum supported format version for reading
pub const MIN_READ_VERSION: u16 = 1;

pub const HEADER_SIZE: usize = 64;
pub const FOOTER_SIZE: usize = 32;
pub const HEADER_UUID_OFFSET: usize = 40;
pub const HEADER_UUID_SIZE: usize = 16;
pub const HEADER_RESERVED_OFFSET: usize = 56;
pub const HEADER_RESERVED_STORED_SIZE: usize = 8;

/// Block types
pub const BlockType = enum(u8) {
    header = 0x00,
    metadata = 0x01,
    bloom_filter = 0x02,
    vector_index = 0x03,
    vector_data = 0x04,
    wal_entry = 0x05,
    footer = 0xFF,
};

/// Compression types
pub const CompressionType = enum(u8) {
    none = 0x00,
    lz4 = 0x01,
    zstd = 0x02,
    delta = 0x03, // Delta encoding for vectors
};

// ============================================================================
// File Header
// ============================================================================

/// 64-byte file header
pub const FileHeader = struct {
    /// Magic bytes "WDBX"
    magic: [4]u8 = MAGIC,
    /// Format version
    version: u16 = FORMAT_VERSION,
    /// Flags (compression, encryption, etc.)
    flags: HeaderFlags = .{},
    /// Creation timestamp (Unix epoch seconds)
    created_at: i64 = 0,
    /// Last modified timestamp
    modified_at: i64 = 0,
    /// Number of vectors
    vector_count: u64 = 0,
    /// Vector dimension
    dimension: u32 = 0,
    /// Distance metric
    distance_metric: DistanceMetric = .euclidean,
    /// Compression type for vector data
    compression: CompressionType = .none,
    /// UUID for database instance
    uuid: [16]u8 = .{0} ** 16,
    /// Reserved for future use
    reserved: [16]u8 = .{0} ** 16,

    pub fn serialize(self: FileHeader) [64]u8 {
        var buf: [64]u8 = .{0} ** 64;
        @memcpy(buf[0..4], &self.magic);
        std.mem.writeInt(u16, buf[4..6], self.version, .little);
        buf[6] = @bitCast(self.flags);
        std.mem.writeInt(i64, buf[8..16], self.created_at, .little);
        std.mem.writeInt(i64, buf[16..24], self.modified_at, .little);
        std.mem.writeInt(u64, buf[24..32], self.vector_count, .little);
        std.mem.writeInt(u32, buf[32..36], self.dimension, .little);
        buf[36] = @intFromEnum(self.distance_metric);
        buf[37] = @intFromEnum(self.compression);
        @memcpy(
            buf[HEADER_UUID_OFFSET .. HEADER_UUID_OFFSET + HEADER_UUID_SIZE],
            &self.uuid,
        );
        @memcpy(
            buf[HEADER_RESERVED_OFFSET .. HEADER_RESERVED_OFFSET + HEADER_RESERVED_STORED_SIZE],
            self.reserved[0..HEADER_RESERVED_STORED_SIZE],
        );
        return buf;
    }

    pub fn deserialize(buf: [64]u8) !FileHeader {
        if (!std.mem.eql(u8, buf[0..4], &MAGIC)) {
            return error.InvalidMagic;
        }

        const version = std.mem.readInt(u16, buf[4..6], .little);
        if (version > FORMAT_VERSION) {
            return error.UnsupportedVersion;
        }

        var reserved: [16]u8 = .{0} ** 16;
        @memcpy(
            reserved[0..HEADER_RESERVED_STORED_SIZE],
            buf[HEADER_RESERVED_OFFSET .. HEADER_RESERVED_OFFSET + HEADER_RESERVED_STORED_SIZE],
        );

        return .{
            .magic = buf[0..4].*,
            .version = version,
            .flags = @bitCast(buf[6]),
            .created_at = std.mem.readInt(i64, buf[8..16], .little),
            .modified_at = std.mem.readInt(i64, buf[16..24], .little),
            .vector_count = std.mem.readInt(u64, buf[24..32], .little),
            .dimension = std.mem.readInt(u32, buf[32..36], .little),
            .distance_metric = @enumFromInt(buf[36]),
            .compression = @enumFromInt(buf[37]),
            .uuid = buf[HEADER_UUID_OFFSET .. HEADER_UUID_OFFSET + HEADER_UUID_SIZE].*,
            .reserved = reserved,
        };
    }
};

pub const HeaderFlags = packed struct {
    has_bloom_filter: bool = false,
    has_index: bool = false,
    has_metadata: bool = false,
    encrypted: bool = false,
    compressed: bool = false,
    has_wal: bool = false,
    reserved1: bool = false,
    reserved2: bool = false,
};

pub const DistanceMetric = enum(u8) {
    euclidean = 0,
    cosine = 1,
    dot_product = 2,
    manhattan = 3,
};

// ============================================================================
// File Footer
// ============================================================================

/// 32-byte file footer with checksums
pub const FileFooter = struct {
    /// Marker for footer start
    marker: [4]u8 = .{ 'E', 'N', 'D', 0 },
    /// CRC32 of header
    header_checksum: u32 = 0,
    /// CRC32 of all data blocks
    data_checksum: u32 = 0,
    /// CRC32 of entire file (excluding footer)
    file_checksum: u32 = 0,
    /// Total file size (excluding footer)
    file_size: u64 = 0,
    /// Number of blocks
    num_blocks: u32 = 0,
    /// Reserved
    reserved: [8]u8 = .{0} ** 8,

    pub fn serialize(self: FileFooter) [32]u8 {
        var buf: [32]u8 = undefined;
        @memcpy(buf[0..4], &self.marker);
        std.mem.writeInt(u32, buf[4..8], self.header_checksum, .little);
        std.mem.writeInt(u32, buf[8..12], self.data_checksum, .little);
        std.mem.writeInt(u32, buf[12..16], self.file_checksum, .little);
        std.mem.writeInt(u64, buf[16..24], self.file_size, .little);
        std.mem.writeInt(u32, buf[24..28], self.num_blocks, .little);
        @memcpy(buf[28..32], self.reserved[0..4]);
        return buf;
    }

    pub fn deserialize(buf: [32]u8) !FileFooter {
        if (!std.mem.eql(u8, buf[0..4], &.{ 'E', 'N', 'D', 0 })) {
            return error.InvalidFooter;
        }

        return .{
            .marker = buf[0..4].*,
            .header_checksum = std.mem.readInt(u32, buf[4..8], .little),
            .data_checksum = std.mem.readInt(u32, buf[8..12], .little),
            .file_checksum = std.mem.readInt(u32, buf[12..16], .little),
            .file_size = std.mem.readInt(u64, buf[16..24], .little),
            .num_blocks = std.mem.readInt(u32, buf[24..28], .little),
        };
    }
};
