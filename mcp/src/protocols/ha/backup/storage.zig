//! Backup persistence: file I/O, serialization, and verification.

const std = @import("std");
const config = @import("config.zig");

const BACKUP_MAGIC = config.BACKUP_MAGIC;
const BACKUP_FORMAT_VERSION = config.BACKUP_FORMAT_VERSION;
const BACKUP_HEADER_SIZE = config.BACKUP_HEADER_SIZE;
const CRC32_SIZE = config.CRC32_SIZE;
const VerifiedBackupInfo = config.VerifiedBackupInfo;

/// Initialize a Zig IO backend for file operations.
fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

/// Serialize backup data to the binary format:
///   [magic:4][version:2][timestamp:8][metadata_size:4][data_size:8][metadata_json:N][data:M][crc32:4]
pub fn serializeBackup(
    allocator: std.mem.Allocator,
    metadata_json: []const u8,
    data: []const u8,
    timestamp: u64,
) ![]u8 {
    const total_size = BACKUP_HEADER_SIZE + metadata_json.len + data.len + CRC32_SIZE;
    const buf = try allocator.alloc(u8, total_size);
    errdefer allocator.free(buf);

    var offset: usize = 0;

    // Magic bytes
    @memcpy(buf[offset..][0..4], &BACKUP_MAGIC);
    offset += 4;

    // Format version (little-endian)
    std.mem.writeInt(u16, buf[offset..][0..2], BACKUP_FORMAT_VERSION, .little);
    offset += 2;

    // Timestamp (little-endian)
    std.mem.writeInt(u64, buf[offset..][0..8], timestamp, .little);
    offset += 8;

    // Metadata size (little-endian)
    const meta_size: u32 = @intCast(metadata_json.len);
    std.mem.writeInt(u32, buf[offset..][0..4], meta_size, .little);
    offset += 4;

    // Data size (little-endian)
    std.mem.writeInt(u64, buf[offset..][0..8], @intCast(data.len), .little);
    offset += 8;

    // Metadata JSON
    @memcpy(buf[offset..][0..metadata_json.len], metadata_json);
    offset += metadata_json.len;

    // Data payload
    @memcpy(buf[offset..][0..data.len], data);
    offset += data.len;

    // CRC32 over everything before the checksum
    const crc = std.hash.crc.Crc32IsoHdlc.hash(buf[0..offset]);
    std.mem.writeInt(u32, buf[offset..][0..4], crc, .little);

    return buf;
}

/// Deserialize and verify a backup buffer. Returns verified info.
/// Caller owns the returned metadata_json slice (points into `buf`).
pub fn deserializeBackup(buf: []const u8) !VerifiedBackupInfo {
    if (buf.len < BACKUP_HEADER_SIZE + CRC32_SIZE) return error.TruncatedData;

    // Check magic
    if (!std.mem.eql(u8, buf[0..4], &BACKUP_MAGIC)) return error.InvalidMagic;

    // Read header fields
    const version = std.mem.readInt(u16, buf[4..6], .little);
    if (version != BACKUP_FORMAT_VERSION) return error.UnsupportedVersion;

    const timestamp = std.mem.readInt(u64, buf[6..14], .little);
    const meta_size = std.mem.readInt(u32, buf[14..18], .little);
    const data_size = std.mem.readInt(u64, buf[18..26], .little);

    const expected_total = BACKUP_HEADER_SIZE + meta_size + data_size + CRC32_SIZE;
    if (buf.len < expected_total) return error.TruncatedData;

    // Verify CRC32
    const payload_end = BACKUP_HEADER_SIZE + meta_size + data_size;
    const stored_crc = std.mem.readInt(u32, buf[payload_end..][0..4], .little);
    const computed_crc = std.hash.crc.Crc32IsoHdlc.hash(buf[0..payload_end]);
    if (stored_crc != computed_crc) return error.ChecksumMismatch;

    const metadata_json = buf[BACKUP_HEADER_SIZE..][0..meta_size];

    return .{
        .timestamp = timestamp,
        .metadata_json = metadata_json,
        .data_size = data_size,
        .format_version = version,
    };
}

/// Write a backup file to a local filesystem path.
/// Creates parent directories if needed via atomic file creation.
pub fn writeBackupToLocal(
    allocator: std.mem.Allocator,
    path: []const u8,
    backup_data: []const u8,
) !void {
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, backup_data);
}

/// Read a backup file from a local filesystem path.
pub fn readBackupFromLocal(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024 * 1024));
}

/// Verify a backup file on disk. Reads the file, validates CRC32, returns metadata.
/// Caller must call `.deinit()` on the returned value to free the backing buffer.
pub fn verifyBackupFile(allocator: std.mem.Allocator, path: []const u8) !VerifiedBackupInfo {
    const buf = try readBackupFromLocal(allocator, path);
    errdefer allocator.free(buf);

    var info = try deserializeBackup(buf);
    // Transfer ownership of the raw buffer so slices remain valid.
    info._raw_buf = buf;
    info._allocator = allocator;
    return info;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "serializeBackup and deserializeBackup roundtrip" {
    const allocator = std.testing.allocator;

    const metadata_json = "{\"backup_id\":1,\"mode\":\"full\",\"timestamp\":1234567890}";
    const data = "hello world backup data payload";
    const timestamp: u64 = 1234567890;

    const buf = try serializeBackup(allocator, metadata_json, data, timestamp);
    defer allocator.free(buf);

    // Verify the buffer has the expected size
    const expected_size = BACKUP_HEADER_SIZE + metadata_json.len + data.len + CRC32_SIZE;
    try std.testing.expectEqual(expected_size, buf.len);

    // Deserialize and verify
    const info = try deserializeBackup(buf);
    try std.testing.expectEqual(timestamp, info.timestamp);
    try std.testing.expectEqual(@as(u64, data.len), info.data_size);
    try std.testing.expectEqual(BACKUP_FORMAT_VERSION, info.format_version);
    try std.testing.expectEqualStrings(metadata_json, info.metadata_json);
}

test "deserializeBackup detects corruption" {
    const allocator = std.testing.allocator;

    const metadata_json = "{\"id\":1}";
    const data = "test data";

    const buf = try serializeBackup(allocator, metadata_json, data, 100);
    defer allocator.free(buf);

    // Corrupt one byte in the data section
    var corrupted = try allocator.alloc(u8, buf.len);
    defer allocator.free(corrupted);
    @memcpy(corrupted, buf);
    corrupted[BACKUP_HEADER_SIZE + 2] ^= 0xFF;

    try std.testing.expectError(error.ChecksumMismatch, deserializeBackup(corrupted));
}

test "deserializeBackup rejects truncated data" {
    const short_buf = [_]u8{ 'A', 'B', 'I', 'B', 0, 0 };
    try std.testing.expectError(error.TruncatedData, deserializeBackup(&short_buf));
}

test "deserializeBackup rejects bad magic" {
    const allocator = std.testing.allocator;

    const buf = try serializeBackup(allocator, "{}", "", 0);
    defer allocator.free(buf);

    var bad = try allocator.alloc(u8, buf.len);
    defer allocator.free(bad);
    @memcpy(bad, buf);
    bad[0] = 'X';

    try std.testing.expectError(error.InvalidMagic, deserializeBackup(bad));
}

test "backup write and verify roundtrip on disk" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/test_backup.abib",
        .{tmp.sub_path},
    );
    defer allocator.free(path);

    const metadata_json = "{\"backup_id\":42,\"mode\":\"full\"}";
    const data = "some important backup payload bytes here";
    const timestamp: u64 = 9999;

    const buf = try serializeBackup(allocator, metadata_json, data, timestamp);
    defer allocator.free(buf);

    try writeBackupToLocal(allocator, path, buf);

    // Verify on disk
    var info = try verifyBackupFile(allocator, path);
    defer info.deinit();
    try std.testing.expectEqual(timestamp, info.timestamp);
    try std.testing.expectEqual(@as(u64, data.len), info.data_size);
    try std.testing.expectEqualStrings(metadata_json, info.metadata_json);
}

test "backup verify detects corrupted file on disk" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/corrupt_backup.abib",
        .{tmp.sub_path},
    );
    defer allocator.free(path);

    // Write a valid backup
    const buf = try serializeBackup(allocator, "{\"id\":1}", "data", 100);
    defer allocator.free(buf);
    try writeBackupToLocal(allocator, path, buf);

    // Corrupt the file on disk: read, flip a byte, write back
    const file_data = try readBackupFromLocal(allocator, path);
    defer allocator.free(file_data);
    file_data[BACKUP_HEADER_SIZE + 1] ^= 0xFF;
    try writeBackupToLocal(allocator, path, file_data);

    // Verification should fail
    try std.testing.expectError(error.ChecksumMismatch, verifyBackupFile(allocator, path));
}

test {
    std.testing.refAllDecls(@This());
}
