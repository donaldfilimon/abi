//! Search Index Persistence
//!
//! Binary serialization/deserialization for inverted indexes.
//! File format: magic header + terms section + documents section.

 const std = @import("std");
const index = @import("index.zig");

const InvertedIndex = index.InvertedIndex;
const PostingList = index.PostingList;
const DocumentMeta = index.DocumentMeta;

/// Magic bytes identifying the BM25 index file format.
pub const INDEX_MAGIC = [4]u8{ 'S', 'R', 'C', 'H' };
pub const INDEX_VERSION: u16 = 1;

/// Convert a path slice to a sentinel-terminated buffer for POSIX calls.
fn toPathZ(path: []const u8) error{IoError}![std.fs.max_path_bytes:0]u8 {
    var buf: [std.fs.max_path_bytes:0]u8 = [_:0]u8{0} ** std.fs.max_path_bytes;
    if (path.len >= buf.len) return error.IoError;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    return buf;
}

fn openFileZ(path: []const u8, flags: std.posix.O, mode: std.posix.mode_t) error{IoError}!std.posix.fd_t {
    const path_buf = try toPathZ(path);
    const path_z: [:0]const u8 = path_buf[0..path.len :0];
    return std.posix.openatZ(std.posix.AT.FDCWD, path_z, flags, mode) catch return error.IoError;
}

fn closeFile(fd: std.posix.fd_t) void {
    _ = std.posix.system.close(fd);
}

pub fn unlinkFile(path: []const u8) void {
    const path_buf = toPathZ(path) catch return;
    const path_z: [*:0]const u8 = @ptrCast(path_buf[0..path.len :0]);
    _ = std.posix.system.unlink(path_z);
}

fn writeAll(fd: std.posix.fd_t, buf: []const u8) !void {
    var written: usize = 0;
    while (written < buf.len) {
        const rc = std.posix.system.write(fd, buf[written..].ptr, buf[written..].len);
        if (rc < 0) return error.IoError;
        const n: usize = @intCast(rc);
        if (n == 0) return error.IoError;
        written += n;
    }
}

fn readAll(fd: std.posix.fd_t, buf: []u8) !void {
    var total: usize = 0;
    while (total < buf.len) {
        const n = std.posix.read(fd, buf[total..]) catch return error.IoError;
        if (n == 0) return error.IoError;
        total += n;
    }
}

fn writeU16(fd: std.posix.fd_t, val: u16) !void {
    const bytes = std.mem.toBytes(std.mem.nativeTo(u16, val, .little));
    try writeAll(fd, &bytes);
}

fn writeU32(fd: std.posix.fd_t, val: u32) !void {
    const bytes = std.mem.toBytes(std.mem.nativeTo(u32, val, .little));
    try writeAll(fd, &bytes);
}

fn readU16(fd: std.posix.fd_t) !u16 {
    var bytes: [2]u8 = undefined;
    try readAll(fd, &bytes);
    return std.mem.readInt(u16, &bytes, .little);
}

fn readU32(fd: std.posix.fd_t) !u32 {
    var bytes: [4]u8 = undefined;
    try readAll(fd, &bytes);
    return std.mem.readInt(u32, &bytes, .little);
}

/// Serialize a named inverted index to disk at the given path.
///
/// File format:
/// ```
/// [4B] Magic "SRCH"
/// [2B] Version (1)
/// [4B] term_count
/// [4B] doc_count
/// [4B] total_doc_length (for BM25 avgdl, truncated to u32)
/// --- Terms section ---
/// per term:
///   [2B] term_len
///   [term_len bytes] term string
///   [4B] posting_count
///   per posting:
///     [4B] doc_id_len  (length of doc_id string)
///     [doc_id_len bytes] doc_id
///     [4B] term_freq
///     [4B] doc_len
/// --- Documents section ---
/// per document:
///   [4B] doc_id_len
///   [doc_id_len bytes] doc_id
///   [4B] content_len
///   [content_len bytes] content
///   [4B] term_count
/// ```
pub fn saveIndex(idx: *InvertedIndex, path: []const u8) error{ IoError, IndexNotFound, FeatureDisabled }!void {
    const fd = openFileZ(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644) catch
        return error.IoError;
    defer closeFile(fd);

    // Header
    writeAll(fd, &INDEX_MAGIC) catch return error.IoError;
    writeU16(fd, INDEX_VERSION) catch return error.IoError;
    writeU32(fd, @intCast(idx.term_index.count())) catch return error.IoError;
    writeU32(fd, @intCast(idx.documents.count())) catch return error.IoError;
    const total_trunc: u32 = @intCast(@min(idx.total_terms, std.math.maxInt(u32)));
    writeU32(fd, total_trunc) catch return error.IoError;

    // Terms section
    var term_iter = idx.term_index.iterator();
    while (term_iter.next()) |entry| {
        const term = entry.key_ptr.*;
        const pl = entry.value_ptr.*;

        writeU16(fd, @intCast(term.len)) catch return error.IoError;
        writeAll(fd, term) catch return error.IoError;
        writeU32(fd, @intCast(pl.postings.items.len)) catch return error.IoError;

        for (pl.postings.items) |posting| {
            writeU32(fd, @intCast(posting.doc_id.len)) catch return error.IoError;
            writeAll(fd, posting.doc_id) catch return error.IoError;
            writeU32(fd, posting.term_freq) catch return error.IoError;
            writeU32(fd, posting.doc_len) catch return error.IoError;
        }
    }

    // Documents section
    var doc_iter = idx.documents.iterator();
    while (doc_iter.next()) |entry| {
        const doc = entry.value_ptr.*;

        writeU32(fd, @intCast(doc.id.len)) catch return error.IoError;
        writeAll(fd, doc.id) catch return error.IoError;
        writeU32(fd, @intCast(doc.content.len)) catch return error.IoError;
        writeAll(fd, doc.content) catch return error.IoError;
        writeU32(fd, doc.term_count) catch return error.IoError;
    }
}

/// Deserialize a named index from disk. Creates a new InvertedIndex and
/// populates it from the file.
pub fn loadIndex(allocator: std.mem.Allocator, name: []const u8, path: []const u8) error{ IoError, IndexCorrupted, IndexAlreadyExists, OutOfMemory, FeatureDisabled }!*InvertedIndex {
    const fd = openFileZ(path, .{ .ACCMODE = .RDONLY }, 0) catch
        return error.IoError;
    defer closeFile(fd);

    // Validate header
    var magic: [4]u8 = undefined;
    readAll(fd, &magic) catch return error.IndexCorrupted;
    if (!std.mem.eql(u8, &magic, &INDEX_MAGIC)) return error.IndexCorrupted;

    const version = readU16(fd) catch return error.IndexCorrupted;
    if (version != INDEX_VERSION) return error.IndexCorrupted;

    const term_count = readU32(fd) catch return error.IndexCorrupted;
    const doc_count = readU32(fd) catch return error.IndexCorrupted;
    const total_doc_length = readU32(fd) catch return error.IndexCorrupted;

    // Create the new index
    const idx = InvertedIndex.create(allocator, name) catch return error.OutOfMemory;
    errdefer idx.destroy();

    idx.total_terms = total_doc_length;

    // Read terms section
    for (0..term_count) |_| {
        const term_len = readU16(fd) catch return error.IndexCorrupted;
        if (term_len == 0) return error.IndexCorrupted;

        const term = allocator.alloc(u8, term_len) catch return error.OutOfMemory;
        errdefer allocator.free(term);
        readAll(fd, term) catch return error.IndexCorrupted;

        const posting_count = readU32(fd) catch return error.IndexCorrupted;

        const pl = allocator.create(PostingList) catch return error.OutOfMemory;
        errdefer allocator.destroy(pl);
        pl.* = .{ .postings = .empty, .doc_freq = @intCast(posting_count) };

        for (0..posting_count) |_| {
            const did_len = readU32(fd) catch return error.IndexCorrupted;
            const did = allocator.alloc(u8, did_len) catch return error.OutOfMemory;
            readAll(fd, did) catch {
                allocator.free(did);
                return error.IndexCorrupted;
            };

            const tf = readU32(fd) catch {
                allocator.free(did);
                return error.IndexCorrupted;
            };
            const dl = readU32(fd) catch {
                allocator.free(did);
                return error.IndexCorrupted;
            };

            pl.postings.append(allocator, .{
                .doc_id = did,
                .term_freq = tf,
                .doc_len = dl,
            }) catch {
                allocator.free(did);
                return error.OutOfMemory;
            };
        }

        idx.term_index.put(allocator, term, pl) catch return error.OutOfMemory;
    }

    // Read documents section
    for (0..doc_count) |_| {
        const did_len = readU32(fd) catch return error.IndexCorrupted;
        const did = allocator.alloc(u8, did_len) catch return error.OutOfMemory;
        readAll(fd, did) catch {
            allocator.free(did);
            return error.IndexCorrupted;
        };

        const content_len = readU32(fd) catch {
            allocator.free(did);
            return error.IndexCorrupted;
        };
        const content = allocator.alloc(u8, content_len) catch {
            allocator.free(did);
            return error.OutOfMemory;
        };
        readAll(fd, content) catch {
            allocator.free(content);
            allocator.free(did);
            return error.IndexCorrupted;
        };

        const tc = readU32(fd) catch {
            allocator.free(content);
            allocator.free(did);
            return error.IndexCorrupted;
        };

        const doc = allocator.create(DocumentMeta) catch {
            allocator.free(content);
            allocator.free(did);
            return error.OutOfMemory;
        };
        doc.* = .{ .id = did, .content = content, .term_count = tc };
        idx.documents.put(allocator, did, doc) catch {
            allocator.free(content);
            allocator.free(did);
            allocator.destroy(doc);
            return error.OutOfMemory;
        };
    }

    return idx;
}
