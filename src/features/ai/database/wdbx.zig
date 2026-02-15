//! WDBX-backed training dataset utilities.
//!
//! Stores token blocks in WDBX metadata for durable, reusable training corpora.
//! Provides conversion helpers for token binaries and JSONL/text ingestion.

const std = @import("std");
const build_options = @import("build_options");
const tokenizer_mod = @import("../llm/tokenizer/mod.zig");

// Import the database module's wdbx wrapper (not this file!)
const db_wdbx = @import("../../database/wdbx.zig");

pub const DatasetError = error{
    DatabaseDisabled,
    InvalidFormat,
    PayloadTooLarge,
    FileNotFound,
    ReadFailed,
    WriteFailed,
    OutOfMemory,
};

const token_block_magic = "WDTK";
const token_block_version: u16 = 1;
const token_block_header_len = 16;

const TokenBlockHeader = struct {
    token_count: u32,
    text_len: u32,
};

pub const TokenBlock = struct {
    allocator: std.mem.Allocator,
    tokens: []u32,
    text: ?[]u8,

    pub fn deinit(self: *TokenBlock) void {
        self.allocator.free(self.tokens);
        if (self.text) |t| {
            self.allocator.free(t);
        }
        self.* = undefined;
    }
};

pub const WdbxTokenDataset = struct {
    allocator: std.mem.Allocator,
    handle: db_wdbx.DatabaseHandle,
    path: []const u8,
    next_id: u64,
    dirty: bool,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) DatasetError!WdbxTokenDataset {
        if (!build_options.enable_database) return error.DatabaseDisabled;

        const path_copy = try allocator.dupe(u8, path);
        errdefer allocator.free(path_copy);

        var handle = db_wdbx.createDatabaseWithConfig(allocator, path, .{
            .cache_norms = false,
            .initial_capacity = 0,
            .use_vector_pool = false,
            .thread_safe = false,
        }) catch |err| return mapWdbxError(err);
        errdefer db_wdbx.closeDatabase(&handle);

        if (fileExists(allocator, path)) {
            db_wdbx.restore(&handle, path) catch |err| switch (err) {
                error.FileNotFound => {},
                else => return mapWdbxError(err),
            };
        }

        const next_id = try computeNextId(allocator, &handle);

        return .{
            .allocator = allocator,
            .handle = handle,
            .path = path_copy,
            .next_id = next_id,
            .dirty = false,
        };
    }

    pub fn deinit(self: *WdbxTokenDataset) void {
        db_wdbx.closeDatabase(&self.handle);
        self.allocator.free(self.path);
        self.* = undefined;
    }

    pub fn save(self: *WdbxTokenDataset) DatasetError!void {
        if (!self.dirty) return;
        db_wdbx.backup(&self.handle, self.path) catch |err| return mapWdbxError(err);
        self.dirty = false;
    }

    pub fn appendTokens(self: *WdbxTokenDataset, tokens: []const u32, text: ?[]const u8) DatasetError!void {
        const metadata = try encodeTokenBlock(self.allocator, tokens, text);
        defer self.allocator.free(metadata);

        const empty_vector: []const f32 = &.{};
        db_wdbx.insertVector(&self.handle, self.next_id, empty_vector, metadata) catch |err| return mapWdbxError(err);
        self.next_id += 1;
        self.dirty = true;
    }

    pub fn importTokenBin(self: *WdbxTokenDataset, tokens: []const u32, block_tokens: u32) DatasetError!void {
        if (block_tokens == 0) return error.InvalidFormat;
        var offset: usize = 0;
        while (offset < tokens.len) {
            const remaining = tokens.len - offset;
            const take = @min(remaining, @as(usize, block_tokens));
            try self.appendTokens(tokens[offset .. offset + take], null);
            offset += take;
        }
    }

    pub fn collectTokens(self: *WdbxTokenDataset, max_tokens: usize) DatasetError![]u32 {
        const stats = db_wdbx.getStats(&self.handle);
        if (stats.count == 0) return self.allocator.alloc(u32, 0);

        const views = db_wdbx.listVectors(&self.handle, self.allocator, stats.count) catch |err| return mapWdbxError(err);
        defer self.allocator.free(views);

        var tokens = std.ArrayListUnmanaged(u32).empty;
        errdefer tokens.deinit(self.allocator);

        for (views) |view| {
            if (view.metadata) |meta| {
                _ = try appendTokensFromBlock(self.allocator, &tokens, meta, max_tokens);
                if (max_tokens > 0 and tokens.items.len >= max_tokens) break;
            }
        }

        return tokens.toOwnedSlice(self.allocator);
    }

    pub fn exportTokenBinFile(
        self: *WdbxTokenDataset,
        path: []const u8,
        max_tokens: usize,
    ) DatasetError!void {
        const tokens = try self.collectTokens(max_tokens);
        defer self.allocator.free(tokens);
        try writeTokenBinFile(self.allocator, path, tokens);
    }

    pub fn ingestText(
        self: *WdbxTokenDataset,
        tokenizer: *tokenizer_mod.Tokenizer,
        text: []const u8,
        block_tokens: u32,
    ) DatasetError!void {
        const tokens = try tokenizer.encode(self.allocator, text);
        defer self.allocator.free(tokens);
        try self.importTokenBin(tokens, block_tokens);
    }
};

pub fn encodeTokenBlock(
    allocator: std.mem.Allocator,
    tokens: []const u32,
    text: ?[]const u8,
) DatasetError![]u8 {
    if (tokens.len > std.math.maxInt(u32)) return error.PayloadTooLarge;
    const text_bytes = text orelse "";
    if (text_bytes.len > std.math.maxInt(u32)) return error.PayloadTooLarge;

    const total_len = token_block_header_len + tokens.len * @sizeOf(u32) + text_bytes.len;
    var buffer = try allocator.alloc(u8, total_len);

    @memcpy(buffer[0..token_block_magic.len], token_block_magic);
    std.mem.writeInt(u16, buffer[4..6], token_block_version, .little);
    std.mem.writeInt(u16, buffer[6..8], 0, .little);
    std.mem.writeInt(u32, buffer[8..12], @intCast(tokens.len), .little);
    std.mem.writeInt(u32, buffer[12..16], @intCast(text_bytes.len), .little);

    var offset: usize = token_block_header_len;
    for (tokens) |token| {
        std.mem.writeInt(u32, buffer[offset..][0..4], token, .little);
        offset += 4;
    }

    if (text_bytes.len > 0) {
        @memcpy(buffer[offset..][0..text_bytes.len], text_bytes);
    }

    return buffer;
}

pub fn decodeTokenBlock(allocator: std.mem.Allocator, bytes: []const u8) DatasetError!TokenBlock {
    const header = try readHeader(bytes);
    const token_bytes_len = @as(usize, header.token_count) * @sizeOf(u32);
    const tokens_start = token_block_header_len;
    const tokens_end = tokens_start + token_bytes_len;

    if (tokens_end > bytes.len) return error.InvalidFormat;
    const tokens_bytes = bytes[tokens_start..tokens_end];
    const tokens = try allocator.alloc(u32, header.token_count);
    var offset: usize = 0;
    var idx: usize = 0;
    while (idx < tokens.len) : (idx += 1) {
        const token = std.mem.readInt(u32, tokens_bytes[offset..][0..4], .little);
        tokens[idx] = token;
        offset += 4;
    }

    const text_start = tokens_end;
    const text_end = text_start + @as(usize, header.text_len);
    if (text_end > bytes.len) return error.InvalidFormat;
    const text = if (header.text_len == 0) null else blk: {
        const copy = try allocator.alloc(u8, header.text_len);
        @memcpy(copy, bytes[text_start..text_end]);
        break :blk copy;
    };

    return .{
        .allocator = allocator,
        .tokens = tokens,
        .text = text,
    };
}

pub fn appendTokensFromBlock(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u32),
    bytes: []const u8,
    max_tokens: usize,
) DatasetError!usize {
    const header = try readHeader(bytes);
    const token_bytes_len = @as(usize, header.token_count) * @sizeOf(u32);
    const tokens_start = token_block_header_len;
    const tokens_end = tokens_start + token_bytes_len;

    if (tokens_end > bytes.len) return error.InvalidFormat;

    var appended: usize = 0;
    var offset: usize = tokens_start;
    while (offset + 4 <= tokens_end) : (offset += 4) {
        if (max_tokens > 0 and out.items.len >= max_tokens) break;
        const token = std.mem.readInt(u32, bytes[offset..][0..4], .little);
        try out.append(allocator, token);
        appended += 1;
    }

    return appended;
}

pub fn readTokenBinFile(allocator: std.mem.Allocator, path: []const u8) DatasetError![]u32 {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const data = std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(512 * 1024 * 1024),
    ) catch |err| switch (err) {
        error.FileNotFound => return error.FileNotFound,
        else => return error.ReadFailed,
    };
    errdefer allocator.free(data);

    if (data.len % 4 != 0) return error.InvalidFormat;
    const count = data.len / 4;
    var tokens = try allocator.alloc(u32, count);

    var offset: usize = 0;
    var idx: usize = 0;
    while (idx < count) : (idx += 1) {
        tokens[idx] = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
    }

    allocator.free(data);
    return tokens;
}

pub fn writeTokenBinFile(allocator: std.mem.Allocator, path: []const u8, tokens: []const u32) DatasetError!void {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch return error.WriteFailed;
    defer file.close(io);

    const bytes = std.mem.sliceAsBytes(tokens);
    file.writeStreamingAll(io, bytes) catch return error.WriteFailed;
}

fn readHeader(bytes: []const u8) DatasetError!TokenBlockHeader {
    if (bytes.len < token_block_header_len) return error.InvalidFormat;
    if (!std.mem.eql(u8, bytes[0..4], token_block_magic)) return error.InvalidFormat;
    const version = std.mem.readInt(u16, bytes[4..6], .little);
    if (version != token_block_version) return error.InvalidFormat;

    return .{
        .token_count = std.mem.readInt(u32, bytes[8..12], .little),
        .text_len = std.mem.readInt(u32, bytes[12..16], .little),
    };
}

fn computeNextId(allocator: std.mem.Allocator, handle: *db_wdbx.DatabaseHandle) DatasetError!u64 {
    const stats = db_wdbx.getStats(handle);
    if (stats.count == 0) return 1;

    const views = db_wdbx.listVectors(handle, allocator, stats.count) catch |err| return mapWdbxError(err);
    defer allocator.free(views);

    var max_id: u64 = 0;
    for (views) |view| {
        if (view.id > max_id) max_id = view.id;
    }
    return max_id + 1;
}

fn fileExists(allocator: std.mem.Allocator, path: []const u8) bool {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

fn mapWdbxError(err: anyerror) DatasetError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.FileNotFound => error.FileNotFound,
        error.InvalidFormat => error.InvalidFormat,
        error.PayloadTooLarge => error.PayloadTooLarge,
        else => error.ReadFailed,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "encodeTokenBlock roundtrip without text" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 100, 200, 300, 400 };

    const encoded = try encodeTokenBlock(allocator, &tokens, null);
    defer allocator.free(encoded);

    var block = try decodeTokenBlock(allocator, encoded);
    defer block.deinit();

    try std.testing.expectEqual(@as(usize, 4), block.tokens.len);
    try std.testing.expectEqual(@as(u32, 100), block.tokens[0]);
    try std.testing.expectEqual(@as(u32, 200), block.tokens[1]);
    try std.testing.expectEqual(@as(u32, 300), block.tokens[2]);
    try std.testing.expectEqual(@as(u32, 400), block.tokens[3]);
    try std.testing.expect(block.text == null);
}

test "encodeTokenBlock roundtrip with text" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 1, 2, 3 };

    const encoded = try encodeTokenBlock(allocator, &tokens, "hello world");
    defer allocator.free(encoded);

    var block = try decodeTokenBlock(allocator, encoded);
    defer block.deinit();

    try std.testing.expectEqual(@as(usize, 3), block.tokens.len);
    try std.testing.expectEqual(@as(u32, 1), block.tokens[0]);
    try std.testing.expectEqualStrings("hello world", block.text.?);
}

test "encodeTokenBlock empty tokens" {
    const allocator = std.testing.allocator;
    const empty = [_]u32{};

    const encoded = try encodeTokenBlock(allocator, &empty, null);
    defer allocator.free(encoded);

    var block = try decodeTokenBlock(allocator, encoded);
    defer block.deinit();

    try std.testing.expectEqual(@as(usize, 0), block.tokens.len);
    try std.testing.expect(block.text == null);
}

test "decodeTokenBlock rejects invalid magic" {
    const bad_data = "BADM\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    const result = decodeTokenBlock(std.testing.allocator, bad_data);
    try std.testing.expectError(error.InvalidFormat, result);
}

test "decodeTokenBlock rejects truncated data" {
    const result = decodeTokenBlock(std.testing.allocator, "short");
    try std.testing.expectError(error.InvalidFormat, result);
}

test "appendTokensFromBlock respects max_tokens" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 10, 20, 30, 40, 50 };

    const encoded = try encodeTokenBlock(allocator, &tokens, null);
    defer allocator.free(encoded);

    var out = std.ArrayListUnmanaged(u32).empty;
    defer out.deinit(allocator);

    const appended = try appendTokensFromBlock(allocator, &out, encoded, 3);
    try std.testing.expectEqual(@as(usize, 3), appended);
    try std.testing.expectEqual(@as(usize, 3), out.items.len);
    try std.testing.expectEqual(@as(u32, 10), out.items[0]);
    try std.testing.expectEqual(@as(u32, 20), out.items[1]);
    try std.testing.expectEqual(@as(u32, 30), out.items[2]);
}

test "appendTokensFromBlock zero max means unlimited" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 1, 2, 3 };

    const encoded = try encodeTokenBlock(allocator, &tokens, null);
    defer allocator.free(encoded);

    var out = std.ArrayListUnmanaged(u32).empty;
    defer out.deinit(allocator);

    const appended = try appendTokensFromBlock(allocator, &out, encoded, 0);
    try std.testing.expectEqual(@as(usize, 3), appended);
    try std.testing.expectEqual(@as(usize, 3), out.items.len);
}

test "readHeader validates format" {
    // Valid header
    const allocator = std.testing.allocator;
    const tokens = [_]u32{42};
    const encoded = try encodeTokenBlock(allocator, &tokens, null);
    defer allocator.free(encoded);

    const header = try readHeader(encoded);
    try std.testing.expectEqual(@as(u32, 1), header.token_count);
    try std.testing.expectEqual(@as(u32, 0), header.text_len);
}

test "mapWdbxError maps known errors" {
    try std.testing.expectEqual(DatasetError.OutOfMemory, mapWdbxError(error.OutOfMemory));
    try std.testing.expectEqual(DatasetError.FileNotFound, mapWdbxError(error.FileNotFound));
    try std.testing.expectEqual(DatasetError.InvalidFormat, mapWdbxError(error.InvalidFormat));
    try std.testing.expectEqual(DatasetError.ReadFailed, mapWdbxError(error.Unexpected));
}
