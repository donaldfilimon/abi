//! WDBX Operations for Aviva Agent
//! Implements vector storage, retrieval, and batch operations with modern Zig 0.17 patterns.

const std = @import("std");
const wdbx = @import("../../core/database/wdbx.zig");
const embeddings = @import("../../embeddings/mod.zig");
const json_utils = @import("../../../foundation/mod.zig").utils.json;

pub const log = std.log.scoped(.aviva_wdbx);

// ============================================================================
// Vector Operations
// ============================================================================

/// Store a conversation turn in WDBX with rich metadata
pub fn storeConversationTurn(
    handle: *wdbx.DatabaseHandle,
    allocator: std.mem.Allocator,
    id: u64,
    embedding: []const f32,
    user_id: []const u8,
    channel_id: []const u8,
    message_id: []const u8,
    content: []const u8,
    response: ?[]const u8,
    timestamp_ms: i64,
) !void {
    // Build JSON metadata using std.Io.Writer.Allocating (Zig 0.17 pattern)
    var meta_builder = std.Io.Writer.Allocating.init(allocator);
    defer meta_builder.deinit();

    const writer = meta_builder.writer();

    // Write JSON metadata
    writer.writeAll("{\"user\":\"") catch |err| {
        log.err("Failed to write user field: {any}", .{err});
        return error.MetadataBuildFailed;
    };
    json_utils.appendJsonEscaped(writer, user_id) catch |err| {
        log.err("Failed to escape user: {any}", .{err});
        return error.MetadataBuildFailed;
    };

    writer.writeAll("\",\"channel\":\"") catch return error.MetadataBuildFailed;
    json_utils.appendJsonEscaped(writer, channel_id) catch return error.MetadataBuildFailed;

    writer.print("\",\"msg_id\":\"{s}\",\"ts\":{d},\"content\":\"", .{
        message_id, timestamp_ms,
    }) catch return error.MetadataBuildFailed;
    json_utils.appendJsonEscaped(writer, content) catch return error.MetadataBuildFailed;

    if (response) |resp| {
        writer.writeAll("\",\"response\":\"") catch return error.MetadataBuildFailed;
        json_utils.appendJsonEscaped(writer, resp) catch return error.MetadataBuildFailed;
    }

    writer.writeAll("\"}") catch return error.MetadataBuildFailed;

    const metadata = try meta_builder.toOwnedSlice();
    defer allocator.free(metadata);

    // Insert into WDBX
    wdbx.insertVector(handle, id, embedding, metadata) catch |err| {
        log.err("WDBX insert failed: {any}", .{err});
        return err;
    };

    log.debug("Stored vector id={d} (user={s}, channel={s})", .{ id, user_id, channel_id });
}

/// Search for similar vectors with optional filtering
pub fn searchWithFilters(
    handle: *wdbx.DatabaseHandle,
    allocator: std.mem.Allocator,
    query_embedding: []const f32,
    top_k: usize,
    comptime T: type,
    context: *const T,
    filter_fn: *const fn ([]const u8, *const T) bool,
) ![][]const u8 {
    // Search WDBX
    const results = try wdbx.searchVectors(handle, allocator, query_embedding, top_k);
    defer allocator.free(results);

    // Filter and collect metadata
    var filtered = std.ArrayListUnmanaged([]const u8).empty;
    errdefer {
        for (filtered.items) |item| allocator.free(item);
        filtered.deinit(allocator);
    }

    for (results) |res| {
        const view = wdbx.getVector(handle, res.id) orelse continue;
        if (view.metadata) |meta| {
            // Apply filter
            if (filter_fn(meta, context)) {
                const owned = try allocator.dupe(u8, meta);
                try filtered.append(allocator, owned);
            }
        }
    }

    return try filtered.toOwnedSlice(allocator);
}

/// Batch store multiple conversation turns
pub fn batchStoreTurns(
    handle: *wdbx.DatabaseHandle,
    allocator: std.mem.Allocator,
    items: []const TurnItem,
    progress_callback: ?*const fn (processed: usize, total: usize) void,
) !void {
    if (items.len == 0) return;

    // Build batch items
    var batch = std.ArrayListUnmanaged(wdbx.BatchItem).empty;
    defer {
        for (batch.items) |item| {
            allocator.free(item.vector);
            if (item.metadata) |meta| allocator.free(meta);
        }
        batch.deinit(allocator);
    }

    for (items, 0..) |item, i| {
        // Generate embedding if not provided
        var emb: ?[]f32 = null;
        defer if (emb) |e| allocator.free(e);

        const embedding = if (item.embedding) |e| e else blk: {
            if (item.model) |model| {
                const generated = model.embed(item.content) catch |err| {
                    log.warn("Embedding generation failed for item {d}: {any}", .{ i, err });
                    break :blk @as([]f32, undefined);
                };
                emb = generated;
                break :blk generated;
            }
            break :blk @as([]f32, undefined);
        };

        if (embedding.ptr == undefined) {
            log.warn("Skipping item {d}: no embedding available", .{i});
            continue;
        }

        // Build metadata
        var meta_builder = std.Io.Writer.Allocating.init(allocator);
        defer meta_builder.deinit();

        meta_builder.writer().print("{{\"user\":\"{s}\",\"channel\":\"{s}\",\"msg_id\":\"{s}\",\"ts\":{d},\"content\":\"", .{
            item.user_id, item.channel_id, item.message_id, item.timestamp_ms,
        }) catch continue;
        json_utils.appendJsonEscaped(meta_builder.writer(), item.content) catch continue;
        meta_builder.writer().writeAll("\"}") catch continue;

        const metadata = meta_builder.toOwnedSlice() catch continue;
        errdefer allocator.free(metadata);

        // Copy embedding
        const emb_copy = try allocator.dupe(f32, embedding);
        errdefer allocator.free(emb_copy);

        // Add to batch
        try batch.append(allocator, .{
            .id = item.id,
            .vector = emb_copy,
            .metadata = metadata,
        });

        // Report progress
        if (progress_callback) |cb| cb(i + 1, items.len);
    }

    // Insert batch
    if (batch.items.len > 0) {
        wdbx.insertBatch(handle, batch.items) catch |err| {
            log.err("Batch insert failed: {any}", .{err});
            return err;
        };
        log.info("Batch stored {d} vectors", .{batch.items.len});
    }
}

// ============================================================================
// Types
// ============================================================================

pub const TurnItem = struct {
    id: u64,
    user_id: []const u8,
    channel_id: []const u8,
    message_id: []const u8,
    content: []const u8,
    timestamp_ms: i64,
    embedding: ?[]const f32 = null,
    model: ?*embeddings.EmbeddingModel = null,
};

// ============================================================================
// Filter Helpers
// ============================================================================

/// Filter by channel ID
pub fn filterByChannel(metadata: []const u8, channel_id: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, metadata, .{
        .ignore_unknown_fields = true,
    }) catch return false;
    defer parsed.deinit();

    const obj = json_utils.getRequiredObject(parsed.value) catch return false;
    const ch = json_utils.parseOptionalStringField(obj, "channel", std.heap.page_allocator) catch return false;
    defer if (ch) |c| std.heap.page_allocator.free(c);

    return if (ch) |c| std.mem.eql(u8, c, channel_id) else false;
}

/// Filter by time window
pub fn filterByTimeWindow(metadata: []const u8, window_ms: i64, now_ms: i64) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, metadata, .{
        .ignore_unknown_fields = true,
    }) catch return false;
    defer parsed.deinit();

    const obj = json_utils.getRequiredObject(parsed.value) catch return false;
    const ts_opt = json_utils.parseOptionalIntField(obj, "ts") catch return false;

    return if (ts_opt) |ts| (now_ms - ts) <= window_ms else false;
}

// ============================================================================
// Tests
// ============================================================================

test "storeConversationTurn basic" {
    const allocator = std.testing.allocator;

    var handle = try wdbx.createDatabase(allocator, "test-store-turn");
    defer wdbx.closeDatabase(&handle);

    const embedding: [4]f32 = .{ 0.1, 0.2, 0.3, 0.4 };

    try storeConversationTurn(
        &handle,
        allocator,
        1,
        &embedding,
        "user1",
        "chan1",
        "msg1",
        "Hello world",
        "Hi there!",
        123456789,
    );

    const stats = wdbx.getStats(&handle);
    try std.testing.expectEqual(@as(usize, 1), stats.count);
}

test "searchWithFilters channel filter" {
    const allocator = std.testing.allocator;

    var handle = try wdbx.createDatabase(allocator, "test-filter");
    defer wdbx.closeDatabase(&handle);

    // Store two turns in different channels
    const emb1: [2]f32 = .{ 1.0, 0.0 };
    try storeConversationTurn(&handle, allocator, 1, &emb1, "user1", "chan1", "msg1", "content1", null, 1000);

    const emb2: [2]f32 = .{ 0.0, 1.0 };
    try storeConversationTurn(&handle, allocator, 2, &emb2, "user2", "chan2", "msg2", "content2", null, 2000);

    // Search with channel filter
    const query: [2]f32 = .{ 0.9, 0.1 };
    const results = try searchWithFilters(
        &handle,
        allocator,
        &query,
        10,
        []const u8,
        "chan1",
        struct {
            fn filter(meta: []const u8, channel: *const []const u8) bool {
                return filterByChannel(meta, channel.*);
            }
        }.filter,
    );
    defer {
        for (results) |r| allocator.free(r);
        allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 1), results.len);
}

test "batchStoreTurns" {
    const allocator = std.testing.allocator;

    var handle = try wdbx.createDatabase(allocator, "test-batch");
    defer wdbx.closeDatabase(&handle);

    const items = [_]TurnItem{
        .{
            .id = 1,
            .user_id = "user1",
            .channel_id = "chan1",
            .message_id = "msg1",
            .content = "Hello",
            .timestamp_ms = 1000,
            .embedding = &[_]f32{ 0.1, 0.2 },
        },
        .{
            .id = 2,
            .user_id = "user2",
            .channel_id = "chan2",
            .message_id = "msg2",
            .content = "World",
            .timestamp_ms = 2000,
            .embedding = &[_]f32{ 0.3, 0.4 },
        },
    };

    var progress_count: usize = 0;
    const callback = struct {
        fn progress(processed: usize, total: usize) void {
            progress_count += 1;
        }
    }.progress;

    try batchStoreTurns(&handle, allocator, &items, callback);

    const stats = wdbx.getStats(&handle);
    try std.testing.expectEqual(@as(usize, 2), stats.count);
    try std.testing.expect(progress_count >= 1);
}
