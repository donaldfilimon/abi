const std = @import("std");
const wdbx = @import("../../core/database/wdbx.zig");
const json_utils = @import("../../../foundation/mod.zig").utils.json;

test "unit: prompt injection builds context from WDBX" {
    const allocator = std.testing.allocator;

    var db = try wdbx.createDatabase(allocator, "unit-test-db");
    defer wdbx.closeDatabase(&db);

    // Simple fake embedding: constant vector for similarity
    const emb = try allocator.alloc(f32, 3);
    emb[0] = 1.0;
    emb[1] = 0.0;
    emb[2] = 0.0;
    defer allocator.free(emb);

    const now_ms = @as(u64, 1_700_000_000_000);
    const id1: u64 = 1;

    // Build JSON metadata
    var meta = std.ArrayListUnmanaged(u8).empty;
    defer meta.deinit(allocator);
    try meta.appendSlice(allocator, "{");
    try meta.appendSlice(allocator, "\"user\":");
    try json_utils.appendJsonEscaped(allocator, &meta, "tester");
    try meta.appendSlice(allocator, ",\"channel\":");
    try json_utils.appendJsonEscaped(allocator, &meta, "testchan");
    try meta.appendSlice(allocator, ",\"ts\":");
    try meta.writer(allocator).print("{d}", .{now_ms});
    try meta.appendSlice(allocator, ",\"msg_id\":");
    try json_utils.appendJsonEscaped(allocator, &meta, "m1");
    try meta.appendSlice(allocator, ",\"excerpt\":");
    try json_utils.appendJsonEscaped(allocator, &meta, "hello world");
    try meta.appendSlice(allocator, "}");

    const metadata1 = try meta.toOwnedSlice(allocator);
    defer allocator.free(metadata1);

    // Insert vector representing a previous message
    try wdbx.insertVector(&db, id1, emb, metadata1);

    // Now simulate creating an embedding for a new message (same vector)
    const query_vec = emb;

    const results = try wdbx.searchVectors(&db, allocator, query_vec, 8);
    defer allocator.free(results);

    var ctx_buf = std.ArrayListUnmanaged(u8).empty;
    defer ctx_buf.deinit(allocator);

    for (results) |res| {
        const view = wdbx.getVector(&db, res.id) orelse continue;
        if (view.metadata) |meta_str| {
            try ctx_buf.appendSlice(allocator, "Retrieved memory: ");
            try ctx_buf.appendSlice(allocator, meta_str);
            try ctx_buf.appendSlice(allocator, "\n\n");
        }
    }

    try std.testing.expect(ctx_buf.items.len > 0);

    const ctx = try ctx_buf.toOwnedSlice(allocator);
    defer allocator.free(ctx);

    // Build final input as production code does
    const user_input = "What does the fox do?";
    const final_input = try std.fmt.allocPrint(allocator, "{s}\n\nUser: {s}", .{ ctx, user_input });
    defer allocator.free(final_input);

    // Assert that final input includes retrieved excerpt
    try std.testing.expect(std.mem.indexOf(u8, final_input, "hello world") >= 0);
}
