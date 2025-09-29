const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub fn embedText(allocator: Allocator, host: []const u8, model: []const u8, text: []const u8) ![]f32 {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const url = try std.fmt.allocPrint(allocator, "{s}/api/embeddings", .{host});
    defer allocator.free(url);

    // Build JSON body: {"model":"...","input":"..."}
    const body = try std.fmt.allocPrint(allocator, "{{\"model\":\"{s}\",\"input\":\"{s}\"}}", .{ model, text });
    defer allocator.free(body);

    var req = try client.request(.POST, try std.Uri.parse(url), .{});
    defer req.deinit();

    // Set headers
    req.headers.content_type = .{ .override = "application/json" };

    // 0.15: send body and receive head
    try req.sendBodyComplete(body);
    var redirect_buf: [1024]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);

    if (response.head.status != .ok) return error.NetworkError;
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();
    var buf: [8192]u8 = undefined;
    const rdr = response.reader(&.{});
    while (true) {
        const slice: []u8 = buf[0..];
        var slices = [_][]u8{slice};
        const n = rdr.readVec(slices[0..]) catch |err| switch (err) {
            error.ReadFailed => return error.NetworkError,
            error.EndOfStream => 0,
        };
        if (n == 0) break;
        try list.appendSlice(buf[0..n]);
    }
    const resp = try list.toOwnedSlice();

    // Expected minimal shape: {"embedding":[...]} or {"data":[{"embedding":[...]}]}
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, resp, .{});
    defer parsed.deinit();
    const root_obj = parsed.value.object;
    if (root_obj.get("embedding")) |val| {
        return parseEmbeddingArray(allocator, val);
    }
    if (root_obj.get("data")) |data_val| {
        const arr = data_val.array;
        if (arr.items.len > 0) {
            const first = arr.items[0].object;
            return parseEmbeddingArray(allocator, first.get("embedding").?);
        }
    }
    return error.InvalidResponse;
}

fn parseEmbeddingArray(allocator: Allocator, v: std.json.Value) ![]f32 {
    const arr = v.array;
    const out = try allocator.alloc(f32, arr.items.len);
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        out[i] = @floatCast(arr.items[i].float);
    }
    return out;
}
