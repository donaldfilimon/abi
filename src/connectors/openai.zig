const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub fn embedText(allocator: Allocator, base_url: []const u8, api_key: []const u8, model: []const u8, text: []const u8) ![]f32 {
    if (api_key.len == 0) return error.MissingApiKey;

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const url = try std.fmt.allocPrint(allocator, "{s}/embeddings", .{base_url});
    defer allocator.free(url);

    // Build JSON body: {"model":"...","input":"..."}
    const body = try std.fmt.allocPrint(allocator, "{{\"model\":\"{s}\",\"input\":\"{s}\"}}", .{ model, text });
    defer allocator.free(body);

    var req = try client.request(.POST, try std.Uri.parse(url), .{});
    defer req.deinit();

    // Set headers manually (since extra_headers is not used elsewhere in repo)
    // Set headers in request before sending
    req.headers.content_type = .{ .override = "application/json" };
    var auth_buf: [512]u8 = undefined;
    const auth_slice = try std.fmt.bufPrint(auth_buf[0..], "Bearer {s}", .{api_key});
    req.headers.authorization = .{ .override = auth_slice };

    try req.sendBodyComplete(body);
    var redirect_buf: [1024]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);

    if (response.head.status != .ok) return error.NetworkError;
    var list = std.array_list.Managed(u8).init(allocator);
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

    // Expected shape: {"data":[{"embedding":[...]}]}
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, resp, .{});
    defer parsed.deinit();
    const root_obj = parsed.value.object;
    const data_val = root_obj.get("data") orelse return error.InvalidResponse;
    const arr = data_val.array;
    if (arr.items.len == 0) return error.InvalidResponse;
    const first = arr.items[0].object;
    return parseEmbeddingArray(allocator, first.get("embedding").?);
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
