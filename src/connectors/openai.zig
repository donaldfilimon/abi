const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub fn embedText(allocator: Allocator, base_url: []const u8, api_key: []const u8, model: []const u8, text: []const u8) ![]f32 {
    if (api_key.len == 0) return error.MissingApiKey;

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const url = try std.fmt.allocPrint(allocator, "{s}/embeddings", .{base_url});
    defer allocator.free(url);

    // Build JSON body: {"model":"...","input":"..."}
    const body = try std.fmt.allocPrint(allocator, "{\"model\":\"{s}\",\"input\":\"{s}\"}", .{ model, text });
    defer allocator.free(body);

    var req = try client.request(.POST, try std.Uri.parse(url), .{});
    defer req.deinit();

    // Set headers manually (since extra_headers is not used elsewhere in repo)
    // Set headers in request before sending
    req.headers.content_type = .{ .override = "application/json" };
    const auth = try std.fmt.allocPrint(allocator, "Bearer {s}", .{api_key});
    defer allocator.free(auth);
    req.headers.authorization = .{ .override = auth };

    try req.sendBodyComplete(body);
    var redirect_buf: [1024]u8 = undefined;
    const response = try req.receiveHead(&redirect_buf);
    defer response.deinit();

    if (response.head.status != .ok) return error.NetworkError;
    var tmp: [8192]u8 = undefined;
    var rdr = response.reader(tmp[0..]);
    const resp = try rdr.readAllAlloc(allocator, 1024 * 1024);
    defer allocator.free(resp);

    // Expected shape: {"data":[{"embedding":[...]}]}
    var parser = std.json.Parser.init(allocator, false);
    defer parser.deinit();
    var tree = try parser.parse(resp);
    defer tree.deinit();

    const root_obj = tree.root.object;
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
        out[i] = @floatCast(arr.items[i].Float);
    }
    return out;
}
