const std = @import("std");

pub fn postMessage(allocator: std.mem.Allocator, token: []const u8, channel_id: []const u8, content: []const u8) !void {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    // Prepare JSON payload
    var body_buf = std.ArrayList(u8).init(allocator);
    defer body_buf.deinit();
    try std.json.stringify(.{ .content = content }, .{}, body_buf.writer());

    const url = try std.fmt.allocPrint(
        allocator,
        "https://discord.com/api/v10/channels/{s}/messages",
        .{channel_id},
    );
    defer allocator.free(url);

    const auth_value = try std.fmt.allocPrint(allocator, "Bot {s}", .{token});
    defer allocator.free(auth_value);

    const headers = [_]std.http.Header{
        .{ .name = "Authorization", .value = auth_value },
        .{ .name = "Content-Type", .value = "application/json" },
    };

    var response = std.ArrayList(u8).init(allocator);
    defer response.deinit();

    const result = try client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .headers = .{ .authorization = .omit },
        .extra_headers = &headers,
        .payload = body_buf.items,
        .response_storage = .dynamic(&response),
    });

    std.debug.print("Discord API response status: {d}\n", .{@intFromEnum(result.status)});
    if (response.items.len > 0) {
        std.debug.print("Response body: {s}\n", .{response.items});
    }
}
