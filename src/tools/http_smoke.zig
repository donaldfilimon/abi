const std = @import("std");
const abi = @import("abi");
const http_client = @import("http_client");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var client = http_client.HttpClient.init(allocator, .{
        .connect_timeout_ms = 5000,
        .read_timeout_ms = 10000,
        .max_retries = 3,
        .initial_backoff_ms = 500,
        .max_backoff_ms = 4000,
        .user_agent = "WDBX-HTTP-Smoke/1.0",
        .follow_redirects = true,
        .verify_ssl = false,
        .verbose = true,
    });

    const base = "http://127.0.0.1:8080";

    try hit(&client, base ++ "/health");
    try hit(&client, base ++ "/stats");

    // Query example (vector length must match server default dimension 8 if initialized)
    const query_url = try std.fmt.allocPrint(allocator, "{s}/query?vec=1,2,3,4,5,6,7,8", .{base});
    defer allocator.free(query_url);
    try hit(&client, query_url);
}

fn hit(client: *http_client.HttpClient, url: []const u8) !void {
    std.debug.print("➡️  GET {s}\n", .{url});
    var resp = client.get(url) catch |err| {
        std.debug.print("❌ Request failed: {any}\n", .{err});
        return err;
    };
    defer resp.deinit();

    std.debug.print("✅ {s} -> HTTP {d}, body {d} bytes\n", .{ url, resp.status_code, resp.body.len });
}
