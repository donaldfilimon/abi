const std = @import("std");
const web = @import("web_server");

test "web server test helper routes" {
    const alloc = std.testing.allocator;
    const server = try web.WebServer.init(alloc, .{});
    defer server.deinit();

    const body = try server.handlePathForTest("/health", alloc);
    defer alloc.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "healthy") != null);

    const body2 = try server.handlePathForTest("/api/status", alloc);
    defer alloc.free(body2);
    try std.testing.expect(std.mem.indexOf(u8, body2, "running") != null);
}
