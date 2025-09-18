const std = @import("std");
const testing = std.testing;
const abi = @import("abi");
const wdbx_http = abi.wdbx.http;
pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const config = wdbx_http.ServerConfig{};
    var server = try wdbx_http.WdbxHttpServer.init(allocator, config);
    defer server.deinit();
    try server.openDatabase("test_http.db");
    try server.start();
    std.debug.print("WDBX HTTP server ready at http://{s}:{d}\n", .{ config.host, config.port });
    std.debug.print("This sample does not spin a network listener; use tests for request behaviour.\n", .{});
}

test "http server add and query" {
    const allocator = testing.allocator;
    var server = try wdbx_http.WdbxHttpServer.init(allocator, .{});
    defer server.deinit();

    const add_body = "{\"vector\":[1.0,2.0,3.0]}";
    var response = try server.respond("POST", "/add", add_body);
    defer response.deinit(allocator);
    try testing.expectEqual(@as(u16, 200), response.status);

    const query = try server.respond("GET", "/query?vec=1.0,2.0,3.0&k=1", "");
    defer query.deinit(allocator);
    try testing.expect(std.mem.indexOf(u8, query.body, "\"matches\"") != null);
}

test "http server stats" {
    const allocator = testing.allocator;
    var server = try wdbx_http.WdbxHttpServer.init(allocator, .{});
    defer server.deinit();

    const stats = try server.respond("GET", "/stats", "");
    defer stats.deinit(allocator);
    try testing.expect(std.mem.indexOf(u8, stats.body, "\"vectors\":0") != null);
}
