const std = @import("std");
const http = @import("http");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var server = try http.WdbxHttpServer.init(allocator, .{});
    defer server.deinit();
    try server.start();
}
