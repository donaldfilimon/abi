//! Shared CLI helpers used by the `abi` binary and integration tests.

const std = @import("std");
const build_options = @import("build_options");
const acp = @import("protocols/acp/mod.zig");

const default_host = "127.0.0.1";
const default_port: u16 = 8080;

fn formatServeAddress(allocator: std.mem.Allocator, host: []const u8, port: u16) ![]u8 {
    const bracketed = host.len >= 2 and host[0] == '[' and host[host.len - 1] == ']';
    if (bracketed or std.mem.indexOfScalar(u8, host, ':') == null) {
        return std.fmt.allocPrint(allocator, "{s}:{d}", .{ host, port });
    }

    return std.fmt.allocPrint(allocator, "[{s}]:{d}", .{ host, port });
}

pub fn isServeInvocation(args: []const [:0]const u8) bool {
    if (args.len == 0) return false;
    if (std.mem.eql(u8, args[0], "serve")) return true;
    return args.len >= 2 and std.mem.eql(u8, args[0], "acp") and std.mem.eql(u8, args[1], "serve");
}

pub fn parseServeAddress(allocator: std.mem.Allocator, args: []const [:0]const u8) ![]u8 {
    var host: []const u8 = default_host;
    var port = default_port;
    var explicit_address: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--addr")) {
            if (i + 1 >= args.len) return error.InvalidServeArgs;
            explicit_address = args[i + 1];
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--host")) {
            if (i + 1 >= args.len) return error.InvalidServeArgs;
            host = args[i + 1];
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--port")) {
            if (i + 1 >= args.len) return error.InvalidServeArgs;
            port = std.fmt.parseInt(u16, args[i + 1], 10) catch return error.InvalidServePort;
            i += 1;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "-")) return error.InvalidServeArgs;
        if (explicit_address == null) {
            explicit_address = arg;
        } else {
            return error.InvalidServeArgs;
        }
    }

    if (explicit_address) |address| {
        return allocator.dupe(u8, address);
    }

    return formatServeAddress(allocator, host, port);
}

pub fn runServe(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    const address = try parseServeAddress(allocator, args);
    defer allocator.free(address);

    const io_options: std.Io.Threaded.InitOptions = .{ .environ = std.process.Environ.empty };
    var io_backend = blk: {
        const InitResult = @TypeOf(std.Io.Threaded.init(allocator, io_options));
        if (@typeInfo(InitResult) == .error_union) {
            break :blk try std.Io.Threaded.init(allocator, io_options);
        }
        break :blk std.Io.Threaded.init(allocator, io_options);
    };
    defer io_backend.deinit();

    const url = try std.fmt.allocPrint(allocator, "http://{s}", .{address});
    defer allocator.free(url);

    const card = acp.AgentCard{
        .name = "abi",
        .description = "ABI Agent Communication Protocol server",
        .version = build_options.package_version,
        .url = url,
        .capabilities = .{},
    };

    try acp.serveHttp(allocator, io_backend.io(), address, card);
}

test "serve invocation recognises both aliases" {
    try std.testing.expect(isServeInvocation(&.{"serve"}));
    try std.testing.expect(isServeInvocation(&.{ "acp", "serve" }));
    try std.testing.expect(!isServeInvocation(&.{ "acp", "status" }));
}

test "serve address parsing honors addr and port flags" {
    const port_args = [_][:0]const u8{ "--port", "9090" };
    const port_address = try parseServeAddress(std.testing.allocator, &port_args);
    defer std.testing.allocator.free(port_address);
    try std.testing.expectEqualStrings("127.0.0.1:9090", port_address);

    const ipv6_args = [_][:0]const u8{ "--host", "::1", "--port", "9090" };
    const ipv6_address = try parseServeAddress(std.testing.allocator, &ipv6_args);
    defer std.testing.allocator.free(ipv6_address);
    try std.testing.expectEqualStrings("[::1]:9090", ipv6_address);

    const addr_args = [_][:0]const u8{ "--addr", "0.0.0.0:8080" };
    const explicit_address = try parseServeAddress(std.testing.allocator, &addr_args);
    defer std.testing.allocator.free(explicit_address);
    try std.testing.expectEqualStrings("0.0.0.0:8080", explicit_address);
}

test {
    std.testing.refAllDecls(@This());
}
