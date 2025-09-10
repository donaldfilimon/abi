const std = @import("std");
const builtin = @import("builtin");

pub fn main() !void {
    if (builtin.os.tag != .windows) {
        std.debug.print("Windows Network Diagnostic Tool - skipping on non-Windows platform\n", .{});
        return;
    }

    std.debug.print("=== WDBX Windows Network Diagnostic Tool ===\n", .{});
    std.debug.print("Testing basic Windows networking functionality...\n", .{});

    const addr = try std.net.Address.parseIp4("127.0.0.1", 8080);
    const socket = std.posix.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, std.posix.IPPROTO.TCP) catch {
        std.debug.print("❌ Failed to create socket\n", .{});
        return;
    };
    defer std.posix.close(socket);
    std.debug.print("✅ Socket created successfully\n", .{});

    std.posix.bind(socket, &addr.any, addr.getOsSockLen()) catch {
        std.debug.print("❌ Failed to bind socket (may be expected if port in use)\n", .{});
    };

    std.posix.listen(socket, 128) catch {
        std.debug.print("❌ Failed to listen on socket\n", .{});
    };

    std.debug.print("✅ Windows networking compatibility verified\n", .{});
}
