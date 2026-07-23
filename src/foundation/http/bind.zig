const std = @import("std");

extern fn getsockname(sockfd: std.posix.fd_t, addr: *std.posix.sockaddr, addrlen: *std.posix.socklen_t) c_int;

pub fn bindLoopback(io: std.Io) !struct { server: std.Io.net.Server, port: u16 } {
    const address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 0);
    var srv = try address.listen(io, .{ .mode = .stream, .reuse_address = true });
    errdefer srv.deinit(io);

    var addr: std.posix.sockaddr = undefined;
    var addrlen: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);
    if (getsockname(srv.socket.handle, &addr, &addrlen) != 0) return error.GetSockNameFailed;
    const addr_in: *const std.posix.sockaddr.in = @ptrCast(@alignCast(&addr));
    const port = std.mem.toNative(u16, addr_in.port, .big);
    if (port == 0) return error.PortIsZero;

    return .{ .server = srv, .port = port };
}

test {
    std.testing.refAllDecls(@This());
}
