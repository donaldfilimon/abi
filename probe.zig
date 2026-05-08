const std = @import("std");

pub fn main() !void {
    const bind = @hasDecl(std.posix.system, "bind");
    const socket = @hasDecl(std.posix.system, "socket");
    const close = @hasDecl(std.posix.system, "close");
    const listen = @hasDecl(std.posix.system, "listen");
    const accept = @hasDecl(std.posix.system, "accept");
    const read = @hasDecl(std.posix.system, "read");
    const write = @hasDecl(std.posix.system, "write");
    std.debug.print("system.bind: {any}\n", .{bind});
    std.debug.print("system.socket: {any}\n", .{socket});
    std.debug.print("system.close: {any}\n", .{close});
    std.debug.print("system.listen: {any}\n", .{listen});
    std.debug.print("system.accept: {any}\n", .{accept});
    std.debug.print("system.read: {any}\n", .{read});
    std.debug.print("system.write: {any}\n", .{write});
}
