const std = @import("std");
const types = @import("types.zig");

pub fn dispatch(modules: anytype, path: []const u8, server: anytype, request: *std.http.Request, conn: anytype) !void {
    inline for (modules) |mod| {
        if (@hasDecl(mod, "endpoints")) {
            for (mod.endpoints) |ep| {
                if (std.mem.eql(u8, path, ep.path)) {
                    return ep.handler(server, request, conn);
                }
            }
        }
    }
    return error.NotFound;
}
