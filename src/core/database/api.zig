//! Stable boundaries for outside use.

const std = @import("std");

pub const RpcServer = struct {
    port: u16 = 9000,
    use_tls: bool = false,

    pub const RequestType = enum { ingest, query, snapshot, compaction };
    pub const ResponseStatus = enum { ok, err, pending };

    pub const RpcMessage = struct {
        id: u64,
        method: RequestType,
        payload: []const u8,
    };

    pub fn start(self: *RpcServer) !void {
        std.debug.print("WDBX RPC Server starting on port {d} (TLS={})\n", .{ self.port, self.use_tls });
        // Minimal server loop skeleton
        // const listener = try std.net.Address.listen(...);
    }

    pub fn handleHttpAdmin(self: *RpcServer, endpoint: []const u8) ![]const u8 {
        _ = self;
        if (std.mem.eql(u8, endpoint, "/admin/status")) return "{\"status\": \"ok\"}";
        if (std.mem.eql(u8, endpoint, "/admin/metrics")) return "{\"reclaimed_bytes\": 0}";
        return "404 Not Found";
    }

    pub fn handleOperatorApi(self: *RpcServer, command: []const u8) !void {
        _ = self;
        std.debug.print("Executing operator command: {s}\n", .{command});
    }
};
