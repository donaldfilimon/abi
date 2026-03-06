//! Stable boundaries for outside use.

const std = @import("std");

pub const RpcServer = struct {
    port: u16 = 9000,
    use_tls: bool = false,
    
    pub fn start(self: *RpcServer) !void {
        std.debug.print("WDBX RPC Server starting on port {d} (TLS={})\n", .{ self.port, self.use_tls });
        // FIXME: implement actual server loop with std.net
    }

    // FIXME: implement binary RPC protocol, HTTP admin endpoints, and operator APIs
};
