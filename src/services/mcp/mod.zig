//! MCP (Model Context Protocol) Service module switcher.

const build_options = @import("build_options");

const impl = if (build_options.feat_mcp)
    @import("real.zig")
else
    @import("stub.zig");

pub const types = impl.types;
pub const Server = impl.Server;
pub const RegisteredTool = impl.RegisteredTool;
pub const zls_bridge = impl.zls_bridge;
pub const createZlsServer = impl.createZlsServer;
pub const createCombinedServer = impl.createCombinedServer;
pub const createDatabaseServer = impl.createDatabaseServer;

pub fn isEnabled() bool {
    return build_options.feat_mcp;
}

pub const Context = struct {
    pub fn isEnabled() bool {
        return build_options.feat_mcp;
    }
};

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
