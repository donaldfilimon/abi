//! Public protocol API wiring.

const build_options = @import("build_options");

/// Model Context Protocol (MCP) server and client implementation.
pub const mcp = if (build_options.feat_mcp) @import("../protocols/mcp/mod.zig") else @import("../protocols/mcp/stub.zig");
/// Language Server Protocol (LSP) implementation.
pub const lsp = if (build_options.feat_lsp) @import("../protocols/lsp/mod.zig") else @import("../protocols/lsp/stub.zig");
/// Agent Communication Protocol (ACP) for multi-agent messaging.
pub const acp = if (build_options.feat_acp) @import("../protocols/acp/mod.zig") else @import("../protocols/acp/stub.zig");
/// High availability: leader election, failover, health monitoring.
pub const ha = if (build_options.feat_ha) @import("../protocols/ha/mod.zig") else @import("../protocols/ha/stub.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
