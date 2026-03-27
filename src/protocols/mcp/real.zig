//! MCP (Model Context Protocol) Service
//!
//! Provides a JSON-RPC 2.0 server over stdio for exposing ABI framework
//! tools to MCP-compatible AI clients (Claude Desktop, Cursor, etc.).
//!
//! ## Usage
//! ```bash
//! abi mcp serve                          # Start MCP server (stdio)
//! echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}' | abi mcp serve
//! ```
//!
//! ## Exposed Tools
//! - `db_*` — Database tools
//! - `zls_*` — ZLS LSP tools (hover, completion, definition, etc.)
//! - `discord_*` — Discord REST API tools (messages, channels, guilds, etc.)

pub const types = @import("types.zig");
pub const Server = @import("server.zig").Server;
pub const RegisteredTool = @import("server.zig").RegisteredTool;
pub const RegisteredResource = @import("server.zig").RegisteredResource;
pub const ResourceHandler = @import("server.zig").ResourceHandler;
pub const ToolHandler = @import("server.zig").ToolHandler;
pub const zls_bridge = @import("zls_bridge.zig");
pub const transport = @import("transport/mod.zig");

pub const createZlsServer = zls_bridge.createZlsServer;

const factories = @import("factories.zig");
pub const createStatusServer = factories.createStatusServer;
pub const createCombinedServer = factories.createCombinedServer;
pub const createDatabaseServer = factories.createDatabaseServer;
pub const createDiscordServer = factories.createDiscordServer;

// Sub-modules for direct access to handlers
pub const handlers = struct {
    pub const status = @import("handlers/status.zig");
    pub const database = @import("handlers/database.zig");
    pub const discord = @import("handlers/discord.zig");
};

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
