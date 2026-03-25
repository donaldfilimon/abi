//! ZLS LSP client (JSON-RPC over stdio).
//!
//! Re-exports from `client/mod.zig`. The implementation is decomposed into:
//! - `client/mod.zig` — Client struct, init/deinit, connection management
//! - `client/requests.zig` — hover, completion, definition, references, rename, formatting
//! - `client/notifications.zig` — didOpen, didClose, didChange
//! - `client/transport.zig` — JSON-RPC send/receive, raw request/notification helpers

const client_mod = @import("client/mod.zig");

pub const Config = client_mod.Config;
pub const Response = client_mod.Response;
pub const Client = client_mod.Client;
pub const resolveWorkspaceRoot = client_mod.resolveWorkspaceRoot;
pub const resolvePath = client_mod.resolvePath;
pub const pathToUri = client_mod.pathToUri;

// Sub-modules for direct access.
pub const transport = client_mod.transport;
pub const requests = client_mod.requests;
pub const notifications = client_mod.notifications;

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
