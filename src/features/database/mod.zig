//! Database Feature Module
//!
//! Vector database with HNSW indexing, sharding, and secure servers

const std = @import("std");
const lifecycle = @import("../lifecycle.zig");

pub const database = @import("database.zig");
pub const db_helpers = @import("db_helpers.zig");
pub const unified = @import("unified.zig");
pub const cli = @import("cli.zig");
pub const http = @import("http.zig");
pub const server = @import("server.zig");
pub const config = @import("config.zig");
pub const sharding = @import("database_sharding.zig");
pub const wdbx_adapter = @import("wdbx_adapter.zig");
pub const search_operations = @import("search_operations.zig");
pub const tools = struct {
    pub const vector_search = @import("tools/vector_search.zig");
};

/// Initialize the database feature module
pub const init = lifecycle.init;

/// Deinitialize the database feature module
pub const deinit = lifecycle.deinit;

test {
    std.testing.refAllDecls(@This());
}
