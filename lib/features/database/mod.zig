//! Database Feature Module
//!
//! Vector database with HNSW indexing, sharding, and secure servers

const std = @import("std");

pub const database = @import("database.zig");
pub const db_helpers = @import("db_helpers.zig");
pub const unified = @import("unified.zig");
pub const cli = @import("cli.zig");
pub const http = @import("http.zig");
pub const server = @import("server.zig");
pub const config = @import("config.zig");
pub const sharding = @import("database_sharding.zig");
pub const vector_search_gpu = @import("vector_search_gpu.zig");
pub const wdbx_adapter = @import("wdbx_adapter.zig");
pub const tools = struct {
    pub const vector_search = @import("tools/vector_search.zig");
};

test {
    std.testing.refAllDecls(@This());
}
