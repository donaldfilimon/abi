//! Database Feature Module
//!
//! Vector database and data persistence functionality

const std = @import("std");

// Core database components
pub const database = @import("database.zig");
pub const core = @import("core.zig");
pub const config = @import("config.zig");

// Database utilities and helpers
pub const utils = @import("utils.zig");
pub const db_helpers = @import("db_helpers.zig");
pub const unified = @import("unified.zig");
pub const wdbx_adapter = @import("wdbx_adapter.zig");
pub const tools = struct {
    pub const vector_search = @import("tools/vector_search.zig");
};

// Advanced database features
pub const database_sharding = @import("database_sharding.zig");

// HTTP and CLI interfaces
pub const http = @import("http.zig");
pub const cli = @import("cli.zig");

// Legacy compatibility removed - circular import fixed

test {
    std.testing.refAllDecls(@This());
}
