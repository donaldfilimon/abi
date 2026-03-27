//! MCP Server — handler and registration type definitions.
//!
//! Contains the public type contracts for tool and resource handlers,
//! plus the registered wrappers that pair definitions with handlers.

const std = @import("std");
const types = @import("../types.zig");

/// Tool handler function signature
pub const ToolHandler = *const fn (
    allocator: std.mem.Allocator,
    params_json: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) anyerror!void;

/// Registered tool with metadata and handler
pub const RegisteredTool = struct {
    def: types.ToolDef,
    handler: ToolHandler,
};

/// Resource handler function signature
pub const ResourceHandler = *const fn (
    allocator: std.mem.Allocator,
    uri: []const u8,
    out: *std.ArrayListUnmanaged(u8),
) anyerror!void;

/// Registered resource with metadata and handler
pub const RegisteredResource = struct {
    def: types.ResourceDef,
    handler: ResourceHandler,
};

test {
    std.testing.refAllDecls(@This());
}
