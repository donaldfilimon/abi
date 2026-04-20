//! MCP Tool Registry
//!
//! Provides a modular registration pattern for MCP tools, allowing handler modules
//! to export tool definitions that are auto-discovered at comptime.
//!
//! ## Usage
//! ```zig
//! const registry = @import("registry.zig");
//!
//! // Export tools from your handler module
//! pub const tools = [_]registry.ToolDef{...};
//!
//! // In factory, register all tools at once
//! try registry.registerTools(server, .{ status, database, ai, discord });
//! ```

const std = @import("std");
const Server = @import("server.zig").Server;

// =============================================================================
// Registry Types
// =============================================================================

/// Definition of an MCP tool for registry export
pub const ToolDef = struct {
    name: []const u8,
    description: []const u8,
    /// JSON Schema for the tool's input parameters
    input_schema: []const u8,
    /// Handler function signature must match Server.ToolHandler
    handler: @import("server/registration.zig").ToolHandler,
};

/// Resource definition for registry export
pub const ResourceDef = struct {
    uri: []const u8,
    name: []const u8,
    description: []const u8,
    mime_type: []const u8 = "text/plain",
    /// Handler function signature must match Server.ResourceHandler
    handler: @import("server/registration.zig").ResourceHandler,
};

/// Combined module exports interface
pub const ModuleExports = struct {
    /// Optional tools array
    tools: ?[]const ToolDef = null,
    /// Optional resources array
    resources: ?[]const ResourceDef = null,
};

// =============================================================================
// Registration Functions
// =============================================================================

/// Register all tools and resources from the given modules.
/// Uses comptime reflection to discover exported `tools` and `resources` arrays.
pub fn registerModules(server: *Server, modules: anytype) !void {
    inline for (modules) |mod| {
        // Register tools if module exports them
        if (@hasDecl(mod, "tools")) {
            inline for (mod.tools) |t| {
                try server.addTool(.{
                    .def = .{
                        .name = t.name,
                        .description = t.description,
                        .input_schema = t.input_schema,
                    },
                    .handler = t.handler,
                });
            }
        }

        // Register resources if module exports them
        if (@hasDecl(mod, "resources")) {
            inline for (mod.resources) |r| {
                try server.addResource(.{
                    .def = .{
                        .uri = r.uri,
                        .name = r.name,
                        .description = r.description,
                        .mime_type = r.mime_type,
                    },
                    .handler = r.handler,
                });
            }
        }
    }
}

/// Register only tools from the given modules
pub fn registerTools(server: *Server, modules: anytype) !void {
    inline for (modules) |mod| {
        if (@hasDecl(mod, "tools")) {
            inline for (mod.tools) |t| {
                try server.addTool(.{
                    .def = .{
                        .name = t.name,
                        .description = t.description,
                        .input_schema = t.input_schema,
                    },
                    .handler = t.handler,
                });
            }
        }
    }
}

/// Register only resources from the given modules
pub fn registerResources(server: *Server, modules: anytype) !void {
    inline for (modules) |mod| {
        if (@hasDecl(mod, "resources")) {
            inline for (mod.resources) |r| {
                try server.addResource(.{
                    .def = .{
                        .uri = r.uri,
                        .name = r.name,
                        .description = r.description,
                        .mime_type = r.mime_type,
                    },
                    .handler = r.handler,
                });
            }
        }
    }
}

// =============================================================================
// Comptime Discovery Helpers
// =============================================================================

/// Count total tools exported by all modules (comptime)
pub fn countTools(comptime modules: anytype) usize {
    var count: usize = 0;
    inline for (modules) |mod| {
        if (@hasDecl(mod, "tools")) {
            count += mod.tools.len;
        }
    }
    return count;
}

/// Count total resources exported by all modules (comptime)
pub fn countResources(comptime modules: anytype) usize {
    var count: usize = 0;
    inline for (modules) |mod| {
        if (@hasDecl(mod, "resources")) {
            count += mod.resources.len;
        }
    }
    return count;
}

/// Collect all tool names into a buffer (comptime)
pub fn collectToolNames(comptime modules: anytype, buf: *std.ArrayListUnmanaged([]const u8), allocator: std.mem.Allocator) !void {
    inline for (modules) |mod| {
        if (@hasDecl(mod, "tools")) {
            inline for (mod.tools) |t| {
                try buf.append(allocator, t.name);
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "ToolDef format" {
    const def = ToolDef{
        .name = "test_tool",
        .description = "A test tool",
        .input_schema = "{}",
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: ?std.json.ObjectMap, out: *std.ArrayListUnmanaged(u8)) !void {
                try out.appendSlice(std.testing.allocator, "ok");
            }
        }.handle,
    };

    try std.testing.expectEqualStrings("test_tool", def.name);
    try std.testing.expectEqualStrings("A test tool", def.description);
    try std.testing.expectEqualStrings("{}", def.input_schema);
}

test "ResourceDef format" {
    const def = ResourceDef{
        .uri = "abi://test",
        .name = "Test Resource",
        .description = "A test resource",
        .mime_type = "application/json",
        .handler = struct {
            fn handle(_: std.mem.Allocator, _: []const u8, out: *std.ArrayListUnmanaged(u8)) !void {
                try out.appendSlice(std.testing.allocator, "ok");
            }
        }.handle,
    };

    try std.testing.expectEqualStrings("abi://test", def.uri);
    try std.testing.expectEqualStrings("Test Resource", def.name);
    try std.testing.expectEqualStrings("application/json", def.mime_type);
}

test "countTools with single module" {
    const test_mod = struct {
        pub const tools = [_]ToolDef{
            .{ .name = "tool1", .description = "desc1", .input_schema = "{}", .handler = undefined },
            .{ .name = "tool2", .description = "desc2", .input_schema = "{}", .handler = undefined },
        };
    };

    const count = countTools(.{test_mod});
    try std.testing.expectEqual(@as(usize, 2), count);
}

test "countTools with multiple modules" {
    const mod1 = struct {
        pub const tools = [_]ToolDef{
            .{ .name = "tool1", .description = "", .input_schema = "{}", .handler = undefined },
        };
    };
    const mod2 = struct {
        pub const tools = [_]ToolDef{
            .{ .name = "tool2", .description = "", .input_schema = "{}", .handler = undefined },
            .{ .name = "tool3", .description = "", .input_schema = "{}", .handler = undefined },
        };
    };

    const count = countTools(.{ mod1, mod2 });
    try std.testing.expectEqual(@as(usize, 3), count);
}

test "countTools skips modules without tools" {
    const mod_with = struct {
        pub const tools = [_]ToolDef{
            .{ .name = "tool1", .description = "", .input_schema = "{}", .handler = undefined },
        };
    };
    const mod_without = struct {
        pub const not_tools = "hello";
    };

    const count = countTools(.{ mod_with, mod_without });
    try std.testing.expectEqual(@as(usize, 1), count);
}

test {
    std.testing.refAllDecls(@This());
}
