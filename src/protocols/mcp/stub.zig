//! MCP (Model Context Protocol) service stub.
//!
//! Mirrors the full API of mod.zig, returning error.FeatureDisabled for all operations.

const std = @import("std");
pub const types = @import("types.zig");

/// MCP tool handler signature.
pub const ToolHandler = *const fn (
    allocator: std.mem.Allocator,
    params_json: ?std.json.ObjectMap,
    out: *std.ArrayListUnmanaged(u8),
) anyerror!void;

/// Registered tool with metadata and handler.
pub const RegisteredTool = struct {
    def: types.ToolDef,
    handler: ToolHandler,
};

/// MCP Server stub.
pub const Server = struct {
    allocator: std.mem.Allocator,
    tools: std.ArrayListUnmanaged(RegisteredTool),
    server_name: []const u8,
    server_version: []const u8,
    initialized: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        version: []const u8,
    ) Server {
        return .{
            .allocator = allocator,
            .tools = .empty,
            .server_name = name,
            .server_version = version,
            .initialized = false,
        };
    }

    pub fn deinit(self: *Server) void {
        _ = self;
    }

    pub fn addTool(self: *Server, _: RegisteredTool) !void {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn run(self: *Server, _: anytype) !void {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn runInfo(self: *Server) void {
        _ = self;
    }
};

pub fn createCombinedServer(_: std.mem.Allocator, _: []const u8) !Server {
    return error.FeatureDisabled;
}

pub fn createDatabaseServer(_: std.mem.Allocator, _: []const u8) !Server {
    return error.FeatureDisabled;
}

pub fn createZlsServer(_: std.mem.Allocator, _: []const u8) !Server {
    return error.FeatureDisabled;
}

pub const zls_bridge = struct {
    pub const createZlsServer = struct {
        pub fn f(_: std.mem.Allocator, _: []const u8) !Server {
            return error.FeatureDisabled;
        }
    }.f;
};

pub fn isEnabled() bool {
    return false;
}

pub const Context = struct {
    pub fn isEnabled() bool {
        return false;
    }
};

test {
    std.testing.refAllDecls(@This());
}
