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

/// Resource handler signature.
pub const ResourceHandler = *const fn (
    allocator: std.mem.Allocator,
    uri: []const u8,
    out: *std.ArrayListUnmanaged(u8),
) anyerror!void;

/// Registered resource with metadata and handler.
pub const RegisteredResource = struct {
    def: types.ResourceDef,
    handler: ResourceHandler,
};

/// MCP Server stub.
pub const Server = struct {
    allocator: std.mem.Allocator,
    tools: std.ArrayListUnmanaged(RegisteredTool),
    resources: std.ArrayListUnmanaged(RegisteredResource),
    subscriptions: std.StringHashMapUnmanaged(bool),
    server_name: []const u8,
    server_version: []const u8,
    initialized: bool,
    auth_token: ?[]const u8,

    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        version: []const u8,
    ) Server {
        return .{
            .allocator = allocator,
            .tools = .empty,
            .resources = .empty,
            .subscriptions = .empty,
            .server_name = name,
            .server_version = version,
            .initialized = false,
            .auth_token = null,
        };
    }

    pub fn deinit(self: *Server) void {
        _ = self;
    }

    pub fn addTool(self: *Server, _: RegisteredTool) !void {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn addResource(self: *Server, _: RegisteredResource) !void {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn subscribeResource(self: *Server, _: []const u8) !bool {
        _ = self;
        return error.FeatureDisabled;
    }

    pub fn unsubscribeResource(self: *Server, _: []const u8) bool {
        _ = self;
        return false;
    }

    pub fn isSubscribed(self: *Server, _: []const u8) bool {
        _ = self;
        return false;
    }

    pub fn notifyResourceChanged(self: *Server, _: []const u8, _: anytype) !void {
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

    pub fn processMessage(self: *Server, _: []const u8, _: anytype) !void {
        _ = self;
        return error.FeatureDisabled;
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

pub fn createStatusServer(_: std.mem.Allocator, _: []const u8) !Server {
    return error.FeatureDisabled;
}

pub const zls_bridge = struct {
    pub const createZlsServer = struct {
        pub fn f(_: std.mem.Allocator, _: []const u8) !Server {
            return error.FeatureDisabled;
        }
    }.f;
};

/// Transport stub — mirrors the real transport module's public surface.
pub const transport = struct {
    pub const stdio = struct {
        pub const Config = struct {
            read_buf_size: usize = 65536,
            write_buf_size: usize = 65536,
        };

        pub fn run(_: anytype, _: anytype) !void {
            return error.FeatureDisabled;
        }
    };

    pub const sse = struct {
        pub const Config = struct {
            port: u16 = 8081,
            host: []const u8 = "127.0.0.1",
            heartbeat_interval_s: u32 = 30,
            max_clients: u16 = 64,
            max_body_size: usize = 4 * 1024 * 1024,
        };

        pub const SseResponseCollector = struct {
            pub fn init(_: std.mem.Allocator) @This() {
                return .{};
            }
            pub fn deinit(_: *@This()) void {}
            pub fn reset(_: *@This()) void {}
        };

        pub fn run(_: anytype, _: anytype, _: Config) !void {
            return error.FeatureDisabled;
        }

        pub fn formatSseFrame(_: std.mem.Allocator, _: []const u8) ![]u8 {
            return error.FeatureDisabled;
        }

        pub const heartbeat_comment = ":ping\n\n";
    };

    pub const Transport = union(enum) {
        stdio_transport: struct {},
        sse_transport: sse.Config,

        pub const StdioConfig = struct {};

        pub fn run(_: @This(), _: anytype, _: anytype) !void {
            return error.FeatureDisabled;
        }

        pub fn initStdio() @This() {
            return .{ .stdio_transport = .{} };
        }

        pub fn initSse(config: sse.Config) @This() {
            return .{ .sse_transport = config };
        }

        pub fn initSseDefault() @This() {
            return .{ .sse_transport = .{} };
        }
    };

    pub fn runStdio(_: anytype, _: anytype) !void {
        return error.FeatureDisabled;
    }

    pub fn runSse(_: anytype, _: anytype, _: sse.Config) !void {
        return error.FeatureDisabled;
    }
};

// Parity stubs for isEnabled and Context to mirror mod.zig surface
pub fn isEnabled() bool {
    return false;
}
pub const Context = struct {
    pub fn isEnabled() bool {
        return false;
    }
};

pub const registry = @import("registry.zig");

test {
    std.testing.refAllDecls(@This());
}
