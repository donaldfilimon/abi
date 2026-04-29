//! MCP (Model Context Protocol) service stub.
//!
//! This stub mirrors the full API of mod.zig but returns error.FeatureDisabled for all operations.
//! It is intended for use as a placeholder during development or testing when the full implementation is not yet available.

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

    /// Initializes a new MCP server instance.
    ///
    /// @param allocator - Memory allocator to use.
    /// @param name - Name of the server.
    /// @param version - Version of the server.
    /// @returns Initialized Server instance.
    pub fn init(self: *Server, allocator: std.mem.Allocator, name: []const u8, version: []const u8) Server {
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

    /// Deinitializes the server. Currently does nothing as there are no resources to clean up.
    ///
    /// @param self - Pointer to the Server instance.
    pub fn deinit(self: *Server) void {
        _ = self;
    }

    /// Adds a tool to the server. This feature is disabled.
    ///
    /// @param self - Pointer to the Server instance.
    /// @param tool - RegisteredTool to add.
    /// @returns error.FeatureDisabled.
    pub fn addTool(self: *Server, tool: RegisteredTool) !void {
        _ = self;
        _ = tool;
        return error.FeatureDisabled;
    }

    /// Adds a resource to the server. This feature is disabled.
    ///
    /// @param self - Pointer to the Server instance.
    /// @param resource - RegisteredResource to add.
    /// @returns error.FeatureDisabled.
    pub fn addResource(self: *Server, resource: RegisteredResource) !void {
        _ = self;
        _ = resource;
        return error.FeatureDisabled;
    }

    /// Subscribes to a resource. This feature is disabled and always returns false.
    ///
    /// @param self - Pointer to the Server instance.
    /// @param uri - URI of the resource to subscribe to.
    /// @returns false, indicating subscription failed.
    pub fn subscribeResource(self: *Server, uri: []const u8) !bool {
        _ = self;
        _ = uri;
        return false;
    }

    /// Unsubscribes from a resource. This feature is always successful and returns true.
    ///
    /// @param self - Pointer to the Server instance.
    /// @param uri - URI of the resource to unsubscribe from.
    /// @returns true, indicating successful unsubscription.
    pub fn unsubscribeResource(self: *Server, uri: []const u8) bool {
        _ = self;
        _ = uri;
        self.subscriptions.remove(uri);
        return true;
    }

    /// Checks if a resource is subscribed. This feature is always false.
    ///
    /// @param self - Pointer to the Server instance.
    /// @param uri - URI of the resource to check subscription.
    /// @returns false, indicating not subscribed.
    pub fn isSubscribed(self: *Server, uri: []const u8) bool {
        _ = self;
        _ = uri;
        return false;
    }

    /// Notifies that a resource has changed. This feature is disabled.
    ///
    /// @param self - Pointer to the Server instance.
    /// @param uri - URI of the resource that changed.
    /// @param change - Details of the change (not used).
    /// @returns error.FeatureDisabled.
    pub fn notifyResourceChanged(self: *Server, uri: []const u8, change: anytype) !void {
        _ = self;
        _ = uri;
        _ = change;
        return error.FeatureDisabled;
    }

    /// Starts the MCP server. This feature is disabled.
    ///
    /// @param self - Pointer to the Server instance.
    /// @param context - Context information (not used).
    /// @returns error.FeatureDisabled.
    pub fn run(self: *Server, context: anytype) !void {
        _ = self;
        _ = context;
        return error.FeatureDisabled;
    }

    /// Prints server run information. Currently does nothing.
    ///
    /// @param self - Pointer to the Server instance.
    pub fn runInfo(self: *Server) void {
        _ = self;
    }

    /// Processes an incoming message. This feature is disabled.
    ///
    /// @param self - Pointer to the Server instance.
    /// @param message - Message to process (not used).
    /// @param context - Context information (not used).
    /// @returns error.FeatureDisabled.
    pub fn processMessage(self: *Server, message: []const u8, context: anytype) !void {
        _ = self;
        _ = message;
        _ = context;
        return error.FeatureDisabled;
    }
};

/// Parity stub for createCombinedServer to mirror mod.zig surface.
pub fn createCombinedServer(_: std.mem.Allocator, _: []const u8) !Server {
    return error.FeatureDisabled;
}

/// Parity stub for createDatabaseServer to mirror mod.zig surface.
pub fn createDatabaseServer(_: std.mem.Allocator, _: []const u8) !Server {
    return error.FeatureDisabled;
}

/// Parity stub for createZlsServer to mirror mod.zig surface.
pub fn createZlsServer(_: std.mem.Allocator, _: []const u8) !Server {
    return error.FeatureDisabled;
}

/// Parity stub for createStatusServer to mirror mod.zig surface.
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

        /// Starts the stdio transport. This feature is disabled.
        ///
        /// @param self - Pointer to the Transport instance.
        /// @param context - Context information (not used).
        /// @returns error.FeatureDisabled.
        pub fn run(self: *@This(), context: anytype) !void {
            _ = self;
            _ = context;
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

        /// Starts the SSE transport. This feature is disabled.
        ///
        /// @param self - Pointer to the Transport instance.
        /// @param context - Context information (not used).
        /// @param config - SSE configuration.
        /// @returns error.FeatureDisabled.
        pub fn run(self: *@This(), context: anytype, config: Config) !void {
            _ = self;
            _ = context;
            _ = config;
            return error.FeatureDisabled;
        }

        /// Formats an SSE frame. This feature is disabled and always returns an empty slice.
        ///
        /// @param self - Pointer to the Transport instance.
        /// @param data - Data to format.
        /// @returns Empty slice.
        pub fn formatSseFrame(self: *@This(), data: []const u8) ![]u8 {
            _ = self;
            _ = data;
            return []u8{};
        }

        pub const heartbeat_comment = ":ping\n\n";
    };

    pub const Transport = union(enum) {
        stdio_transport: struct {},
        sse_transport: sse.Config,

        pub const StdioConfig = struct {};

        /// Starts the stdio transport. This feature is disabled.
        ///
        /// @param self - Pointer to the Transport instance.
        /// @param context - Context information (not used).
        /// @param config - Stdio configuration (not used).
        /// @returns error.FeatureDisabled.
        pub fn run(self: @This(), context: anytype, config: StdioConfig) !void {
            _ = self;
            _ = context;
            _ = config;
            return error.FeatureDisabled;
        }

        /// Initializes the stdio transport.
        ///
        /// @param self - Pointer to the Transport instance.
        /// @returns Initialized stdio transport.
        pub fn initStdio(self: *@This()) @This() {
            return .{ .stdio_transport = .{} };
        }

        /// Initializes the SSE transport with a given configuration.
        ///
        /// @param self - Pointer to the Transport instance.
        /// @param config - SSE configuration.
        /// @returns Initialized SSE transport.
        pub fn initSse(self: *@This(), config: sse.Config) @This() {
            return .{ .sse_transport = config };
        }

        /// Initializes the SSE transport with default configuration.
        ///
        /// @param self - Pointer to the Transport instance.
        /// @returns Initialized SSE transport with default configuration.
        pub fn initSseDefault(self: *@This()) @This() {
            return .{ .sse_transport = .{} };
        }
    };

    /// Starts the stdio transport. This feature is disabled.
    ///
    /// @param self - Pointer to the Transport instance.
    /// @param context - Context information (not used).
    /// @returns error.FeatureDisabled.
    pub fn runStdio(self: *@This(), context: anytype) !void {
        _ = self;
        _ = context;
        return error.FeatureDisabled;
    }

    /// Starts the SSE transport. This feature is disabled.
    ///
    /// @param self - Pointer to the Transport instance.
    /// @param context - Context information (not used).
    /// @param config - SSE configuration.
    /// @returns error.FeatureDisabled.
    pub fn runSse(self: *@This(), context: anytype, config: sse.Config) !void {
        _ = self;
        _ = context;
        _ = config;
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

test {
    std.testing.refAllDecls(@This());
}
