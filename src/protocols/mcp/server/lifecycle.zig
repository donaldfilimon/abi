//! MCP Server — core struct, lifecycle, and message routing.
//!
//! Defines the `Server` struct with init/deinit, tool/resource registration,
//! and the main run loop + message processing entry point.

const std = @import("std");
const types = @import("../types.zig");
const registration = @import("registration.zig");
const io_loop = @import("io_loop.zig");
const dispatch_mod = @import("dispatch.zig");

pub const RegisteredTool = registration.RegisteredTool;
pub const RegisteredResource = registration.RegisteredResource;

/// Maximum message size accepted by the server (4 MB).
/// Messages exceeding this limit are rejected with a JSON-RPC Invalid Request error
/// to prevent denial-of-service via oversized stdin payloads.
pub const MAX_MESSAGE_SIZE: usize = 4 * 1024 * 1024;

/// MCP Server state
pub const Server = struct {
    allocator: std.mem.Allocator,
    tools: std.ArrayListUnmanaged(RegisteredTool),
    resources: std.ArrayListUnmanaged(RegisteredResource),
    subscriptions: std.StringHashMapUnmanaged(bool),
    server_name: []const u8,
    server_version: []const u8,
    initialized: bool,
    /// Optional authentication token. When set, tool and resource requests
    /// must include a matching `_auth_token` field in their params object.
    /// Protocol methods (initialize, ping, notifications) are always allowed.
    auth_token: ?[]const u8,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        name: []const u8,
        version: []const u8,
    ) Self {
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

    pub fn deinit(self: *Self) void {
        self.tools.deinit(self.allocator);
        self.resources.deinit(self.allocator);
        self.subscriptions.deinit(self.allocator);
    }

    /// Register a tool with the server
    pub fn addTool(self: *Self, tool: RegisteredTool) !void {
        try self.tools.append(self.allocator, tool);
    }

    /// Register a resource with the server
    pub fn addResource(self: *Self, resource: RegisteredResource) !void {
        try self.resources.append(self.allocator, resource);
    }

    /// Subscribe to resource change notifications for a given URI.
    pub fn subscribeResource(self: *Self, uri: []const u8) !bool {
        const gop = try self.subscriptions.getOrPut(self.allocator, uri);
        if (gop.found_existing) return false;
        gop.value_ptr.* = true;
        return true;
    }

    /// Unsubscribe from resource change notifications.
    pub fn unsubscribeResource(self: *Self, uri: []const u8) bool {
        return self.subscriptions.remove(uri);
    }

    /// Check if a URI is subscribed.
    pub fn isSubscribed(self: *Self, uri: []const u8) bool {
        return self.subscriptions.contains(uri);
    }

    /// Notify subscribers that a resource has changed.
    pub fn notifyResourceChanged(self: *Self, uri: []const u8, data: anytype) !void {
        _ = data;
        if (!self.isSubscribed(uri)) return;
        // In stdio mode, emit a JSON-RPC notification
        std.log.info("MCP: resource changed: {s}", .{uri});
    }

    /// Run the server loop — reads from stdin, writes to stdout.
    /// The caller must provide a Zig 0.16 I/O handle (from `std.Io.Threaded`).
    pub fn run(self: *Self, io: std.Io) !void {
        return io_loop.run(self, io);
    }

    /// Run without I/O — logs readiness (for environments without I/O backend).
    pub fn runInfo(self: *Self) void {
        io_loop.runInfo(self);
    }

    /// Process a single JSON-RPC message with size validation.
    /// This is the public entry point for message handling — it enforces
    /// MAX_MESSAGE_SIZE and delegates to the internal dispatch logic.
    /// Returns without error even when the message is invalid; error
    /// responses are written to `writer` per JSON-RPC 2.0 spec.
    pub fn processMessage(self: *Self, line: []const u8, writer: anytype) !void {
        if (line.len > MAX_MESSAGE_SIZE) {
            std.log.warn("MCP: rejecting oversized message ({d} bytes, limit {d})", .{ line.len, MAX_MESSAGE_SIZE });
            try types.writeError(
                writer,
                null,
                types.ErrorCode.invalid_request,
                "Invalid Request - message too large",
            );
            return;
        }
        return dispatch_mod.handleMessage(self, line, writer);
    }
};
