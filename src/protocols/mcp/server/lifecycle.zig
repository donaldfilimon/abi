//! MCP Server — core struct, lifecycle, and message routing.
//!
//! Defines the `Server` struct with init/deinit, tool/resource registration,
//! and the main run loop + message processing entry point.

const std = @import("std");
const types = @import("../types.zig");
const registration = @import("registration.zig");
const io_loop = @import("io_loop.zig");
const dispatch_mod = @import("dispatch.zig");
const json_write = @import("json_write.zig");

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
        // Free owned subscription keys before releasing the map.
        var it = self.subscriptions.keyIterator();
        while (it.next()) |key_ptr| {
            self.allocator.free(key_ptr.*);
        }
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
    /// The URI is duped into the server allocator so the caller's slice
    /// need not outlive this call.
    pub fn subscribeResource(self: *Self, uri: []const u8) !bool {
        const gop = try self.subscriptions.getOrPut(self.allocator, uri);
        if (gop.found_existing) return false;
        const owned_key = try self.allocator.dupe(u8, uri);
        gop.key_ptr.* = owned_key;
        gop.value_ptr.* = true;
        return true;
    }

    /// Unsubscribe from resource change notifications.
    pub fn unsubscribeResource(self: *Self, uri: []const u8) bool {
        const kv = self.subscriptions.fetchRemove(uri) orelse return false;
        self.allocator.free(kv.key);
        return true;
    }

    /// Check if a URI is subscribed.
    pub fn isSubscribed(self: *Self, uri: []const u8) bool {
        return self.subscriptions.contains(uri);
    }

    /// Notify subscribers that a resource has changed.
    /// Emits a JSON-RPC notification to the given writer when the URI is subscribed.
    pub fn notifyResourceChanged(self: *Self, uri: []const u8, writer: anytype) !void {
        if (!self.isSubscribed(uri)) return;
        var buf = std.ArrayListUnmanaged(u8).empty;
        defer buf.deinit(self.allocator);
        try buf.appendSlice(self.allocator, "{\"uri\":\"");
        try json_write.appendJsonEscaped(self.allocator, &buf, uri);
        try buf.appendSlice(self.allocator, "\"}");
        try types.writeNotification(writer, "notifications/resources/updated", buf.items);
    }

    /// Run the server loop — reads from stdin, writes to stdout.
    /// The caller must provide a Zig 0.17 I/O handle (from `std.Io.Threaded`).
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

test "subscribe and notify round-trip" {
    var server = Server.init(std.testing.allocator, "test", "0.1");
    defer server.deinit();

    // Not subscribed — notify should be a no-op
    var buf: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try server.notifyResourceChanged("abi://status", &writer);
    try std.testing.expectEqual(@as(usize, 0), writer.end);

    // Subscribe and notify
    const added = try server.subscribeResource("abi://status");
    try std.testing.expect(added);
    try std.testing.expect(server.isSubscribed("abi://status"));

    writer = std.Io.Writer.fixed(&buf);
    try server.notifyResourceChanged("abi://status", &writer);
    const out = buf[0..writer.end];
    try std.testing.expect(std.mem.indexOf(u8, out, "notifications/resources/updated") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "abi://status") != null);

    // Unsubscribe
    try std.testing.expect(server.unsubscribeResource("abi://status"));
    try std.testing.expect(!server.isSubscribed("abi://status"));
}

test {
    std.testing.refAllDecls(@This());
}
