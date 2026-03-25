//! MCP Transport Layer
//!
//! Provides a unified transport interface for the MCP server. Supports:
//! - **stdio** — newline-delimited JSON-RPC over stdin/stdout (default, for CLI)
//! - **sse** — Server-Sent Events over HTTP (for browser/web clients)
//!
//! ## Usage
//! ```zig
//! const transport = mcp.transport;
//!
//! // Stdio (default)
//! try transport.Transport.runStdio(&server, io);
//!
//! // SSE on custom port
//! try transport.Transport.runSse(&server, io, .{ .port = 9090 });
//! ```

const std = @import("std");
pub const stdio = @import("stdio.zig");
pub const sse = @import("sse.zig");

/// Transport selection for the MCP server.
pub const Transport = union(enum) {
    /// Standard I/O transport (stdin/stdout). Default for CLI usage.
    stdio_transport: StdioConfig,
    /// Server-Sent Events over HTTP. For browser/web clients.
    sse_transport: sse.Config,

    pub const StdioConfig = struct {};

    /// Run the selected transport with the given MCP server and I/O backend.
    pub fn run(self: Transport, server: anytype, io: std.Io) !void {
        switch (self) {
            .stdio_transport => {
                return stdio.run(server, io);
            },
            .sse_transport => |config| {
                return sse.run(server, io, config);
            },
        }
    }

    /// Create a stdio transport (the default).
    pub fn initStdio() Transport {
        return .{ .stdio_transport = .{} };
    }

    /// Create an SSE transport with the given configuration.
    pub fn initSse(config: sse.Config) Transport {
        return .{ .sse_transport = config };
    }

    /// Create an SSE transport with default configuration (port 8081).
    pub fn initSseDefault() Transport {
        return .{ .sse_transport = .{} };
    }
};

/// Convenience: run the MCP server with stdio transport.
pub fn runStdio(server: anytype, io: std.Io) !void {
    return stdio.run(server, io);
}

/// Convenience: run the MCP server with SSE transport.
pub fn runSse(server: anytype, io: std.Io, config: sse.Config) !void {
    return sse.run(server, io, config);
}

// ===================================================================
// Tests
// ===================================================================

test "Transport union initStdio" {
    const t = Transport.initStdio();
    try std.testing.expect(t == .stdio_transport);
}

test "Transport union initSse default" {
    const t = Transport.initSseDefault();
    switch (t) {
        .sse_transport => |config| {
            try std.testing.expectEqual(@as(u16, 8081), config.port);
            try std.testing.expectEqualStrings("127.0.0.1", config.host);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "Transport union initSse custom" {
    const t = Transport.initSse(.{ .port = 9090, .host = "0.0.0.0" });
    switch (t) {
        .sse_transport => |config| {
            try std.testing.expectEqual(@as(u16, 9090), config.port);
            try std.testing.expectEqualStrings("0.0.0.0", config.host);
        },
        else => return error.TestUnexpectedResult,
    }
}

test {
    std.testing.refAllDecls(@This());
}
