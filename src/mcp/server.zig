const std = @import("std");
const shutdown = @import("shutdown.zig");
const rpc = @import("rpc.zig");
const stdio_transport = @import("stdio_transport.zig");
const http_transport = @import("http_transport.zig");

// --- HTTP/SSE port + token env names (re-exported for `main`) ---

pub const HTTP_PORT_ENV = http_transport.HTTP_PORT_ENV;
pub const HTTP_TOKEN_ENV = http_transport.HTTP_TOKEN_ENV;

// --- Shutdown Coordination ---

pub const requestShutdown = shutdown.request;
pub const isShutdownRequested = shutdown.isRequested;
pub const installSignalHandlers = shutdown.installSignalHandlers;
pub const processJsonRpc = rpc.processJsonRpc;

// --- Stdio Transport ---

pub const runStdioLoop = stdio_transport.runStdioLoop;

// --- HTTP/SSE Transport ---

pub const runHttpServer = http_transport.runHttpServer;
pub const wakeHttpServer = http_transport.wakeHttpServer;
pub const setHttpPort = http_transport.setHttpPort;
pub const configuredHttpPort = http_transport.configuredHttpPort;
pub const setHttpToken = http_transport.setHttpToken;
pub const configuredHttpToken = http_transport.configuredHttpToken;

test {
    std.testing.refAllDecls(@This());
}
