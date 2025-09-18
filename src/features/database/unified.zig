//! WDBX Vector Database - Unified Module
//!
//! This is the main entry point for the WDBX vector database system.
//! It provides a clean, unified interface to all WDBX functionality.

const std = @import("std");

// Import all WDBX submodules
const cli = @import("cli.zig");
const wdbx_core = @import("core.zig");
const http = @import("http.zig");

// Import database module directly for standalone usage
const database = @import("./db_helpers.zig");

// Re-export main types and functions for easy access
pub const WdbxCLI = cli.WdbxCLI;
pub const WdbxHttpServer = http.WdbxHttpServer;
pub const Command = cli.Command;
pub const Options = cli.Options;
pub const ServerConfig = http.ServerConfig;

// Re-export core types
pub const WdbxError = wdbx_core.WdbxError;
pub const VERSION = wdbx_core.VERSION;
pub const OutputFormat = wdbx_core.OutputFormat;
pub const LogLevel = wdbx_core.LogLevel;
pub const Config = wdbx_core.Config;
pub const Timer = wdbx_core.Timer;
pub const Logger = wdbx_core.Logger;
pub const MemoryStats = wdbx_core.MemoryStats;

// Re-export main entry points
pub const main = cli.main;
pub const createServer = http.createServer;

// Re-export submodules for advanced usage
pub const wdbx = struct {
    pub const cli_module = cli;
    pub const core_module = wdbx_core;
    pub const http_module = http;
};

// Convenience functions
pub fn createCLI(allocator: std.mem.Allocator, options: Options) !*WdbxCLI {
    return try WdbxCLI.init(allocator, options);
}

pub fn createHttpServer(allocator: std.mem.Allocator, config: ServerConfig) !*WdbxHttpServer {
    return try WdbxHttpServer.init(allocator, config);
}

// Version information
pub const version = VERSION.string();
pub const version_major = VERSION.MAJOR;
pub const version_minor = VERSION.MINOR;
pub const version_patch = VERSION.PATCH;

// Quick start functions
pub fn quickStart(allocator: std.mem.Allocator) !void {
    const options = Options{};
    var cli_instance = try createCLI(allocator, options);
    defer cli_instance.deinit();

    try cli_instance.showHelp();
}

pub fn startHttpServer(allocator: std.mem.Allocator, port: u16) !*WdbxHttpServer {
    const config = ServerConfig{
        .port = port,
        .host = "127.0.0.1",
        .enable_auth = true,
        .enable_cors = true,
    };
    return try createHttpServer(allocator, config);
}

// Test suite
test "WDBX module initialization" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test CLI creation
    const options = Options{};
    var cli_instance = try createCLI(allocator, options);
    defer cli_instance.deinit();

    try testing.expectEqual(Command.help, cli_instance.options.command);

    // Test HTTP server creation
    const config = ServerConfig{};
    var server = try createHttpServer(allocator, config);
    defer server.deinit();

    try testing.expectEqual(@as(u16, 8080), server.config.port);
}

test "Version information" {
    try std.testing.expectEqualStrings("1.0.0-alpha", version);
    try std.testing.expectEqual(@as(u32, 1), version_major);
    try std.testing.expectEqual(@as(u32, 0), version_minor);
    try std.testing.expectEqual(@as(u32, 0), version_patch);
}

test "Quick start functions" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test quick start (should not crash)
    try quickStart(allocator);

    // Test HTTP server start
    var server = try startHttpServer(allocator, 8081);
    defer server.deinit();

    try testing.expectEqual(@as(u16, 8081), server.config.port);
}
