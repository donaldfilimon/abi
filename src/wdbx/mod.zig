//! WDBX Vector Database Module Index
//!
//! This module provides a clean interface to all WDBX submodules.
//! For the main WDBX interface, use @import("@wdbx.zig") instead.

pub const cli = @import("cli.zig");
pub const http = @import("http.zig");
pub const core = @import("core.zig");

// Re-export main types and functions
pub const WdbxCLI = cli.WdbxCLI;
pub const WdbxHttpServer = http.WdbxHttpServer;
pub const Command = cli.Command;
pub const Options = cli.Options;
pub const ServerConfig = http.ServerConfig;

// Re-export core types
pub const WdbxError = core.WdbxError;
pub const VERSION = core.VERSION;
pub const OutputFormat = core.OutputFormat;
pub const LogLevel = core.LogLevel;
pub const Config = core.Config;
pub const Timer = core.Timer;
pub const Logger = core.Logger;
pub const MemoryStats = core.MemoryStats;

// Re-export main entry points
pub const main = cli.main;
pub const createServer = http.createServer;
