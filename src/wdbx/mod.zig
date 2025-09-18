//! WDBX Vector Database Module Index
//!
//! This module provides a clean interface to all WDBX submodules.
//! For the main WDBX interface, use the unified module instead.

pub const cli = @import("cli.zig");
pub const http = @import("http.zig");
pub const core = @import("core.zig");
pub const database = @import("./db_helpers.zig");
pub const unified = @import("unified.zig");

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

// Re-export database types
pub const Db = database.Db;
pub const DatabaseError = database.DatabaseError;
pub const WdbxHeader = database.WdbxHeader;

// Re-export main entry points
pub const main_cli = cli.main;
pub const createServer = http.createServer;

// Re-export unified interface
pub const createCLI = unified.createCLI;
pub const createHttpServer = unified.createHttpServer;
pub const quickStart = unified.quickStart;
pub const startHttpServer = unified.startHttpServer;
