//! WDBX Vector Database Module Index
//!
//! This module provides a clean interface to all WDBX submodules.
//! For the main WDBX interface, use the unified module instead.

pub const cli = @import("cli.zig");
pub const http = @import("../server/wdbx_http.zig");
pub const core = @import("core.zig");
<<<<<<< HEAD
pub const database = @import("./db_helpers.zig");
pub const unified = @import("unified.zig");
=======
pub const config = @import("config.zig");
>>>>>>> d9df96b0b53b2769af5f5da0390774a813448a2b

// Re-export main types and functions
pub const WdbxCLI = cli.WdbxCLI;
pub const WdbxHttpServer = http.WdbxHttpServer;
pub const Command = cli.Command;
pub const Options = cli.Options;
pub const ServerConfig = http.ServerConfig;
// Add missing re-exports used by root module
pub const OutputFormat = cli.OutputFormat;
pub const LogLevel = cli.LogLevel;

// Re-export configuration types
pub const WdbxConfig = config.WdbxConfig;
pub const ConfigManager = config.ConfigManager;
pub const ConfigUtils = config.ConfigUtils;

// Re-export database module under wdbx namespace
pub const database = @import("database.zig");

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
