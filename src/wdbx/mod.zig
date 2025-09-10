//! WDBX Vector Database Module
//!
//! This module provides a unified, well-organized interface to the WDBX vector database
//! functionality, consolidating the best features from all WDBX implementations.

pub const cli = @import("cli.zig");
pub const http = @import("http.zig");
pub const core = @import("core.zig");
pub const config = @import("config.zig");

// Re-export main types and functions
pub const WdbxCLI = cli.WdbxCLI;
pub const WdbxHttpServer = http.WdbxHttpServer;
pub const Command = cli.Command;
pub const Options = cli.Options;
pub const ServerConfig = http.ServerConfig;

// Re-export configuration types
pub const WdbxConfig = config.WdbxConfig;
pub const ConfigManager = config.ConfigManager;
pub const ConfigUtils = config.ConfigUtils;

// Re-export main entry points
pub const main = cli.main;
pub const createServer = http.createServer;
