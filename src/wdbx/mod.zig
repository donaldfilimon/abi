//! WDBX Vector Database Module Index
//!
//! This module provides a clean interface to all WDBX submodules.
//! For the main WDBX interface, use the unified module instead.

pub const core = @import("core.zig");
pub const database = @import("./db_helpers.zig");
pub const config = @import("config.zig");

// Re-export core types
pub const WdbxError = core.WdbxError;
pub const VERSION = core.VERSION;

// Re-export database types
pub const Db = database.Db;
pub const DatabaseError = database.DatabaseError;
pub const WdbxHeader = database.WdbxHeader;

// Re-export config types
pub const WdbxConfig = config.WdbxConfig;
pub const ConfigManager = config.ConfigManager;
pub const ConfigUtils = config.ConfigUtils;
pub const ConfigValidationError = config.ConfigValidationError;
