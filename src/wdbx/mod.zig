//! WDBX-AI Vector Database - Unified Implementation
//!
//! This module consolidates all WDBX functionality into a single, high-performance
//! implementation with:
//! - HNSW indexing for fast approximate search
//! - Parallel search capabilities
//! - Advanced SIMD optimizations
//! - Production-ready features
//! - Comprehensive error handling

pub const cli = @import("cli.zig");
pub const core = @import("core.zig");
pub const http = @import("http.zig");

// Re-export main types
pub const Command = cli.Command;
pub const OutputFormat = cli.OutputFormat;
pub const LogLevel = cli.LogLevel;
pub const Options = cli.Options;
pub const WdbxCLI = cli.WdbxCLI;

// Re-export core WDBX functionality
pub const WdbxCore = core.WdbxCore;
pub const WdbxConfig = core.WdbxConfig;
pub const WdbxStats = core.WdbxStats;

// Re-export HTTP server
pub const WdbxHttpServer = http.WdbxHttpServer;

/// Main entry point for WDBX CLI
pub fn main() !void {
    try cli.main();
}

/// Initialize WDBX system
pub fn init(allocator: anytype, config: WdbxConfig) !*WdbxCore {
    return try core.WdbxCore.init(allocator, config);
}

/// Get WDBX version
pub fn getVersion() []const u8 {
    return "2.0.0";
}
