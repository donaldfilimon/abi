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
pub const unified = @import("unified.zig");

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

// Re-export unified WDBX functionality
pub const UnifiedWdbx = unified.UnifiedWdbx;
pub const UnifiedConfig = unified.UnifiedConfig;
pub const WdbxError = unified.WdbxError;
pub const SearchResult = unified.SearchResult;
pub const IndexType = unified.IndexType;
pub const Metrics = unified.Metrics;

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

/// Create unified WDBX instance
pub fn createUnified(allocator: std.mem.Allocator, path: []const u8, config: UnifiedConfig) !*UnifiedWdbx {
    return try unified.create(allocator, path, config);
}

/// Create unified WDBX with defaults
pub fn createWithDefaults(allocator: std.mem.Allocator, path: []const u8, dimension: u16) !*UnifiedWdbx {
    return try unified.createWithDefaults(allocator, path, dimension);
}

/// Create production WDBX instance
pub fn createProduction(allocator: std.mem.Allocator, path: []const u8, dimension: u16) !*UnifiedWdbx {
    return try unified.createProduction(allocator, path, dimension);
}

/// Get WDBX version
pub fn getVersion() []const u8 {
    return "2.0.0";
}
