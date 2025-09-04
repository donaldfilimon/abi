//! Database Module - Unified Vector Database Implementation
//!
//! This module consolidates all database functionality from the various WDBX
//! implementations into a single, coherent interface.

const std = @import("std");

// Import the enhanced database implementation
pub const enhanced_db = @import("enhanced_db.zig");

// Re-export the main database implementation
pub const Db = @import("database.zig").Db;
pub const DbError = @import("database.zig").DatabaseError;
pub const Result = @import("database.zig").Result;
pub const WdbxHeader = @import("database.zig").WdbxHeader;

// Enhanced database features
pub const EnhancedDb = enhanced_db.EnhancedDb;
pub const DatabaseConfig = enhanced_db.DatabaseConfig;
pub const SearchOptions = enhanced_db.SearchOptions;

/// Initialize database module
pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    // Database module initialization if needed
}

/// Cleanup database module
pub fn deinit() void {
    // Database module cleanup if needed
}

/// Create a new database instance with enhanced features
pub fn createEnhanced(allocator: std.mem.Allocator, path: []const u8, config: DatabaseConfig) !*EnhancedDb {
    return try EnhancedDb.init(allocator, path, config);
}

/// Create a standard database instance
pub fn createStandard(path: []const u8, create_if_not_exists: bool) !Db {
    return try Db.open(path, create_if_not_exists);
}

test "Database module" {
    const testing = std.testing;
    
    try init(testing.allocator);
    defer deinit();
    
    // Test that we can create database instances
    const test_file = "test_db_module.wdbx";
    defer std.fs.cwd().deleteFile(test_file) catch {};
    
    var db = try createStandard(test_file, true);
    defer db.close();
    try db.init(64);
    
    // Test basic operations
    const test_vector = [_]f32{0.1} ** 64;
    _ = try db.addEmbedding(&test_vector);
    
    const results = try db.search(&test_vector, 5, testing.allocator);
    defer testing.allocator.free(results);
    try testing.expect(results.len > 0);
}
