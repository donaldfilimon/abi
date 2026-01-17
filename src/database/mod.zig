//! Database Module
//!
//! Vector database (WDBX) with HNSW and IVF-PQ indexing for similarity search.
//!
//! ## Features
//! - HNSW index for approximate nearest neighbor search
//! - IVF-PQ for large-scale vector storage
//! - Hybrid search combining vector and keyword search
//! - Write-ahead logging for durability

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

// Re-export from features/database
const features_database = @import("../features/database/mod.zig");

pub const Database = features_database.database;
pub const helpers = features_database.db_helpers;
pub const cli = features_database.cli;
pub const http = features_database.http;
pub const wdbx = features_database.wdbx;

pub const Error = error{
    DatabaseDisabled,
    ConnectionFailed,
    QueryFailed,
    IndexError,
    StorageError,
};

/// Database context for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.DatabaseConfig,
    db: ?*anyopaque = null, // Opaque database handle

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.DatabaseConfig) !*Context {
        if (!isEnabled()) return error.DatabaseDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        // Close database if open
        if (self.db != null) {
            // wdbx.closeDatabase would be called here
        }
        self.allocator.destroy(self);
    }

    /// Open or create the database.
    pub fn open(self: *Context) !void {
        if (self.db != null) return;
        // Database connection would be established here
        _ = self.config.path;
    }

    /// Insert a vector.
    pub fn insertVector(self: *Context, id: []const u8, vector: []const f32, metadata: ?[]const u8) !void {
        try self.open();
        _ = id;
        _ = vector;
        _ = metadata;
    }

    /// Search for similar vectors.
    pub fn searchVectors(self: *Context, query: []const f32, k: usize) ![]ContextSearchResult {
        try self.open();
        _ = query;
        _ = k;
        return &.{};
    }

    /// Delete a vector.
    pub fn deleteVector(self: *Context, id: []const u8) !void {
        try self.open();
        _ = id;
    }

    pub const ContextSearchResult = struct {
        id: []const u8,
        score: f32,
        metadata: ?[]const u8,
    };
};

pub fn isEnabled() bool {
    return build_options.enable_database;
}

pub fn isInitialized() bool {
    return features_database.isInitialized();
}

pub fn init(allocator: std.mem.Allocator) Error!void {
    if (!isEnabled()) return error.DatabaseDisabled;
    features_database.init(allocator) catch return error.DatabaseDisabled;
}

pub fn deinit() void {
    features_database.deinit();
}

// Re-export WDBX functions for compatibility
pub const createDatabase = if (build_options.enable_database) features_database.wdbx.createDatabase else stubCreateDatabase;
pub const connectDatabase = if (build_options.enable_database) features_database.wdbx.connectDatabase else stubConnectDatabase;
pub const closeDatabase = if (build_options.enable_database) features_database.wdbx.closeDatabase else stubCloseDatabase;
pub const insertVectorFn = if (build_options.enable_database) features_database.wdbx.insertVector else stubInsertVector;
pub const searchVectorsFn = if (build_options.enable_database) features_database.wdbx.searchVectors else stubSearchVectors;
pub const deleteVectorFn = if (build_options.enable_database) features_database.wdbx.deleteVector else stubDeleteVector;
pub const getStats = if (build_options.enable_database) features_database.wdbx.getStats else stubGetStats;
pub const optimize = if (build_options.enable_database) features_database.wdbx.optimize else stubOptimize;

fn stubCreateDatabase(_: anytype) Error!void {
    return error.DatabaseDisabled;
}
fn stubConnectDatabase(_: anytype) Error!void {
    return error.DatabaseDisabled;
}
fn stubCloseDatabase(_: anytype) void {}
fn stubInsertVector(_: anytype, _: anytype, _: anytype, _: anytype) Error!void {
    return error.DatabaseDisabled;
}
fn stubSearchVectors(_: anytype, _: anytype, _: anytype) Error!void {
    return error.DatabaseDisabled;
}
fn stubDeleteVector(_: anytype, _: anytype) Error!void {
    return error.DatabaseDisabled;
}
fn stubGetStats(_: anytype) Error!void {
    return error.DatabaseDisabled;
}
fn stubOptimize(_: anytype) Error!void {
    return error.DatabaseDisabled;
}

// Additional re-exports for backward compatibility
pub const DatabaseHandle = features_database.DatabaseHandle;
pub const openOrCreate = features_database.openOrCreate;
pub const insert = features_database.insert;
pub const search = features_database.search;
pub const SearchResult = features_database.SearchResult;
pub const close = features_database.close;
pub const stats = features_database.stats;
pub const Stats = features_database.Stats;
