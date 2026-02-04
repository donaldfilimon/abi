//! Database Stub Module
//!
//! This module provides API-compatible no-op implementations for all public
//! database functions when the database feature is disabled at compile time.
//! All functions return `error.DatabaseDisabled` or empty/default values as
//! appropriate. Types are kept minimal - only essential ones needed for
//! compile-time checking.
//!
//! The database module encompasses:
//! - WDBX vector database with HNSW/IVF-PQ indexing
//! - Parallel search and batch operations
//! - Full-text search and hybrid queries
//! - Clustering and quantization
//! - Backup, restore, and persistence
//! - HTTP server for remote access
//!
//! To enable the real implementation, build with `-Denable-database=true`.

const std = @import("std");
const config_module = @import("../config/mod.zig");

// ============================================================================
// Error Types
// ============================================================================

pub const DatabaseFeatureError = error{DatabaseDisabled};
pub const DatabaseError = error{
    DuplicateId,
    VectorNotFound,
    InvalidDimension,
    PoolExhausted,
    PersistenceError,
    ConcurrencyError,
    DatabaseDisabled,
};

// ============================================================================
// Local Stubs Imports
// ============================================================================

pub const types = @import("stubs/types.zig");
pub const wdbx_mod = @import("stubs/wdbx.zig");
pub const parallel = @import("stubs/parallel.zig");
pub const misc = @import("stubs/misc.zig");

// ============================================================================
// Core Types Re-exports
// ============================================================================

pub const DatabaseHandle = types.DatabaseHandle;
pub const SearchResult = types.SearchResult;
pub const VectorView = types.VectorView;
pub const Stats = types.Stats;
pub const BatchItem = types.BatchItem;
pub const DiagnosticsInfo = types.DiagnosticsInfo;

// ============================================================================
// Context (Framework integration)
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.DatabaseConfig,
    handle: ?DatabaseHandle = null,

    pub fn init(_: std.mem.Allocator, _: config_module.DatabaseConfig) !*Context {
        return error.DatabaseDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn getHandle(_: *Context) !*DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn openDatabase(_: *Context, _: []const u8) !*DatabaseHandle {
        return error.DatabaseDisabled;
    }
    pub fn insertVector(_: *Context, _: u64, _: []const f32, _: ?[]const u8) !void {
        return error.DatabaseDisabled;
    }
    pub fn searchVectors(_: *Context, _: []const f32, _: usize) ![]SearchResult {
        return error.DatabaseDisabled;
    }
    pub fn getStats(_: *Context) !Stats {
        return error.DatabaseDisabled;
    }
    pub fn optimize(_: *Context) !void {
        return error.DatabaseDisabled;
    }
};

// ============================================================================
// Sub-module Stubs Re-exports
// ============================================================================

pub const wdbx = wdbx_mod.wdbx;
pub const cli = misc.cli;

// Parallel search stubs
pub const ParallelSearchConfig = parallel.ParallelSearchConfig;
pub const ParallelSearchExecutor = parallel.ParallelSearchExecutor;
pub const ParallelBeamState = parallel.ParallelBeamState;
pub const ParallelWorkQueue = parallel.ParallelWorkQueue;
pub const BatchSearchResult = parallel.BatchSearchResult;
pub const ParallelSearchStats = parallel.ParallelSearchStats;
pub const batchCosineDistances = parallel.batchCosineDistances;

// Empty namespace stubs for less commonly used features
pub const parallel_search = misc.parallel_search;
pub const database = misc.database;
pub const db_helpers = misc.db_helpers;
pub const storage = misc.storage;
pub const http = misc.http;
pub const fulltext = misc.fulltext;
pub const hybrid = misc.hybrid;
pub const filter = misc.filter;
pub const batch = misc.batch;
pub const clustering = misc.clustering;
pub const quantization = misc.quantization;
pub const formats = misc.formats;

// ============================================================================
// Module Lifecycle
// ============================================================================

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    return error.DatabaseDisabled;
}
pub fn deinit() void {
    initialized = false;
}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return initialized;
}

// ============================================================================
// Core Database Operations
// ============================================================================

pub fn open(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
pub fn connect(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
pub fn close(_: *DatabaseHandle) void {}
pub fn insert(_: *DatabaseHandle, _: u64, _: []const f32, _: ?[]const u8) !void {
    return error.DatabaseDisabled;
}
pub fn search(_: *DatabaseHandle, _: std.mem.Allocator, _: []const f32, _: usize) ![]SearchResult {
    return error.DatabaseDisabled;
}
pub fn remove(_: *DatabaseHandle, _: u64) bool {
    return false;
}
pub fn update(_: *DatabaseHandle, _: u64, _: []const f32) !bool {
    return error.DatabaseDisabled;
}
pub fn get(_: *DatabaseHandle, _: u64) ?VectorView {
    return null;
}
pub fn list(_: *DatabaseHandle, _: std.mem.Allocator, _: usize) ![]VectorView {
    return error.DatabaseDisabled;
}
pub fn stats(_: *DatabaseHandle) Stats {
    return .{};
}
pub fn diagnostics(_: *DatabaseHandle) DiagnosticsInfo {
    return .{};
}
pub fn optimize(_: *DatabaseHandle) !void {
    return error.DatabaseDisabled;
}
pub fn backup(_: *DatabaseHandle, _: []const u8) !void {
    return error.DatabaseDisabled;
}
pub fn restore(_: *DatabaseHandle, _: []const u8) !void {
    return error.DatabaseDisabled;
}
pub fn openFromFile(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
pub fn openOrCreate(_: std.mem.Allocator, _: []const u8) !DatabaseHandle {
    return error.DatabaseDisabled;
}
