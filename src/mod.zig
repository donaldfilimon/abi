//! WDBX-AI Vector Database - Unified Module Interface
//!
//! This module provides a clean, organized interface to all WDBX-AI functionality.
//! It consolidates the various WDBX implementations into a single, maintainable structure.

const std = @import("std");

// Core system modules
pub const core = @import("core/mod.zig");
pub const database = @import("database/mod.zig");
pub const simd = @import("simd/mod.zig");
pub const ai = @import("ai/mod.zig");
pub const plugins = @import("plugins/mod.zig");

// WDBX implementation (consolidated)
pub const wdbx = @import("wdbx/mod.zig");

// Utility modules
pub const utils = @import("utils.zig");
pub const performance = @import("core/performance_utils.zig");
pub const memory_tracker = @import("core/memory_tracker.zig");

// Re-export commonly used types
pub const Db = database.Db;
pub const DbError = database.DbError;
pub const Result = database.Result;
pub const WdbxHeader = database.WdbxHeader;

// Re-export SIMD operations
pub const Vector = simd.Vector;
pub const VectorOps = simd.VectorOps;
pub const MatrixOps = simd.MatrixOps;

// Re-export AI capabilities
pub const NeuralNetwork = ai.NeuralNetwork;
pub const EmbeddingGenerator = ai.EmbeddingGenerator;
pub const ModelTrainer = ai.ModelTrainer;

// Re-export core utilities
pub const Allocator = core.Allocator;
pub const ArrayList = core.ArrayList;
pub const HashMap = core.HashMap;

// Re-export WDBX utilities
pub const Command = wdbx.Command;
pub const OutputFormat = wdbx.OutputFormat;
pub const LogLevel = wdbx.LogLevel;
pub const WdbxCLI = wdbx.WdbxCLI;

/// Main application entry point
pub fn main() !void {
    try wdbx.main();
}

/// Initialize the WDBX-AI system
pub fn init(allocator: std.mem.Allocator) !void {
    try core.init(allocator);
    try database.init(allocator);
    try simd.init(allocator);
    try ai.init(allocator);
}

/// Cleanup the WDBX-AI system
pub fn deinit() void {
    ai.deinit();
    simd.deinit();
    database.deinit();
    core.deinit();
}

/// Get system information
pub fn getSystemInfo() struct {
    version: []const u8,
    features: []const []const u8,
    simd_support: struct {
        f32x4: bool,
        f32x8: bool,
        f32x16: bool,
    },
} {
    return .{
        .version = "2.0.0",
        .features = &[_][]const u8{
            "HNSW Indexing",
            "Parallel Search",
            "SIMD Optimization",
            "Neural Networks",
            "Embedding Generation",
            "Performance Monitoring",
            "Plugin System",
            "Unified CLI",
        },
        .simd_support = .{
            .f32x4 = simd.Vector.isSimdAvailable(4),
            .f32x8 = simd.Vector.isSimdAvailable(8),
            .f32x16 = simd.Vector.isSimdAvailable(16),
        },
    };
}
