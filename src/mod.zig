//! ABI Framework - Main Module Interface
//!
//! Feature-based modular architecture providing:
//! - AI/ML functionality
//! - GPU acceleration
//! - Vector databases
//! - Web services
//! - Monitoring and observability
//! - External service connectors

const std = @import("std");

// =============================================================================
// FEATURE MODULES
// =============================================================================

/// AI/ML functionality and neural networks
pub const ai = @import("features/ai/mod.zig");

/// GPU acceleration and compute
pub const gpu = @import("features/gpu/mod.zig");

/// Vector databases and data persistence
pub const database = @import("features/database/mod.zig");

/// Web servers, HTTP clients, and services
pub const web = @import("features/web/mod.zig");

/// System monitoring and observability
pub const monitoring = @import("features/monitoring/mod.zig");

/// External service integrations
pub const connectors = @import("features/connectors/mod.zig");

// =============================================================================
// SHARED MODULES
// =============================================================================

/// Cross-cutting utilities and helpers
pub const utils = @import("shared/utils/mod.zig");

/// Core system functionality
pub const core = @import("shared/core/mod.zig");

/// Platform-specific abstractions
pub const platform = @import("shared/platform/mod.zig");

/// Logging and telemetry
pub const logging = @import("shared/logging/mod.zig");

// =============================================================================
// LEGACY COMPATIBILITY
// =============================================================================

/// SIMD operations (moved to shared)
pub const simd = @import("shared/simd.zig");

/// Main application entry point
pub const main = @import("main.zig");

/// Root configuration
pub const root = @import("root.zig");

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Initialize the entire ABI framework
pub fn init(allocator: std.mem.Allocator) !void {
    // Initialize shared modules first
    try core.lifecycle.init(allocator);
    try logging.logging.init();

    // Initialize features as needed
    try ai.init();
    try database.init();
    try web.init();
    try monitoring.init();
}

/// Shutdown the entire ABI framework
pub fn deinit() void {
    // Shutdown features in reverse order
    monitoring.deinit();
    web.deinit();
    database.deinit();
    ai.deinit();

    // Shutdown shared modules
    logging.logging.deinit();
    core.lifecycle.deinit();
}

/// Get framework version information
pub fn version() []const u8 {
    return "1.0.0-alpha";
}

test {
    // Run comprehensive framework tests
    std.testing.refAllDecls(@This());
}
