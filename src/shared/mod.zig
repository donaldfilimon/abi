//! Shared Utilities Module
//!
//! Common utilities, helpers, and cross-cutting concerns used throughout the ABI framework.
//! This module consolidates logging, SIMD operations, platform utilities, and security.

const std = @import("std");

// Core shared utilities
pub const logging = @import("logging.zig");
pub const plugins = @import("plugins.zig");
pub const simd = @import("simd.zig");
pub const utils = @import("utils.zig");
pub const os = @import("os.zig");
pub const time = @import("time.zig");
pub const io = @import("io.zig");
pub const stub_common = @import("stub_common.zig");

// Security sub-module
pub const security = @import("security/mod.zig");

// Utils sub-modules (for direct access)
pub const memory = @import("utils/memory/mod.zig");
pub const crypto = @import("utils/crypto/mod.zig");
pub const encoding = @import("utils/encoding/mod.zig");
pub const fs = @import("utils/fs/mod.zig");
pub const http = @import("utils/http/mod.zig");
pub const json = @import("utils/json/mod.zig");
pub const net = @import("utils/net/mod.zig");

// Legacy compatibility
pub const legacy = @import("legacy/mod.zig");

// Re-export commonly used items
pub const log = logging.log;
pub const Logger = logging.Logger;

// SIMD re-exports for convenience
pub const vectorAdd = simd.vectorAdd;
pub const vectorDot = simd.vectorDot;
pub const vectorL2Norm = simd.vectorL2Norm;
pub const cosineSimilarity = simd.cosineSimilarity;
pub const hasSimdSupport = simd.hasSimdSupport;

// Lifecycle utilities
pub const SimpleModuleLifecycle = utils.SimpleModuleLifecycle;
pub const LifecycleError = utils.LifecycleError;

test "shared module" {
    // Basic smoke test
    try std.testing.expect(hasSimdSupport() or !hasSimdSupport());
}
