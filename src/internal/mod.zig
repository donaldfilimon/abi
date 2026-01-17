//! Internal Module
//!
//! Shared internal utilities used across the framework.
//! These are implementation details and not part of the public API.
//!
//! **Note**: This module is for internal use only. APIs may change without notice.

const std = @import("std");

// Re-export from shared/ for gradual migration
pub const logging = @import("../shared/logging/mod.zig");
pub const observability = @import("../shared/observability/mod.zig");
pub const platform = @import("../shared/platform/mod.zig");
pub const plugins = @import("../shared/plugins/mod.zig");
pub const security = @import("../shared/security/mod.zig");
pub const simd = @import("../shared/simd.zig");
pub const utils = @import("../shared/utils/mod.zig");

// Convenience re-exports for common utilities
pub const config = utils.config;
pub const memory = utils.memory;
pub const time = utils.time;
pub const retry = utils.retry;
pub const string = utils.string;
pub const net = utils.net;
pub const lifecycle = utils.lifecycle;

// SIMD convenience exports
pub const vectorAdd = simd.vectorAdd;
pub const vectorDot = simd.vectorDot;
pub const vectorL2Norm = simd.vectorL2Norm;
pub const cosineSimilarity = simd.cosineSimilarity;
pub const hasSimdSupport = simd.hasSimdSupport;

// Logging convenience exports
pub const Logger = logging.Logger;
pub const LogLevel = logging.LogLevel;
pub const log = logging.log;

// Platform detection
pub const Platform = platform.Platform;
pub const getPlatform = platform.getPlatform;
pub const isWindows = platform.isWindows;
pub const isLinux = platform.isLinux;
pub const isMacos = platform.isMacos;
