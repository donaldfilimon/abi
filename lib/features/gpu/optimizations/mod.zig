//! GPU Optimizations Module
//!
//! This module provides platform-specific optimizations and backend detection
//! for enhanced GPU performance across different platforms and architectures.

pub const platform_optimizations = @import("platform_optimizations.zig");
pub const backend_detection = @import("backend_detection.zig");

// Re-export key types and functions
pub const PlatformOptimizations = platform_optimizations.PlatformOptimizations;
pub const PlatformConfig = platform_optimizations.PlatformConfig;
pub const PlatformMetrics = platform_optimizations.PlatformMetrics;
pub const PlatformUtils = platform_optimizations.PlatformUtils;

pub const BackendDetector = backend_detection.BackendDetector;
