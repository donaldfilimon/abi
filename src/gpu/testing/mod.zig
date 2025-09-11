//! GPU Testing Module
//!
//! This module provides comprehensive testing frameworks for cross-platform
//! validation and performance analysis.

pub const cross_platform_tests = @import("cross_platform_tests.zig");

// Re-export key types and functions
pub const CrossPlatformTestSuite = cross_platform_tests.CrossPlatformTestSuite;
