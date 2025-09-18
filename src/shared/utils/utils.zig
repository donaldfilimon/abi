//! Utilities Module - Unified Interface
//!
//! This module provides a unified interface to all utility modules:
//! - HTTP utilities (status codes, methods, headers, requests/responses)
//! - JSON parsing and serialization
//! - String manipulation and array operations
//! - Mathematical and statistical functions
//! - Time and duration utilities
//! - Random number generation
//! - File system operations
//! - Validation utilities
//! - Memory management helpers
//!
//! ## Usage
//! ```zig
//! const utils = @import("utils");
//!
//! // HTTP utilities
//! const status = utils.http.HttpStatus.ok;
//!
//! // JSON operations
//! const value = try utils.json.JsonUtils.parse(allocator, json_str);
//!
//! // String operations
//! const trimmed = utils.string.StringUtils.trim(input);
//!
//! // Math functions
//! const result = utils.math.MathUtils.clamp(f32, 5.0, 0.0, 10.0);
//! ```

const std = @import("std");

// =============================================================================
// VERSION INFORMATION
// =============================================================================

/// Project version information
pub const VERSION = .{
    .major = 1,
    .minor = 0,
    .patch = 0,
    .pre_release = "alpha",
};

/// Render version as semantic version string: "major.minor.patch[-pre]"
pub fn versionString(allocator: std.mem.Allocator) ![]u8 {
    if (VERSION.pre_release.len > 0) {
        return std.fmt.allocPrint(allocator, "{d}.{d}.{d}-{s}", .{ VERSION.major, VERSION.minor, VERSION.patch, VERSION.pre_release });
    }
    return std.fmt.allocPrint(allocator, "{d}.{d}.{d}", .{ VERSION.major, VERSION.minor, VERSION.patch });
}

// =============================================================================
// MODULE IMPORTS - MODULARIZED UTILITIES
// =============================================================================

/// HTTP utilities (status codes, methods, headers, requests/responses)
pub const http = @import("http/mod.zig");

/// JSON parsing, serialization, and manipulation utilities
pub const json = @import("json/mod.zig");

/// String manipulation, array operations, and time utilities
pub const string = @import("string/mod.zig");

/// Mathematical functions, statistics, geometry, and random numbers
pub const math = @import("math/mod.zig");

// =============================================================================
// COMMON DEFINITIONS (LEGACY COMPATIBILITY)
// =============================================================================

/// Common configuration struct (maintained for backward compatibility)
pub const Config = struct {
    name: []const u8 = "abi-ai",
    version: u32 = 1,
    debug_mode: bool = false,

    pub fn init(name: []const u8) Config {
        return .{ .name = name };
    }
};

/// Definition types used throughout the project
pub const DefinitionType = enum {
    core,
    database,
    neural,
    web,
    cli,

    pub fn toString(self: DefinitionType) []const u8 {
        return switch (self) {
            .core => "core",
            .database => "database",
            .neural => "neural",
            .web => "web",
            .cli => "cli",
        };
    }
};

// =============================================================================
// LEGACY TYPE ALIASES (FOR BACKWARD COMPATIBILITY)
// =============================================================================
// These maintain compatibility with existing code that imports directly from utils

/// Legacy HTTP types - redirect to new modules
pub const HttpStatus = http.HttpStatus;
pub const HttpMethod = http.HttpMethod;
pub const Headers = http.Headers;
pub const HttpRequest = http.HttpRequest;
pub const HttpResponse = http.HttpResponse;

/// Legacy string and array utilities - redirect to new modules
pub const StringUtils = string.StringUtils;
pub const ArrayUtils = string.ArrayUtils;
pub const TimeUtils = string.TimeUtils;

/// Legacy JSON utilities - redirect to new modules
pub const JsonUtils = json.JsonUtils;
pub const JsonValue = json.JsonValue;

/// Legacy math utilities - redirect to new modules
pub const MathUtils = math.MathUtils;

// =============================================================================
// TESTS
// =============================================================================

test "Utilities module integration" {
    const testing = std.testing;

    // Test version functionality
    const version_str = try versionString(testing.allocator);
    defer testing.allocator.free(version_str);
    try testing.expect(std.mem.indexOf(u8, version_str, "1.0.0") != null);

    // Test config
    const config = Config.init("test");
    try testing.expectEqualStrings("test", config.name);

    // Test definition types
    try testing.expectEqualStrings("core", DefinitionType.core.toString());
}

test "Legacy compatibility - HTTP types" {
    try std.testing.expectEqual(HttpStatus.ok, http.HttpStatus.ok);
    try std.testing.expectEqual(HttpMethod.GET, http.HttpMethod.GET);
}

test "Legacy compatibility - String utilities" {
    const trimmed = StringUtils.trim("  hello  ");
    try std.testing.expectEqualStrings("hello", trimmed);
}

test "Legacy compatibility - Math utilities" {
    const clamped = MathUtils.clamp(f32, 5.0, 0.0, 3.0);
    try std.testing.expectEqual(@as(f32, 3.0), clamped);
}
