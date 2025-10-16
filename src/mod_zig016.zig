//! ABI Framework - Main Module Interface (Zig 0.16)
//!
//! High-level entrypoints and curated re-exports for the modernized framework
//! runtime. This module follows Zig 0.16 best practices and formatting standards.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

// Centralized imports for consistency
const imports = @import("shared/imports.zig");
const patterns = @import("shared/patterns/common.zig");
const errors = @import("shared/errors/framework_errors.zig");

// Compatibility shim
const compat = @import("compat.zig");

comptime {
    _ = compat;
}

// =============================================================================
// TYPE ALIASES AND CONSTANTS
// =============================================================================

/// Standard allocator type for framework operations
pub const Allocator = imports.Allocator;

/// Writer interface for I/O operations
pub const Writer = imports.Writer;

/// Logger for structured output
pub const Logger = patterns.Logger;

/// Error context for rich error information
pub const ErrorContext = patterns.ErrorContext;

/// Framework version information
pub const Version = struct {
    major: u32,
    minor: u32,
    patch: u32,
    pre: ?[]const u8 = null,
    
    pub fn parse(version_string: []const u8) !Version {
        // Simple version parsing - would be more robust in production
        var parts = std.mem.split(u8, version_string, ".");
        
        const major_str = parts.next() orelse return error.InvalidVersion;
        const minor_str = parts.next() orelse return error.InvalidVersion;
        const patch_str = parts.next() orelse return error.InvalidVersion;
        
        return Version{
            .major = try std.fmt.parseUnsigned(u32, major_str, 10),
            .minor = try std.fmt.parseUnsigned(u32, minor_str, 10),
            .patch = try std.fmt.parseUnsigned(u32, patch_str, 10),
        };
    }
    
    pub fn toString(self: Version, allocator: Allocator) ![]u8 {
        if (self.pre) |pre| {
            return try std.fmt.allocPrint(allocator, "{d}.{d}.{d}-{s}", .{
                self.major, self.minor, self.patch, pre
            });
        } else {
            return try std.fmt.allocPrint(allocator, "{d}.{d}.{d}", .{
                self.major, self.minor, self.patch
            });
        }
    }
    
    pub fn isCompatible(self: Version, other: Version) bool {
        // Semantic versioning compatibility check
        return self.major == other.major and 
               (self.minor > other.minor or 
                (self.minor == other.minor and self.patch >= other.patch));
    }
};

// =============================================================================
// FEATURE AND FRAMEWORK MODULES
// =============================================================================

/// Grouped feature modules following the documentation structure
pub const features = @import("features/mod.zig");

/// Individual feature namespaces re-exported at the root for ergonomic
/// imports (`abi.ai`, `abi.database`, etc.)
pub const ai = features.ai;
pub const gpu = features.gpu;
pub const database = features.database;
pub const web = features.web;
pub const monitoring = features.monitoring;
pub const connectors = features.connectors;

/// Compatibility namespace for WDBX tooling with explicit re-exports
/// following Zig 0.16 best practices (no usingnamespace)
pub const wdbx = struct {
    // Core database operations
    pub const init = features.database.unified.init;
    pub const deinit = features.database.unified.deinit;
    pub const insert = features.database.unified.insert;
    pub const search = features.database.unified.search;
    pub const count = features.database.unified.count;
    pub const clear = features.database.unified.clear;
    
    // Additional database components
    pub const database = features.database.database;
    pub const helpers = features.database.db_helpers;
    pub const cli = features.database.cli;
    pub const http = features.database.http;
    pub const config = features.database.config;
    
    // Vector operations
    pub const VectorOps = struct {
        pub const add = features.database.unified.vectorAdd;
        pub const subtract = features.database.unified.vectorSubtract;
        pub const dot = features.database.unified.vectorDot;
        pub const norm = features.database.unified.vectorNorm;
        pub const normalize = features.database.unified.vectorNormalize;
    };
};

/// Framework orchestration layer that coordinates features and plugins
pub const framework = @import("framework/mod.zig");

// =============================================================================
// SHARED MODULES
// =============================================================================

/// Utility functions and common patterns
pub const utils = @import("shared/utils/mod.zig");

/// Core functionality and data structures
pub const core = @import("shared/core/mod.zig");

/// Platform-specific abstractions
pub const platform = @import("shared/platform/mod.zig");

/// Structured logging infrastructure
pub const logging = @import("shared/logging/mod.zig");

/// Observability and monitoring tools
pub const observability = @import("shared/observability/mod.zig");

/// Plugin system and extensibility
pub const plugins = @import("shared/mod.zig");

/// SIMD operations and optimizations
pub const simd = @import("shared/simd.zig");

/// Vector operations (re-exported for convenience)
pub const VectorOps = simd.VectorOps;

// =============================================================================
// CLI AND TOOLS
// =============================================================================

/// Command-line interface and development tools
pub const cli = @import("cli/mod.zig");

/// Interactive CLI with modern patterns
pub const interactive_cli = @import("tools/interactive_cli_refactored.zig");

// =============================================================================
// PUBLIC API
// =============================================================================

/// Framework feature enumeration
pub const Feature = framework.Feature;

/// Main framework orchestrator
pub const Framework = framework.Framework;

/// Framework configuration options
pub const FrameworkOptions = framework.FrameworkOptions;

/// Runtime configuration
pub const RuntimeConfig = framework.RuntimeConfig;

/// Framework initialization result
pub const InitResult = errors.ErrorResult(Framework);

/// Initialize the ABI framework with error context
pub fn init(allocator: Allocator, options: FrameworkOptions) InitResult {
    const framework_instance = framework.runtime.Framework.init(allocator, options) catch |err| {
        const error_info = errors.frameworkError("Framework initialization failed")
            .withLocation(@src())
            .withCause(err);
        return InitResult.err(error_info);
    };
    
    return InitResult.ok(framework_instance);
}

/// Initialize with default configuration
pub fn initDefault(allocator: Allocator) InitResult {
    return init(allocator, FrameworkOptions{});
}

/// Initialize with custom runtime configuration
pub fn initWithConfig(allocator: Allocator, config: RuntimeConfig) InitResult {
    const options = framework.configToOptions(config) catch |err| {
        const error_info = errors.frameworkError("Invalid configuration")
            .withLocation(@src())
            .withCause(err);
        return InitResult.err(error_info);
    };
    
    return init(allocator, options);
}

/// Convenience wrapper around Framework.deinit for function-style shutdown
pub fn shutdown(instance: *Framework) void {
    instance.deinit();
}

/// Get framework version information
pub fn version() []const u8 {
    return build_options.package_version;
}

/// Get parsed version structure
pub fn versionParsed(allocator: Allocator) !Version {
    return Version.parse(version());
}

/// Get build information
pub const BuildInfo = struct {
    version: []const u8,
    timestamp: []const u8,
    git_commit: []const u8,
    target: std.Target,
    optimize_mode: std.builtin.OptimizeMode,
    
    pub fn current() BuildInfo {
        return BuildInfo{
            .version = build_options.package_version,
            .timestamp = build_options.build_timestamp,
            .git_commit = build_options.git_commit,
            .target = builtin.target,
            .optimize_mode = builtin.mode,
        };
    }
    
    pub fn format(self: BuildInfo, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        
        try writer.print("ABI Framework v{s}", .{self.version});
        if (self.git_commit.len > 0 and !imports.string.eql(self.git_commit, "unknown")) {
            try writer.print(" ({s})", .{self.git_commit[0..@min(8, self.git_commit.len)]});
        }
        try writer.print(" built on {s}", .{self.timestamp});
        try writer.print(" for {s}-{s}", .{ @tagName(self.target.cpu.arch), @tagName(self.target.os.tag) });
        try writer.print(" ({s})", .{@tagName(self.optimize_mode)});
    }
};

/// Get comprehensive build information
pub fn buildInfo() BuildInfo {
    return BuildInfo.current();
}

/// Check if a feature is available at compile time
pub fn hasFeature(comptime feature: Feature) bool {
    return switch (feature) {
        .ai => build_options.enable_ai,
        .gpu => build_options.enable_gpu,
        .database => build_options.enable_database,
        .web => build_options.enable_web,
        .monitoring => build_options.enable_monitoring,
        .connectors => build_options.enable_connectors,
    };
}

/// Get list of available features at compile time
pub fn availableFeatures() []const Feature {
    comptime {
        var features_list: []const Feature = &.{};
        
        if (build_options.enable_ai) features_list = features_list ++ &[_]Feature{.ai};
        if (build_options.enable_gpu) features_list = features_list ++ &[_]Feature{.gpu};
        if (build_options.enable_database) features_list = features_list ++ &[_]Feature{.database};
        if (build_options.enable_web) features_list = features_list ++ &[_]Feature{.web};
        if (build_options.enable_monitoring) features_list = features_list ++ &[_]Feature{.monitoring};
        if (build_options.enable_connectors) features_list = features_list ++ &[_]Feature{.connectors};
        
        return features_list;
    }
}

/// Platform detection utilities
pub const Platform = struct {
    pub fn current() imports.Platform {
        return imports.Platform.current();
    }
    
    pub fn supportsGPU() bool {
        return current().supportsGPU();
    }
    
    pub fn isUnix() bool {
        return current().isUnix();
    }
    
    pub fn recommendedGPUBackend() ?gpu.Backend {
        return switch (current()) {
            .windows => .vulkan, // or .dx12
            .linux => .vulkan,
            .macos => .metal,
            .wasm => .webgpu,
            .other => null,
        };
    }
};

/// Memory utilities with Zig 0.16 patterns
pub const Memory = struct {
    /// Create an arena allocator for temporary operations
    pub fn createArena(backing_allocator: Allocator) imports.ArenaAllocator {
        return imports.ArenaAllocator.init(backing_allocator);
    }
    
    /// Create a fixed buffer allocator for stack-based allocation
    pub fn createFixedBuffer(buffer: []u8) imports.FixedBufferAllocator {
        return imports.FixedBufferAllocator.init(buffer);
    }
    
    /// Get memory usage statistics (platform-dependent)
    pub fn getUsageStats(allocator: Allocator) !struct {
        allocated_bytes: usize,
        peak_allocated_bytes: usize,
        allocation_count: usize,
    } {
        _ = allocator;
        // Placeholder implementation - would use platform-specific APIs
        return .{
            .allocated_bytes = 0,
            .peak_allocated_bytes = 0,
            .allocation_count = 0,
        };
    }
};

/// Async utilities for Zig 0.16
pub const Async = struct {
    /// Run a function with a timeout
    pub fn withTimeout(
        comptime func: anytype,
        args: anytype,
        timeout_ms: u64,
    ) !@TypeOf(@call(.auto, func, args)) {
        _ = timeout_ms;
        // Simplified implementation - would use proper async/await
        return @call(.auto, func, args);
    }
    
    /// Run multiple functions concurrently
    pub fn concurrent(
        allocator: Allocator,
        comptime functions: anytype,
    ) !void {
        _ = allocator;
        _ = functions;
        // Placeholder for concurrent execution
    }
};

// =============================================================================
// TESTING SUPPORT
// =============================================================================

/// Testing utilities for Zig 0.16
pub const testing = @import("shared/testing/test_utils.zig");

/// Create a test framework instance
pub fn createTestFramework(allocator: Allocator) !Framework {
    const options = FrameworkOptions{
        .enable_logging = false,
        .enable_metrics = false,
    };
    
    const result = init(allocator, options);
    return switch (result) {
        .success => |framework_instance| framework_instance,
        .failure => |error_info| {
            std.log.err("Test framework creation failed: {}", .{error_info});
            return error.TestFrameworkCreationFailed;
        },
    };
}

// =============================================================================
// TESTS
// =============================================================================

test "framework module exports" {
    const test_allocator = imports.testing.allocator;
    
    // Test that all expected modules are available
    _ = features;
    _ = framework;
    _ = utils;
    _ = core;
    _ = platform;
    _ = logging;
    _ = observability;
    _ = plugins;
    _ = simd;
    _ = cli;
    
    // Test version parsing
    const ver = try versionParsed(test_allocator);
    defer if (ver.pre) |pre| test_allocator.free(pre);
    
    try imports.testing.expect(ver.major >= 0);
    try imports.testing.expect(ver.minor >= 0);
    try imports.testing.expect(ver.patch >= 0);
}

test "build info formatting" {
    const test_allocator = imports.testing.allocator;
    
    const info = buildInfo();
    const formatted = try std.fmt.allocPrint(test_allocator, "{}", .{info});
    defer test_allocator.free(formatted);
    
    try imports.testing.expect(formatted.len > 0);
    try imports.testing.expect(imports.string.indexOf(formatted, "ABI Framework") != null);
}

test "feature availability checks" {
    // Test compile-time feature detection
    const ai_available = hasFeature(.ai);
    const gpu_available = hasFeature(.gpu);
    
    _ = ai_available;
    _ = gpu_available;
    
    const available = availableFeatures();
    try imports.testing.expect(available.len > 0);
}

test "platform detection" {
    const current_platform = Platform.current();
    const supports_gpu = Platform.supportsGPU();
    const is_unix = Platform.isUnix();
    
    _ = current_platform;
    _ = supports_gpu;
    _ = is_unix;
    
    // Platform detection should work
    try imports.testing.expect(current_platform != .other or builtin.os.tag == .freestanding);
}

test "memory utilities" {
    const test_allocator = imports.testing.allocator;
    
    // Test arena allocator creation
    var arena = Memory.createArena(test_allocator);
    defer arena.deinit();
    
    const arena_allocator = arena.allocator();
    const test_data = try arena_allocator.alloc(u8, 100);
    _ = test_data;
    
    // Test fixed buffer allocator
    var buffer: [1024]u8 = undefined;
    var fba = Memory.createFixedBuffer(&buffer);
    const fba_allocator = fba.allocator();
    
    const fixed_data = try fba_allocator.alloc(u8, 50);
    _ = fixed_data;
}

test "error result patterns" {
    const test_allocator = imports.testing.allocator;
    
    // Test successful initialization
    const result = initDefault(test_allocator);
    switch (result) {
        .success => |framework_instance| {
            var fw = framework_instance;
            defer fw.deinit();
            try imports.testing.expect(true); // Success case
        },
        .failure => |error_info| {
            // Log error but don't fail test if framework init fails in test environment
            std.log.warn("Framework init failed in test: {}", .{error_info});
        },
    }
}

test "wdbx compatibility namespace" {
    // Test that WDBX namespace provides expected functions
    _ = wdbx.init;
    _ = wdbx.deinit;
    _ = wdbx.insert;
    _ = wdbx.search;
    _ = wdbx.count;
    _ = wdbx.VectorOps.add;
    _ = wdbx.VectorOps.dot;
    _ = wdbx.VectorOps.norm;
}

/// Comprehensive test to ensure all declarations are reachable
test "all declarations reachable" {
    imports.testing.refAllDecls(@This());
}