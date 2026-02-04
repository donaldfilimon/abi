//! API parity tests for stub modules.
//!
//! These comptime tests verify that the public API surface of feature modules
//! remains consistent, catching API drift at compile time.
//!
//! ## How It Works
//!
//! Since Zig 0.16's module system prevents a file from belonging to multiple
//! modules, we cannot directly compare real and stub module implementations.
//! Instead, we verify that the abi module's public API includes all expected
//! declarations regardless of which implementation (real or stub) is active.
//!
//! This approach ensures:
//! 1. All documented API declarations exist
//! 2. The API is consistent whether features are enabled or disabled
//! 3. Build failures occur at compile time if API drift is detected
//!
//! ## Usage
//!
//! Run with: `zig build test --summary all`
//!
//! To verify parity when features are disabled, rebuild with:
//! `zig build test -Denable-gpu=false -Denable-ai=false ...`

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

/// Verifies that a module has all expected public declarations.
/// Fails compilation if any expected declaration is missing.
pub fn verifyDeclarations(comptime Module: type, comptime expected: []const []const u8) void {
    inline for (expected) |name| {
        if (!@hasDecl(Module, name)) {
            @compileError("Module missing expected declaration: '" ++ name ++ "'");
        }
    }
}

/// Checks if all expected declarations exist without failing compilation.
/// Returns the list of missing declarations.
pub fn getMissingDeclarations(comptime Module: type, comptime expected: []const []const u8) []const []const u8 {
    comptime var missing: []const []const u8 = &.{};

    inline for (expected) |name| {
        if (!@hasDecl(Module, name)) {
            missing = missing ++ .{name};
        }
    }

    return missing;
}

/// Checks if all expected declarations exist.
pub fn hasAllDeclarations(comptime Module: type, comptime expected: []const []const u8) bool {
    const missing = getMissingDeclarations(Module, expected);
    return missing.len == 0;
}

// ============================================================================
// Expected API Declarations
// ============================================================================
//
// These lists define the minimum required public API for each module.
// Both real and stub implementations must export these declarations.

/// GPU module required declarations
const gpu_required = [_][]const u8{
    // Core types
    "Context",
    "Gpu",
    "GpuConfig",
    "GpuError",
    "Backend",

    // Buffer types
    "Buffer",
    "UnifiedBuffer",
    "BufferOptions",
    "BufferFlags",

    // Device types
    "Device",
    "DeviceType",

    // Stream types
    "Stream",
    "StreamOptions",
    "Event",
    "EventOptions",

    // Execution types
    "ExecutionResult",
    "LaunchConfig",
    "HealthStatus",

    // DSL types
    "KernelBuilder",

    // Module functions
    "init",
    "deinit",
    "isEnabled",
    "isInitialized",
};

/// AI module required declarations
const ai_required = [_][]const u8{
    // Core types
    "Context",
    "Error",
    "Agent",

    // Training types
    "TrainingConfig",
    "TrainingResult",

    // Tool types
    "Tool",
    "ToolResult",
    "ToolRegistry",

    // LLM types
    "LlmEngine",
    "LlmModel",
    "LlmConfig",

    // Streaming types
    "StreamingGenerator",
    "StreamToken",

    // Sub-modules
    "llm",
    "embeddings",
    "agents",
    "training",
    "streaming",

    // Module functions
    "init",
    "deinit",
    "isEnabled",
    "isInitialized",
};

/// Database module required declarations
const database_required = [_][]const u8{
    // Core types
    "Context",
    "DatabaseHandle",
    "SearchResult",
    "VectorView",
    "Stats",

    // Sub-modules
    "wdbx",

    // Module functions
    "init",
    "deinit",
    "isEnabled",
    "isInitialized",
    "open",
    "close",
    "insert",
    "search",
};

/// Network module required declarations
const network_required = [_][]const u8{
    // Core types
    "Context",
    "Error",
    "NetworkConfig",
    "NodeInfo",
    "NodeRegistry",

    // Module functions
    "init",
    "deinit",
    "isEnabled",
    "isInitialized",
    "defaultRegistry",
    "defaultConfig",
};

// ============================================================================
// GPU Module Parity Tests
// ============================================================================

test "gpu module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.gpu, &gpu_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("GPU module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    // Verify at compile time
    comptime verifyDeclarations(abi.gpu, &gpu_required);
}

test "gpu module Context follows pattern" {
    const Context = abi.gpu.Context;
    try std.testing.expect(@hasDecl(Context, "init"));
    try std.testing.expect(@hasDecl(Context, "deinit"));
}

// ============================================================================
// AI Module Parity Tests
// ============================================================================

test "ai module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.ai, &ai_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("AI module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.ai, &ai_required);
}

test "ai module Context follows pattern" {
    const Context = abi.ai.Context;
    try std.testing.expect(@hasDecl(Context, "init"));
    try std.testing.expect(@hasDecl(Context, "deinit"));
}

test "ai submodules accessible" {
    // Verify key submodules are accessible
    _ = abi.ai.llm;
    _ = abi.ai.embeddings;
    _ = abi.ai.agents;
    _ = abi.ai.training;
    _ = abi.ai.streaming;
}

// ============================================================================
// Database Module Parity Tests
// ============================================================================

test "database module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.database, &database_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Database module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.database, &database_required);
}

test "database module Context follows pattern" {
    const Context = abi.database.Context;
    try std.testing.expect(@hasDecl(Context, "init"));
    try std.testing.expect(@hasDecl(Context, "deinit"));
}

// ============================================================================
// Network Module Parity Tests
// ============================================================================

test "network module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.network, &network_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Network module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.network, &network_required);
}

test "network module Context follows pattern" {
    const Context = abi.network.Context;
    try std.testing.expect(@hasDecl(Context, "init"));
    try std.testing.expect(@hasDecl(Context, "deinit"));
}

// ============================================================================
// Cross-Module Consistency Tests
// ============================================================================

test "all feature modules follow Context pattern" {
    // All feature modules should have Context with init/deinit
    const modules = .{
        abi.gpu,
        abi.ai,
        abi.database,
        abi.network,
        abi.web,
        abi.observability,
    };

    inline for (modules) |mod| {
        try std.testing.expect(@hasDecl(mod, "Context"));
        try std.testing.expect(@hasDecl(mod, "isEnabled"));

        const Context = @field(mod, "Context");
        try std.testing.expect(@hasDecl(Context, "init"));
        try std.testing.expect(@hasDecl(Context, "deinit"));
    }
}

test "all feature modules have lifecycle functions" {
    const modules = .{
        abi.gpu,
        abi.ai,
        abi.database,
        abi.network,
    };

    inline for (modules) |mod| {
        try std.testing.expect(@hasDecl(mod, "init"));
        try std.testing.expect(@hasDecl(mod, "deinit"));
        try std.testing.expect(@hasDecl(mod, "isEnabled"));
        try std.testing.expect(@hasDecl(mod, "isInitialized"));
    }
}

// ============================================================================
// Utility Tests
// ============================================================================

test "parity checker identifies missing declarations" {
    const TestModule = struct {
        pub const TypeA = u32;
        pub const TypeB = i64;
        pub fn funcA() void {}
    };

    // All present
    const all_present = [_][]const u8{ "TypeA", "TypeB", "funcA" };
    try std.testing.expect(comptime hasAllDeclarations(TestModule, &all_present));

    // Some missing
    const some_missing = [_][]const u8{ "TypeA", "TypeC", "funcB" };
    try std.testing.expect(!comptime hasAllDeclarations(TestModule, &some_missing));

    // Verify missing count
    const missing = comptime getMissingDeclarations(TestModule, &some_missing);
    try std.testing.expectEqual(@as(usize, 2), missing.len);
}
