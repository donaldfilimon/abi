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
// Enhanced Parity: Declaration Kind + Signature Checking
// ============================================================================

/// What kind of declaration we expect.
pub const DeclKind = enum { function, type_decl, any };

/// Rich declaration specification: name, expected kind, and constraints.
pub const DeclSpec = struct {
    name: []const u8,
    kind: DeclKind = .any,
    /// For functions: minimum explicit parameter count.
    min_params: ?usize = null,
    /// For types: sub-declarations that must exist (e.g. init, deinit).
    sub_decls: []const []const u8 = &.{},
};

/// Verify declarations with kind and signature checks.
/// Catches drift that @hasDecl alone cannot: a function renamed to a type,
/// or a type losing its init/deinit methods.
pub fn verifyDeclSpecs(comptime Module: type, comptime specs: []const DeclSpec) void {
    inline for (specs) |spec| {
        if (!@hasDecl(Module, spec.name)) {
            @compileError("Module missing declaration: '" ++ spec.name ++ "'");
        }

        const DeclType = @TypeOf(@field(Module, spec.name));
        const info = @typeInfo(DeclType);

        switch (spec.kind) {
            .function => {
                if (info != .@"fn") {
                    @compileError("Expected '" ++ spec.name ++ "' to be a function");
                }
                if (spec.min_params) |min_p| {
                    if (info.@"fn".params.len < min_p) {
                        @compileError("Function '" ++ spec.name ++ "' has fewer params than expected");
                    }
                }
            },
            .type_decl => {
                if (info != .type) {
                    @compileError("Expected '" ++ spec.name ++ "' to be a type");
                }
                const T = @field(Module, spec.name);
                inline for (spec.sub_decls) |sub| {
                    if (!@hasDecl(T, sub)) {
                        @compileError("Type '" ++ spec.name ++ "' missing sub-declaration: '" ++ sub ++ "'");
                    }
                }
            },
            .any => {},
        }
    }
}

/// Non-failing version: returns count of spec violations.
pub fn countSpecViolations(comptime Module: type, comptime specs: []const DeclSpec) usize {
    comptime var violations: usize = 0;

    inline for (specs) |spec| {
        if (!@hasDecl(Module, spec.name)) {
            violations += 1;
            continue;
        }

        const DeclType = @TypeOf(@field(Module, spec.name));
        const info = @typeInfo(DeclType);

        switch (spec.kind) {
            .function => {
                if (info != .@"fn") {
                    violations += 1;
                } else if (spec.min_params) |min_p| {
                    if (info.@"fn".params.len < min_p) {
                        violations += 1;
                    }
                }
            },
            .type_decl => {
                if (info != .type) {
                    violations += 1;
                } else {
                    const T = @field(Module, spec.name);
                    inline for (spec.sub_decls) |sub| {
                        if (!@hasDecl(T, sub)) {
                            violations += 1;
                        }
                    }
                }
            },
            .any => {},
        }
    }

    return violations;
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
// Enhanced Declaration Specs (kind + signature constraints)
// ============================================================================

/// GPU module: types must have init/deinit, functions must have correct arity.
const gpu_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "Gpu", .kind = .type_decl },
    .{ .name = "GpuConfig", .kind = .type_decl },
    .{ .name = "GpuError", .kind = .type_decl },
    .{ .name = "Backend", .kind = .type_decl },
    .{ .name = "Buffer", .kind = .type_decl },
    .{ .name = "UnifiedBuffer", .kind = .type_decl },
    .{ .name = "BufferOptions", .kind = .type_decl },
    .{ .name = "BufferFlags", .kind = .type_decl },
    .{ .name = "Device", .kind = .type_decl },
    .{ .name = "DeviceType", .kind = .type_decl },
    .{ .name = "Stream", .kind = .type_decl },
    .{ .name = "StreamOptions", .kind = .type_decl },
    .{ .name = "Event", .kind = .type_decl },
    .{ .name = "EventOptions", .kind = .type_decl },
    .{ .name = "ExecutionResult", .kind = .type_decl },
    .{ .name = "LaunchConfig", .kind = .type_decl },
    .{ .name = "HealthStatus", .kind = .type_decl },
    .{ .name = "KernelBuilder", .kind = .type_decl },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
};

/// AI module: verify types, functions, and submodule accessibility.
const ai_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "Error", .kind = .type_decl },
    .{ .name = "Agent", .kind = .type_decl },
    .{ .name = "TrainingConfig", .kind = .type_decl },
    .{ .name = "TrainingResult", .kind = .type_decl },
    .{ .name = "Tool", .kind = .type_decl },
    .{ .name = "ToolResult", .kind = .type_decl },
    .{ .name = "ToolRegistry", .kind = .type_decl },
    .{ .name = "LlmEngine", .kind = .type_decl },
    .{ .name = "LlmModel", .kind = .type_decl },
    .{ .name = "LlmConfig", .kind = .type_decl },
    .{ .name = "StreamingGenerator", .kind = .type_decl },
    .{ .name = "StreamToken", .kind = .type_decl },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    // Submodules are type declarations (they're struct namespaces)
    .{ .name = "llm", .kind = .type_decl },
    .{ .name = "embeddings", .kind = .type_decl },
    .{ .name = "agents", .kind = .type_decl },
    .{ .name = "training", .kind = .type_decl },
    .{ .name = "streaming", .kind = .type_decl },
};

/// Database module specs.
const database_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "DatabaseHandle", .kind = .type_decl },
    .{ .name = "SearchResult", .kind = .type_decl },
    .{ .name = "VectorView", .kind = .type_decl },
    .{ .name = "Stats", .kind = .type_decl },
    .{ .name = "wdbx", .kind = .type_decl },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "open", .kind = .function },
    .{ .name = "close", .kind = .function },
    .{ .name = "insert", .kind = .function },
    .{ .name = "search", .kind = .function },
};

/// Network module specs.
const network_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "Error", .kind = .type_decl },
    .{ .name = "NetworkConfig", .kind = .type_decl },
    .{ .name = "NodeInfo", .kind = .type_decl },
    .{ .name = "NodeRegistry", .kind = .type_decl },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "defaultRegistry", .kind = .function },
    .{ .name = "defaultConfig", .kind = .function },
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

test "gpu module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.gpu, &gpu_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.gpu, &gpu_specs));
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

test "ai module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.ai, &ai_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.ai, &ai_specs));
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

test "database module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.database, &database_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.database, &database_specs));
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

test "network module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.network, &network_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.network, &network_specs));
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

test "enhanced spec checker validates declaration kinds" {
    const TestModule = struct {
        pub const MyType = struct {
            pub fn init() void {}
            pub fn deinit() void {}
        };
        pub fn myFunc(a: u32, b: u32) u64 {
            return @as(u64, a) + @as(u64, b);
        }
        pub const my_const: u32 = 42;
    };

    // Correct specs should have zero violations
    const correct_specs = [_]DeclSpec{
        .{ .name = "MyType", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
        .{ .name = "myFunc", .kind = .function, .min_params = 2 },
        .{ .name = "my_const" }, // .any kind accepts anything
    };
    try std.testing.expectEqual(
        @as(usize, 0),
        comptime countSpecViolations(TestModule, &correct_specs),
    );

    // Wrong kind should be detected
    const wrong_kind = [_]DeclSpec{
        .{ .name = "MyType", .kind = .function }, // MyType is a type, not a function
    };
    try std.testing.expectEqual(
        @as(usize, 1),
        comptime countSpecViolations(TestModule, &wrong_kind),
    );

    // Missing sub-declaration should be detected
    const missing_sub = [_]DeclSpec{
        .{ .name = "MyType", .kind = .type_decl, .sub_decls = &.{ "init", "nonexistent" } },
    };
    try std.testing.expectEqual(
        @as(usize, 1),
        comptime countSpecViolations(TestModule, &missing_sub),
    );
}
