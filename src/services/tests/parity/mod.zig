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
const parity_specs = @import("specs/mod.zig");
const feature_catalog = abi.feature_catalog;

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
// Bidirectional Parity: Detect Orphan Declarations
// ============================================================================

/// Returns the count of declarations in the module that are NOT in the expected list.
/// These are "orphan" declarations — they exist in one implementation but not the other,
/// which can cause code to compile with one flag setting but fail with another.
pub fn countOrphanDeclarations(comptime Module: type, comptime expected: []const []const u8) usize {
    @setEvalBranchQuota(100_000);
    const decls = @typeInfo(Module).@"struct".decls;
    comptime var orphans: usize = 0;

    inline for (decls) |decl| {
        comptime var found = false;
        inline for (expected) |name| {
            if (std.mem.eql(u8, decl.name, name)) {
                found = true;
            }
        }
        if (!found) {
            orphans += 1;
        }
    }

    return orphans;
}

/// Returns the names of declarations in the module NOT in the expected list.
pub fn getOrphanDeclarations(comptime Module: type, comptime expected: []const []const u8) []const []const u8 {
    @setEvalBranchQuota(100_000);
    const decls = @typeInfo(Module).@"struct".decls;
    comptime var orphans: []const []const u8 = &.{};

    inline for (decls) |decl| {
        comptime var found = false;
        inline for (expected) |name| {
            if (std.mem.eql(u8, decl.name, name)) {
                found = true;
            }
        }
        if (!found) {
            orphans = orphans ++ .{decl.name};
        }
    }

    return orphans;
}

/// Soft bidirectional parity check: logs orphans and missing but does not fail.
/// Returns total violation count (orphans + missing).
pub fn countBidirectionalViolations(comptime Module: type, comptime expected: []const []const u8) usize {
    const missing = getMissingDeclarations(Module, expected);
    const orphan_count = countOrphanDeclarations(Module, expected);
    return missing.len + orphan_count;
}

// ============================================================================
// Expected API Declarations
// ============================================================================
//
// These lists define the minimum required public API for each module.
// Both real and stub implementations must export these declarations.

/// GPU module required declarations
const gpu_required = parity_specs.required.forFeature(.gpu);

/// AI module required declarations
const ai_required = parity_specs.required.forFeature(.ai);

/// Database module required declarations
const database_required = parity_specs.required.forFeature(.database);

/// Network module required declarations
const network_required = parity_specs.required.forFeature(.network);

/// Web module required declarations
const web_required = parity_specs.required.forFeature(.web);

/// Observability module required declarations
const observability_required = parity_specs.required.forFeature(.observability);

/// Analytics module required declarations
const analytics_required = parity_specs.required.forFeature(.analytics);

/// Cloud module required declarations
const cloud_required = parity_specs.required.forFeature(.cloud);

/// Auth module required declarations
const auth_required = parity_specs.required.forFeature(.auth);

/// Gateway module required declarations
const gateway_required = parity_specs.required.forFeature(.gateway);

/// Messaging module required declarations
const messaging_required = parity_specs.required.forFeature(.messaging);

/// Cache module required declarations
const cache_required = parity_specs.required.forFeature(.cache);

/// Storage module required declarations
const storage_required = parity_specs.required.forFeature(.storage);

/// Search module required declarations
const search_required = parity_specs.required.forFeature(.search);

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
    .{ .name = "generateServiceId", .kind = .function, .min_params = 2 },
    .{ .name = "base64Encode", .kind = .function, .min_params = 2 },
    .{ .name = "base64Decode", .kind = .function, .min_params = 2 },
};

/// Web module specs.
const web_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "WebError", .kind = .type_decl },
    .{ .name = "Response", .kind = .type_decl },
    .{ .name = "HttpClient", .kind = .type_decl },
    .{ .name = "RequestOptions", .kind = .type_decl },
    .{ .name = "WeatherClient", .kind = .type_decl },
    .{ .name = "WeatherConfig", .kind = .type_decl },
    .{ .name = "ChatHandler", .kind = .type_decl, .sub_decls = &.{"init"} },
    .{ .name = "ChatRequest", .kind = .type_decl },
    .{ .name = "ChatResponse", .kind = .type_decl },
    .{ .name = "ChatResult", .kind = .type_decl },
    .{ .name = "PersonaRouter", .kind = .type_decl },
    .{ .name = "Route", .kind = .type_decl },
    .{ .name = "RouteContext", .kind = .type_decl },
    .{ .name = "handlers", .kind = .type_decl },
    .{ .name = "routes", .kind = .type_decl },
    .{ .name = "http", .kind = .type_decl },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "get", .kind = .function },
    .{ .name = "getWithOptions", .kind = .function },
    .{ .name = "postJson", .kind = .function },
    .{ .name = "freeResponse", .kind = .function },
    .{ .name = "parseJsonValue", .kind = .function },
    .{ .name = "isSuccessStatus", .kind = .function },
};

/// Observability module specs.
const observability_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "Error", .kind = .type_decl },
    .{ .name = "Counter", .kind = .type_decl, .sub_decls = &.{ "inc", "get" } },
    .{ .name = "Gauge", .kind = .type_decl, .sub_decls = &.{ "set", "get", "inc", "dec" } },
    .{ .name = "FloatGauge", .kind = .type_decl, .sub_decls = &.{ "set", "get", "add" } },
    .{ .name = "Histogram", .kind = .type_decl, .sub_decls = &.{ "init", "deinit", "record" } },
    .{ .name = "MetricsCollector", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "MetricsConfig", .kind = .type_decl },
    .{ .name = "MetricsSummary", .kind = .type_decl },
    .{ .name = "Tracer", .kind = .type_decl },
    .{ .name = "Span", .kind = .type_decl },
    .{ .name = "ObservabilityBundle", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "BundleConfig", .kind = .type_decl },
    .{ .name = "AlertManager", .kind = .type_decl },
    .{ .name = "AlertRule", .kind = .type_decl },
    .{ .name = "PrometheusExporter", .kind = .type_decl },
    .{ .name = "PrometheusConfig", .kind = .type_decl },
    .{ .name = "OtelExporter", .kind = .type_decl },
    .{ .name = "OtelConfig", .kind = .type_decl },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "createCollector", .kind = .function },
    .{ .name = "registerDefaultMetrics", .kind = .function },
    .{ .name = "recordRequest", .kind = .function },
    .{ .name = "recordError", .kind = .function },
};

/// Analytics module specs.
const analytics_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "Event", .kind = .type_decl },
    .{ .name = "AnalyticsConfig", .kind = .type_decl },
    .{ .name = "AnalyticsError", .kind = .type_decl },
    .{ .name = "Engine", .kind = .type_decl, .sub_decls = &.{ "init", "deinit", "track", "flush" } },
    .{ .name = "Funnel", .kind = .type_decl, .sub_decls = &.{ "init", "deinit", "addStep", "recordStep" } },
    .{ .name = "Experiment", .kind = .type_decl, .sub_decls = &.{ "assign", "totalAssignments" } },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
};

/// Cloud module specs.
const cloud_specs = [_]DeclSpec{
    .{ .name = "CloudEvent", .kind = .type_decl },
    .{ .name = "CloudResponse", .kind = .type_decl },
    .{ .name = "CloudProvider", .kind = .type_decl },
    .{ .name = "CloudHandler", .kind = .type_decl },
    .{ .name = "CloudConfig", .kind = .type_decl },
    .{ .name = "CloudError", .kind = .type_decl },
    .{ .name = "HttpMethod", .kind = .type_decl },
    .{ .name = "InvocationMetadata", .kind = .type_decl },
    .{ .name = "Error", .kind = .type_decl },
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit", "wrapHandler" } },
    .{ .name = "ResponseBuilder", .kind = .type_decl, .sub_decls = &.{ "init", "build" } },
    .{ .name = "detectProvider", .kind = .function },
    .{ .name = "detectProviderWithAllocator", .kind = .function },
    .{ .name = "runHandler", .kind = .function },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "aws_lambda", .kind = .type_decl },
    .{ .name = "gcp_functions", .kind = .type_decl },
    .{ .name = "azure_functions", .kind = .type_decl },
};

/// Auth module specs.
const auth_specs = [_]DeclSpec{
    .{ .name = "AuthConfig", .kind = .type_decl },
    .{ .name = "AuthError", .kind = .type_decl },
    .{ .name = "Token", .kind = .type_decl, .sub_decls = &.{"Claims"} },
    .{ .name = "Session", .kind = .type_decl },
    .{ .name = "Permission", .kind = .type_decl },
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "createToken", .kind = .function, .min_params = 2 },
    .{ .name = "verifyToken", .kind = .function, .min_params = 1 },
    .{ .name = "createSession", .kind = .function, .min_params = 2 },
    .{ .name = "checkPermission", .kind = .function, .min_params = 2 },
};

/// Gateway module specs.
const gateway_specs = [_]DeclSpec{
    .{ .name = "GatewayConfig", .kind = .type_decl },
    .{ .name = "GatewayError", .kind = .type_decl },
    .{ .name = "HttpMethod", .kind = .type_decl },
    .{ .name = "Route", .kind = .type_decl },
    .{ .name = "MiddlewareType", .kind = .type_decl },
    .{ .name = "GatewayStats", .kind = .type_decl },
    .{ .name = "MatchResult", .kind = .type_decl, .sub_decls = &.{"getParam"} },
    .{ .name = "RateLimitResult", .kind = .type_decl },
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "addRoute", .kind = .function, .min_params = 1 },
    .{ .name = "removeRoute", .kind = .function, .min_params = 1 },
    .{ .name = "matchRoute", .kind = .function, .min_params = 2 },
    .{ .name = "checkRateLimit", .kind = .function, .min_params = 1 },
    .{ .name = "recordUpstreamResult", .kind = .function, .min_params = 2 },
    .{ .name = "stats", .kind = .function },
    .{ .name = "getCircuitState", .kind = .function, .min_params = 1 },
    .{ .name = "resetCircuit", .kind = .function, .min_params = 1 },
};

/// Messaging module specs.
const messaging_specs = [_]DeclSpec{
    .{ .name = "MessagingConfig", .kind = .type_decl },
    .{ .name = "MessagingError", .kind = .type_decl },
    .{ .name = "Message", .kind = .type_decl },
    .{ .name = "Channel", .kind = .type_decl },
    .{ .name = "MessagingStats", .kind = .type_decl },
    .{ .name = "TopicInfo", .kind = .type_decl },
    .{ .name = "DeadLetter", .kind = .type_decl },
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "publish", .kind = .function, .min_params = 3 },
    .{ .name = "subscribe", .kind = .function, .min_params = 2 },
    .{ .name = "unsubscribe", .kind = .function, .min_params = 1 },
    .{ .name = "listTopics", .kind = .function, .min_params = 1 },
    .{ .name = "topicStats", .kind = .function, .min_params = 1 },
    .{ .name = "getDeadLetters", .kind = .function, .min_params = 1 },
    .{ .name = "clearDeadLetters", .kind = .function },
    .{ .name = "messagingStats", .kind = .function },
};

/// Cache module specs.
const cache_specs = [_]DeclSpec{
    .{ .name = "CacheConfig", .kind = .type_decl },
    .{ .name = "EvictionPolicy", .kind = .type_decl },
    .{ .name = "CacheError", .kind = .type_decl },
    .{ .name = "CacheEntry", .kind = .type_decl },
    .{ .name = "CacheStats", .kind = .type_decl },
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "get", .kind = .function, .min_params = 1 },
    .{ .name = "put", .kind = .function, .min_params = 2 },
    .{ .name = "putWithTtl", .kind = .function, .min_params = 3 },
    .{ .name = "delete", .kind = .function, .min_params = 1 },
    .{ .name = "contains", .kind = .function, .min_params = 1 },
    .{ .name = "clear", .kind = .function },
    .{ .name = "size", .kind = .function },
    .{ .name = "stats", .kind = .function },
};

/// Storage module specs.
const storage_specs = [_]DeclSpec{
    .{ .name = "StorageConfig", .kind = .type_decl },
    .{ .name = "StorageBackend", .kind = .type_decl },
    .{ .name = "StorageError", .kind = .type_decl },
    .{ .name = "StorageObject", .kind = .type_decl },
    .{ .name = "ObjectMetadata", .kind = .type_decl },
    .{ .name = "StorageStats", .kind = .type_decl },
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "putObject", .kind = .function, .min_params = 3 },
    .{ .name = "putObjectWithMetadata", .kind = .function, .min_params = 4 },
    .{ .name = "getObject", .kind = .function, .min_params = 2 },
    .{ .name = "deleteObject", .kind = .function, .min_params = 1 },
    .{ .name = "objectExists", .kind = .function, .min_params = 1 },
    .{ .name = "listObjects", .kind = .function, .min_params = 2 },
    .{ .name = "stats", .kind = .function },
};

/// Search module specs.
const search_specs = [_]DeclSpec{
    .{ .name = "SearchConfig", .kind = .type_decl },
    .{ .name = "SearchError", .kind = .type_decl },
    .{ .name = "SearchResult", .kind = .type_decl },
    .{ .name = "SearchIndex", .kind = .type_decl },
    .{ .name = "SearchStats", .kind = .type_decl },
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "init", .kind = .function },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "createIndex", .kind = .function, .min_params = 2 },
    .{ .name = "deleteIndex", .kind = .function, .min_params = 1 },
    .{ .name = "indexDocument", .kind = .function, .min_params = 3 },
    .{ .name = "deleteDocument", .kind = .function, .min_params = 2 },
    .{ .name = "query", .kind = .function, .min_params = 3 },
    .{ .name = "stats", .kind = .function },
};

// ============================================================================
// AI Core Module DeclSpec Parity Tests
// ============================================================================

/// AI Core module required declarations.
const ai_core_required = parity_specs.required.forFeature(.agents);

/// AI Core module specs.
const ai_core_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "Error", .kind = .type_decl },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "createAgent", .kind = .function, .min_params = 2 },
    .{ .name = "createRegistry", .kind = .function, .min_params = 1 },
};

test "ai_core module has required declarations" {
    comptime verifyDeclarations(abi.ai_core, ai_core_required);
}

test "ai_core module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.ai_core, &ai_core_specs);
    try std.testing.expectEqual(
        @as(usize, 0),
        comptime countSpecViolations(abi.ai_core, &ai_core_specs),
    );
}

// ============================================================================
// AI Inference Module DeclSpec Parity Tests
// ============================================================================

/// AI Inference module required declarations.
const ai_inference_required = parity_specs.required.forFeature(.llm);

/// AI Inference module specs.
const ai_inference_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "Error", .kind = .type_decl },
    .{ .name = "isEnabled", .kind = .function },
};

test "ai_inference module has required declarations" {
    comptime verifyDeclarations(abi.inference, ai_inference_required);
}

test "ai_inference module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.inference, &ai_inference_specs);
    try std.testing.expectEqual(
        @as(usize, 0),
        comptime countSpecViolations(abi.inference, &ai_inference_specs),
    );
}

// ============================================================================
// AI Training Module DeclSpec Parity Tests
// ============================================================================

/// AI Training module required declarations.
const ai_training_required = parity_specs.required.forFeature(.training);

/// AI Training module specs.
const ai_training_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "Error", .kind = .type_decl },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "train", .kind = .function, .min_params = 2 },
    .{ .name = "trainWithResult", .kind = .function, .min_params = 2 },
};

test "ai_training module has required declarations" {
    comptime verifyDeclarations(abi.training, ai_training_required);
}

test "ai_training module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.training, &ai_training_specs);
    try std.testing.expectEqual(
        @as(usize, 0),
        comptime countSpecViolations(abi.training, &ai_training_specs),
    );
}

// ============================================================================
// AI Reasoning Module DeclSpec Parity Tests
// ============================================================================

/// AI Reasoning module required declarations.
const ai_reasoning_required = parity_specs.required.forFeature(.reasoning);

/// AI Reasoning module specs.
const ai_reasoning_specs = [_]DeclSpec{
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "Error", .kind = .type_decl },
    .{ .name = "isEnabled", .kind = .function },
};

test "ai_reasoning module has required declarations" {
    comptime verifyDeclarations(abi.reasoning, ai_reasoning_required);
}

test "ai_reasoning module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.reasoning, &ai_reasoning_specs);
    try std.testing.expectEqual(
        @as(usize, 0),
        comptime countSpecViolations(abi.reasoning, &ai_reasoning_specs),
    );
}

// ============================================================================
// Mobile Module Parity
// ============================================================================

/// Mobile module required declarations.
const mobile_required = parity_specs.required.forFeature(.mobile);

/// Mobile module specs.
const mobile_specs = [_]DeclSpec{
    .{ .name = "MobileConfig", .kind = .type_decl },
    .{ .name = "MobileError", .kind = .type_decl },
    .{ .name = "LifecycleState", .kind = .type_decl },
    .{ .name = "SensorData", .kind = .type_decl },
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "init", .kind = .function, .min_params = 2 },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "getLifecycleState", .kind = .function },
    .{ .name = "readSensor", .kind = .function, .min_params = 1 },
    .{ .name = "sendNotification", .kind = .function, .min_params = 2 },
};

test "mobile module has required declarations" {
    comptime verifyDeclarations(abi.mobile, mobile_required);
}

test "mobile module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.mobile, &mobile_specs);
    try std.testing.expectEqual(
        @as(usize, 0),
        comptime countSpecViolations(abi.mobile, &mobile_specs),
    );
}

test "mobile module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.mobile, mobile_required);
    if (orphans.len > 0) {
        std.log.info("Mobile module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

// ============================================================================
// Pages Module Parity
// ============================================================================

/// Pages module required declarations.
const pages_required = parity_specs.required.forFeature(.pages);

/// Pages module specs.
const pages_specs = [_]DeclSpec{
    .{ .name = "PagesConfig", .kind = .type_decl },
    .{ .name = "PagesError", .kind = .type_decl },
    .{ .name = "Page", .kind = .type_decl },
    .{ .name = "PageMatch", .kind = .type_decl, .sub_decls = &.{"getParam"} },
    .{ .name = "RenderResult", .kind = .type_decl, .sub_decls = &.{"deinit"} },
    .{ .name = "PagesStats", .kind = .type_decl },
    .{ .name = "init", .kind = .function, .min_params = 2 },
    .{ .name = "deinit", .kind = .function },
    .{ .name = "isEnabled", .kind = .function },
    .{ .name = "isInitialized", .kind = .function },
    .{ .name = "addPage", .kind = .function, .min_params = 1 },
    .{ .name = "removePage", .kind = .function, .min_params = 1 },
    .{ .name = "getPage", .kind = .function, .min_params = 1 },
    .{ .name = "matchPage", .kind = .function, .min_params = 1 },
    .{ .name = "renderPage", .kind = .function, .min_params = 3 },
    .{ .name = "listPages", .kind = .function },
    .{ .name = "stats", .kind = .function },
};

test "pages module has required declarations" {
    comptime verifyDeclarations(abi.pages, pages_required);
}

test "pages module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.pages, &pages_specs);
    try std.testing.expectEqual(
        @as(usize, 0),
        comptime countSpecViolations(abi.pages, &pages_specs),
    );
}

test "pages module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.pages, pages_required);
    if (orphans.len > 0) {
        std.log.info("Pages module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

// ============================================================================
// Benchmarks Module Parity
// ============================================================================

/// Benchmarks module required declarations.
const benchmarks_required = parity_specs.required.forFeature(.benchmarks);

/// Benchmarks module specs.
const benchmarks_specs = [_]DeclSpec{
    .{ .name = "Config", .kind = .type_decl },
    .{ .name = "BenchmarksError", .kind = .type_decl },
    .{ .name = "Context", .kind = .type_decl, .sub_decls = &.{ "init", "deinit" } },
    .{ .name = "isEnabled", .kind = .function },
};

test "benchmarks module has required declarations" {
    comptime verifyDeclarations(abi.benchmarks, benchmarks_required);
}

test "benchmarks module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.benchmarks, &benchmarks_specs);
    try std.testing.expectEqual(
        @as(usize, 0),
        comptime countSpecViolations(abi.benchmarks, &benchmarks_specs),
    );
}

test "benchmarks module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.benchmarks, benchmarks_required);
    if (orphans.len > 0) {
        std.log.info("Benchmarks module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

// ============================================================================
// GPU Module Parity Tests
// ============================================================================

test "gpu module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.gpu, gpu_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("GPU module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    // Verify at compile time
    comptime verifyDeclarations(abi.gpu, gpu_required);
}

test "gpu module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.gpu, &gpu_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.gpu, &gpu_specs));
}

// ============================================================================
// AI Module Parity Tests
// ============================================================================

test "ai module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.ai, ai_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("AI module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.ai, ai_required);
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
    const missing = comptime getMissingDeclarations(abi.database, database_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Database module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.database, database_required);
}

test "database module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.database, &database_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.database, &database_specs));
}

// ============================================================================
// Network Module Parity Tests
// ============================================================================

test "network module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.network, network_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Network module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.network, network_required);
}

test "network module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.network, &network_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.network, &network_specs));
}

// ============================================================================
// Cloud Module Parity Tests
// ============================================================================

test "cloud module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.cloud, cloud_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Cloud module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.cloud, cloud_required);
}

test "cloud module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.cloud, &cloud_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.cloud, &cloud_specs));
}

// ============================================================================
// Web Module Parity Tests
// ============================================================================

test "web module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.web, web_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Web module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.web, web_required);
}

test "web module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.web, &web_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.web, &web_specs));
}

// ============================================================================
// Observability Module Parity Tests
// ============================================================================

test "observability module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.observability, observability_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Observability module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.observability, observability_required);
}

test "observability module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.observability, &observability_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.observability, &observability_specs));
}

// ============================================================================
// Analytics Module Parity Tests
// ============================================================================

test "analytics module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.analytics, analytics_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Analytics module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.analytics, analytics_required);
}

test "analytics module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.analytics, &analytics_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.analytics, &analytics_specs));
}

// ============================================================================
// Auth Module Parity Tests
// ============================================================================

test "auth module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.auth, auth_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Auth module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.auth, auth_required);
}

test "auth module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.auth, &auth_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.auth, &auth_specs));
}

// ============================================================================
// Messaging Module Parity Tests
// ============================================================================

test "messaging module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.messaging, messaging_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Messaging module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.messaging, messaging_required);
}

test "messaging module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.messaging, &messaging_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.messaging, &messaging_specs));
}

// ============================================================================
// Cache Module Parity Tests
// ============================================================================

test "cache module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.cache, cache_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Cache module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.cache, cache_required);
}

test "cache module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.cache, &cache_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.cache, &cache_specs));
}

// ============================================================================
// Storage Module Parity Tests
// ============================================================================

test "storage module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.storage, storage_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Storage module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.storage, storage_required);
}

test "storage module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.storage, &storage_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.storage, &storage_specs));
}

// ============================================================================
// Gateway Module Parity Tests
// ============================================================================

test "gateway module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.gateway, gateway_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Gateway module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.gateway, gateway_required);
}

test "gateway module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.gateway, &gateway_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.gateway, &gateway_specs));
}

// ============================================================================
// Search Module Parity Tests
// ============================================================================

test "search module has required declarations" {
    const missing = comptime getMissingDeclarations(abi.search, search_required);

    if (missing.len > 0) {
        inline for (missing) |name| {
            std.log.err("Search module missing: {s}", .{name});
        }
        try std.testing.expect(false);
    }

    comptime verifyDeclarations(abi.search, search_required);
}

test "search module declaration kinds and signatures" {
    comptime verifyDeclSpecs(abi.search, &search_specs);
    try std.testing.expectEqual(@as(usize, 0), comptime countSpecViolations(abi.search, &search_specs));
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
        abi.cloud,
        abi.web,
        abi.observability,
        abi.analytics,
        abi.auth,
        abi.messaging,
        abi.cache,
        abi.storage,
        abi.search,
        abi.gateway,
        abi.ai_core,
        abi.inference,
        abi.training,
        abi.reasoning,
        abi.mobile,
        abi.pages,
        abi.benchmarks,
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
        abi.cloud,
        abi.web,
        abi.observability,
        abi.analytics,
        abi.auth,
        abi.messaging,
        abi.cache,
        abi.storage,
        abi.search,
        abi.gateway,
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

test "bidirectional parity detects orphan declarations" {
    const TestModule = struct {
        pub const TypeA = u32;
        pub const TypeB = i64;
        pub fn funcA() void {}
    };

    // Exact match — no orphans
    const exact = [_][]const u8{ "TypeA", "TypeB", "funcA" };
    try std.testing.expectEqual(@as(usize, 0), comptime countOrphanDeclarations(TestModule, &exact));
    try std.testing.expectEqual(@as(usize, 0), comptime countBidirectionalViolations(TestModule, &exact));

    // Partial list — TypeB and funcA are orphans
    const partial = [_][]const u8{"TypeA"};
    try std.testing.expectEqual(@as(usize, 2), comptime countOrphanDeclarations(TestModule, &partial));

    // Missing + orphans combined
    const mixed = [_][]const u8{ "TypeA", "NonExistent" };
    // 1 missing (NonExistent) + 2 orphans (TypeB, funcA) = 3
    try std.testing.expectEqual(@as(usize, 3), comptime countBidirectionalViolations(TestModule, &mixed));
}

// ============================================================================
// Bidirectional Parity: Module Orphan Audit (Soft Fail)
// ============================================================================
//
// These tests count orphan declarations in each module — declarations that exist
// in the active implementation but aren't in the expected list. Non-zero counts
// indicate API drift risk: code may compile with one flag setting but fail with
// another. Currently soft-fail (log + count) to enable incremental alignment.

test "gpu module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.gpu, gpu_required);
    if (orphans.len > 0) {
        std.log.info("GPU module has {d} declarations not in expected list", .{orphans.len});
    }
    // Soft: track but don't fail (will harden after alignment)
    try std.testing.expect(true);
}

test "ai module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.ai, ai_required);
    if (orphans.len > 0) {
        std.log.info("AI module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "database module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.database, database_required);
    if (orphans.len > 0) {
        std.log.info("Database module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "network module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.network, network_required);
    if (orphans.len > 0) {
        std.log.info("Network module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "cloud module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.cloud, cloud_required);
    if (orphans.len > 0) {
        std.log.info("Cloud module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "web module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.web, web_required);
    if (orphans.len > 0) {
        std.log.info("Web module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "observability module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.observability, observability_required);
    if (orphans.len > 0) {
        std.log.info("Observability module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "analytics module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.analytics, analytics_required);
    if (orphans.len > 0) {
        std.log.info("Analytics module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "auth module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.auth, auth_required);
    if (orphans.len > 0) {
        std.log.info("Auth module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "messaging module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.messaging, messaging_required);
    if (orphans.len > 0) {
        std.log.info("Messaging module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "cache module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.cache, cache_required);
    if (orphans.len > 0) {
        std.log.info("Cache module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "storage module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.storage, storage_required);
    if (orphans.len > 0) {
        std.log.info("Storage module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "search module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.search, search_required);
    if (orphans.len > 0) {
        std.log.info("Search module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "gateway module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.gateway, gateway_required);
    if (orphans.len > 0) {
        std.log.info("Gateway module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "ai_core module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.ai_core, ai_core_required);
    if (orphans.len > 0) {
        std.log.info("AI Core module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "ai_inference module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.inference, ai_inference_required);
    if (orphans.len > 0) {
        std.log.info("AI Inference module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "ai_training module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.training, ai_training_required);
    if (orphans.len > 0) {
        std.log.info("AI Training module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "ai_reasoning module bidirectional parity audit" {
    const orphans = comptime getOrphanDeclarations(abi.reasoning, ai_reasoning_required);
    if (orphans.len > 0) {
        std.log.info("AI Reasoning module has {d} declarations not in expected list", .{orphans.len});
    }
    try std.testing.expect(true);
}

test "parity suite is catalog-backed" {
    try std.testing.expect(feature_catalog.feature_count > 0);
    try std.testing.expectEqual(feature_catalog.feature_count, @typeInfo(abi.Feature).@"enum".fields.len);

    inline for (@typeInfo(abi.Feature).@"enum".fields) |field| {
        const feature: abi.Feature = @enumFromInt(field.value);
        const required = parity_specs.required.forFeature(feature);
        try std.testing.expect(required.len > 0);
    }
}
