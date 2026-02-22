//! Canonical feature catalog for ABI.
//!
//! Centralizes feature descriptions, compile-time flag mappings, parent-child
//! relationships, and real/stub module paths used by parity checks.

const std = @import("std");
const build_options = @import("build_options");

pub const Feature = enum {
    gpu,
    ai,
    llm,
    embeddings,
    agents,
    training,
    database,
    network,
    observability,
    web,
    personas,
    cloud,
    analytics,
    auth,
    messaging,
    cache,
    storage,
    search,
    mobile,
    gateway,
    pages,
    benchmarks,
    reasoning,
    constitution,

    pub fn name(self: Feature) []const u8 {
        return std.mem.sliceTo(@tagName(self), 0);
    }

    pub fn description(self: Feature) []const u8 {
        return metadata(self).description;
    }

    pub fn compileFlagField(self: Feature) []const u8 {
        return metadata(self).compile_flag_field;
    }

    pub fn isCompileTimeEnabled(comptime self: Feature) bool {
        return @field(build_options, self.compileFlagField());
    }

    pub fn paritySpec(self: Feature) ParitySpec {
        return metadata(self).parity_spec;
    }
};

/// Canonical parity-spec key consumed by API parity tests.
pub const ParitySpec = enum {
    gpu,
    ai,
    database,
    network,
    observability,
    web,
    analytics,
    cloud,
    auth,
    messaging,
    cache,
    storage,
    search,
    mobile,
    gateway,
    pages,
    benchmarks,
};

pub const Metadata = struct {
    feature: Feature,
    description: []const u8,
    compile_flag_field: []const u8,
    parity_spec: ParitySpec,
    parent: ?Feature = null,
    real_module_path: []const u8,
    stub_module_path: []const u8,
};

pub const all = [_]Metadata{
    .{
        .feature = .gpu,
        .description = "GPU acceleration and compute",
        .compile_flag_field = "enable_gpu",
        .parity_spec = .gpu,
        .real_module_path = "features/gpu/mod.zig",
        .stub_module_path = "features/gpu/stub.zig",
    },
    .{
        .feature = .ai,
        .description = "AI core functionality",
        .compile_flag_field = "enable_ai",
        .parity_spec = .ai,
        .real_module_path = "features/ai/mod.zig",
        .stub_module_path = "features/ai/stub.zig",
    },
    .{
        .feature = .llm,
        .description = "Local LLM inference",
        .compile_flag_field = "enable_llm",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/facades/inference.zig",
        .stub_module_path = "features/ai/facades/inference_stub.zig",
    },
    .{
        .feature = .embeddings,
        .description = "Vector embeddings generation",
        .compile_flag_field = "enable_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/mod.zig",
        .stub_module_path = "features/ai/stub.zig",
    },
    .{
        .feature = .agents,
        .description = "AI agent runtime",
        .compile_flag_field = "enable_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/facades/core.zig",
        .stub_module_path = "features/ai/facades/core_stub.zig",
    },
    .{
        .feature = .training,
        .description = "Model training pipelines",
        .compile_flag_field = "enable_training",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/facades/training.zig",
        .stub_module_path = "features/ai/facades/training_stub.zig",
    },
    .{
        .feature = .database,
        .description = "Vector database (WDBX)",
        .compile_flag_field = "enable_database",
        .parity_spec = .database,
        .real_module_path = "features/database/mod.zig",
        .stub_module_path = "features/database/stub.zig",
    },
    .{
        .feature = .network,
        .description = "Distributed compute network",
        .compile_flag_field = "enable_network",
        .parity_spec = .network,
        .real_module_path = "features/network/mod.zig",
        .stub_module_path = "features/network/stub.zig",
    },
    .{
        .feature = .observability,
        .description = "Metrics, tracing, profiling",
        .compile_flag_field = "enable_profiling",
        .parity_spec = .observability,
        .real_module_path = "features/observability/mod.zig",
        .stub_module_path = "features/observability/stub.zig",
    },
    .{
        .feature = .web,
        .description = "Web/HTTP utilities",
        .compile_flag_field = "enable_web",
        .parity_spec = .web,
        .real_module_path = "features/web/mod.zig",
        .stub_module_path = "features/web/stub.zig",
    },
    .{
        .feature = .personas,
        .description = "Multi-persona AI assistant",
        .compile_flag_field = "enable_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/mod.zig",
        .stub_module_path = "features/ai/stub.zig",
    },
    .{
        .feature = .cloud,
        .description = "Cloud provider integration",
        .compile_flag_field = "enable_cloud",
        .parity_spec = .cloud,
        .real_module_path = "features/cloud/mod.zig",
        .stub_module_path = "features/cloud/stub.zig",
    },
    .{
        .feature = .analytics,
        .description = "Analytics event tracking",
        .compile_flag_field = "enable_analytics",
        .parity_spec = .analytics,
        .real_module_path = "features/analytics/mod.zig",
        .stub_module_path = "features/analytics/stub.zig",
    },
    .{
        .feature = .auth,
        .description = "Authentication and security",
        .compile_flag_field = "enable_auth",
        .parity_spec = .auth,
        .real_module_path = "features/auth/mod.zig",
        .stub_module_path = "features/auth/stub.zig",
    },
    .{
        .feature = .messaging,
        .description = "Event bus and messaging",
        .compile_flag_field = "enable_messaging",
        .parity_spec = .messaging,
        .real_module_path = "features/messaging/mod.zig",
        .stub_module_path = "features/messaging/stub.zig",
    },
    .{
        .feature = .cache,
        .description = "In-memory caching",
        .compile_flag_field = "enable_cache",
        .parity_spec = .cache,
        .real_module_path = "features/cache/mod.zig",
        .stub_module_path = "features/cache/stub.zig",
    },
    .{
        .feature = .storage,
        .description = "Unified file/object storage",
        .compile_flag_field = "enable_storage",
        .parity_spec = .storage,
        .real_module_path = "features/storage/mod.zig",
        .stub_module_path = "features/storage/stub.zig",
    },
    .{
        .feature = .search,
        .description = "Full-text search",
        .compile_flag_field = "enable_search",
        .parity_spec = .search,
        .real_module_path = "features/search/mod.zig",
        .stub_module_path = "features/search/stub.zig",
    },
    .{
        .feature = .mobile,
        .description = "Mobile platform support",
        .compile_flag_field = "enable_mobile",
        .parity_spec = .mobile,
        .real_module_path = "features/mobile/mod.zig",
        .stub_module_path = "features/mobile/stub.zig",
    },
    .{
        .feature = .gateway,
        .description = "API gateway (routing, rate limiting, circuit breaker)",
        .compile_flag_field = "enable_gateway",
        .parity_spec = .gateway,
        .real_module_path = "features/gateway/mod.zig",
        .stub_module_path = "features/gateway/stub.zig",
    },
    .{
        .feature = .pages,
        .description = "Dashboard/UI pages with URL routing",
        .compile_flag_field = "enable_pages",
        .parity_spec = .pages,
        .real_module_path = "features/pages/mod.zig",
        .stub_module_path = "features/pages/stub.zig",
    },
    .{
        .feature = .benchmarks,
        .description = "Performance benchmarking and timing",
        .compile_flag_field = "enable_benchmarks",
        .parity_spec = .benchmarks,
        .real_module_path = "features/benchmarks/mod.zig",
        .stub_module_path = "features/benchmarks/stub.zig",
    },
    .{
        .feature = .reasoning,
        .description = "AI reasoning (Abbey, eval, RAG)",
        .compile_flag_field = "enable_reasoning",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/facades/reasoning.zig",
        .stub_module_path = "features/ai/facades/reasoning_stub.zig",
    },
    .{
        .feature = .constitution,
        .description = "AI safety principles and guardrails",
        .compile_flag_field = "enable_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/mod.zig",
        .stub_module_path = "features/ai/stub.zig",
    },
};

pub const feature_count = all.len;

pub fn fromEnum(feature: anytype) Feature {
    return @enumFromInt(@intFromEnum(feature));
}

pub fn toEnum(comptime T: type, feature: Feature) T {
    return @enumFromInt(@intFromEnum(feature));
}

pub fn metadata(feature: Feature) Metadata {
    return all[@intFromEnum(feature)];
}

pub fn description(feature: Feature) []const u8 {
    return metadata(feature).description;
}

pub fn compileFlagField(feature: Feature) []const u8 {
    return metadata(feature).compile_flag_field;
}

pub fn parent(feature: Feature) ?Feature {
    return metadata(feature).parent;
}

pub fn paritySpec(feature: Feature) ParitySpec {
    return metadata(feature).parity_spec;
}

pub fn featureForCompileFlag(comptime T: type, flag: []const u8) ?T {
    inline for (all) |entry| {
        if (std.mem.eql(u8, entry.compile_flag_field, flag)) {
            return toEnum(T, entry.feature);
        }
    }
    return null;
}

pub fn compileFlagHasFeature(comptime T: type, feature: T) bool {
    inline for (all) |entry| {
        if (toEnum(Feature, entry.feature) == toEnum(Feature, feature)) {
            return true;
        }
    }
    return false;
}

pub fn descriptionFromEnum(feature: anytype) []const u8 {
    return description(fromEnum(feature));
}

pub fn compileFlagFieldFromEnum(feature: anytype) []const u8 {
    return compileFlagField(fromEnum(feature));
}

pub fn parentAsEnum(comptime T: type, feature: anytype) ?T {
    if (parent(fromEnum(feature))) |p| {
        return toEnum(T, p);
    }
    return null;
}

test "catalog order matches feature enum ordinals" {
    inline for (all, 0..) |entry, idx| {
        try std.testing.expectEqual(entry.feature, @as(Feature, @enumFromInt(idx)));
    }
}

test "catalog enum methods remain canonical" {
    try std.testing.expectEqualStrings("gpu", Feature.gpu.name());
    try std.testing.expectEqual(Feature.gpu.description(), description(.gpu));
    try std.testing.expectEqual(Feature.gpu.compileFlagField(), compileFlagField(.gpu));
    try std.testing.expectEqual(Feature.gpu.isCompileTimeEnabled(), @field(build_options, compileFlagField(.gpu)));
    try std.testing.expect(featureForCompileFlag(Feature, Feature.gpu.compileFlagField()) != null);
}

test "ai subfeatures point to ai parent" {
    try std.testing.expectEqual(Feature.ai, parent(.llm).?);
    try std.testing.expectEqual(Feature.ai, parent(.embeddings).?);
    try std.testing.expectEqual(Feature.ai, parent(.agents).?);
    try std.testing.expectEqual(Feature.ai, parent(.training).?);
    try std.testing.expectEqual(Feature.ai, parent(.personas).?);
    try std.testing.expectEqual(Feature.ai, parent(.reasoning).?);
    try std.testing.expectEqual(Feature.ai, parent(.constitution).?);
}
