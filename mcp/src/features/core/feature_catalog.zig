//! Canonical feature catalog for ABI.
//!
//! Centralizes feature descriptions, compile-time flag mappings, parent-child
//! relationships, and real/stub module paths used by parity checks.

const std = @import("std");
const build_options = @import("build_options");

pub const Feature = enum {
    gpu,
    ai,
    core,
    llm,
    embeddings,
    agents,
    training,
    streaming,
    explore,
    abbey,
    tools,
    prompts,
    memory,
    reasoning,
    constitution,
    pipeline,
    eval,
    rag,
    templates,
    orchestration,
    ai_documents,
    ai_database,
    vision,
    multi_agent,
    coordination,
    models,
    transformer,
    federated,
    feedback,
    compliance,
    database,
    profiles,
    profile,
    context_engine,
    self_improve,
    network,
    observability,
    web,
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
    compute,
    documents,
    desktop,
    tui,
    lsp,
    mcp,
    acp,
    ha,
    connectors,
    tasks,
    inference,

    pub fn name(self: Feature) []const u8 {
        return @tagName(self);
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
    compute,
    documents,
    desktop,
    tui,
    lsp,
    mcp,
    acp,
    ha,
    connectors,
    tasks,
    inference,
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
        .compile_flag_field = "feat_gpu",
        .parity_spec = .gpu,
        .real_module_path = "features/gpu/mod.zig",
        .stub_module_path = "features/gpu/stub.zig",
    },
    .{
        .feature = .ai,
        .description = "AI core functionality",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .real_module_path = "features/ai/mod.zig",
        .stub_module_path = "features/ai/stub.zig",
    },
    .{
        .feature = .core,
        .description = "AI core context and shared facades",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/core/mod.zig",
        .stub_module_path = "features/ai/core/stub.zig",
    },
    .{
        .feature = .llm,
        .description = "Local LLM inference",
        .compile_flag_field = "feat_llm",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/llm/mod.zig",
        .stub_module_path = "features/ai/llm/stub.zig",
    },
    .{
        .feature = .embeddings,
        .description = "Vector embeddings generation",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/embeddings/mod.zig",
        .stub_module_path = "features/ai/embeddings/stub.zig",
    },
    .{
        .feature = .agents,
        .description = "AI agent runtime",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/agents/mod.zig",
        .stub_module_path = "features/ai/agents/stub.zig",
    },
    .{
        .feature = .training,
        .description = "Model training pipelines",
        .compile_flag_field = "feat_training",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/training/mod.zig",
        .stub_module_path = "features/ai/training/stub.zig",
    },
    .{
        .feature = .streaming,
        .description = "Streaming AI responses and transport helpers",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/streaming/mod.zig",
        .stub_module_path = "features/ai/streaming/stub.zig",
    },
    .{
        .feature = .explore,
        .description = "AI exploration and discovery workflows",
        .compile_flag_field = "feat_explore",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/explore/mod.zig",
        .stub_module_path = "features/ai/explore/stub.zig",
    },
    .{
        .feature = .abbey,
        .description = "Abbey reasoning profile runtime",
        .compile_flag_field = "feat_reasoning",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/abbey/mod.zig",
        .stub_module_path = "features/ai/abbey/stub.zig",
    },
    .{
        .feature = .tools,
        .description = "AI tool execution and adapters",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/tools/mod.zig",
        .stub_module_path = "features/ai/tools/stub.zig",
    },
    .{
        .feature = .prompts,
        .description = "Prompt construction and formatting",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/prompts/mod.zig",
        .stub_module_path = "features/ai/prompts/stub.zig",
    },
    .{
        .feature = .memory,
        .description = "AI memory storage and retrieval",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/memory/mod.zig",
        .stub_module_path = "features/ai/memory/stub.zig",
    },
    .{
        .feature = .reasoning,
        .description = "AI reasoning (Abbey, eval, RAG)",
        .compile_flag_field = "feat_reasoning",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/reasoning/mod.zig",
        .stub_module_path = "features/ai/reasoning/stub.zig",
    },
    .{
        .feature = .constitution,
        .description = "AI safety principles and guardrails",
        .compile_flag_field = "feat_reasoning",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/constitution/mod.zig",
        .stub_module_path = "features/ai/constitution/stub.zig",
    },
    .{
        .feature = .pipeline,
        .description = "Composable AI pipeline DSL",
        .compile_flag_field = "feat_reasoning",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/pipeline/mod.zig",
        .stub_module_path = "features/ai/pipeline/stub.zig",
    },
    .{
        .feature = .eval,
        .description = "AI evaluation and scoring",
        .compile_flag_field = "feat_reasoning",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/eval/mod.zig",
        .stub_module_path = "features/ai/eval/stub.zig",
    },
    .{
        .feature = .rag,
        .description = "Retrieval-augmented generation",
        .compile_flag_field = "feat_reasoning",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/rag/mod.zig",
        .stub_module_path = "features/ai/rag/stub.zig",
    },
    .{
        .feature = .templates,
        .description = "AI prompt and workflow templates",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/templates/mod.zig",
        .stub_module_path = "features/ai/templates/stub.zig",
    },
    .{
        .feature = .orchestration,
        .description = "AI orchestration flows",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/orchestration/mod.zig",
        .stub_module_path = "features/ai/orchestration/stub.zig",
    },
    .{
        .feature = .ai_documents,
        .description = "AI-native document workflows",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/documents/mod.zig",
        .stub_module_path = "features/ai/documents/stub.zig",
    },
    .{
        .feature = .ai_database,
        .description = "AI memory and database adapters",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/database/mod.zig",
        .stub_module_path = "features/ai/database/stub.zig",
    },
    .{
        .feature = .vision,
        .description = "Vision model support",
        .compile_flag_field = "feat_vision",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/vision/mod.zig",
        .stub_module_path = "features/ai/vision/stub.zig",
    },
    .{
        .feature = .multi_agent,
        .description = "Multi-agent coordination runtime",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/multi_agent/mod.zig",
        .stub_module_path = "features/ai/multi_agent/stub.zig",
    },
    .{
        .feature = .coordination,
        .description = "Agent coordination primitives",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/coordination/mod.zig",
        .stub_module_path = "features/ai/coordination/stub.zig",
    },
    .{
        .feature = .models,
        .description = "AI model registry and selection",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/models/mod.zig",
        .stub_module_path = "features/ai/models/stub.zig",
    },
    .{
        .feature = .transformer,
        .description = "Transformer model utilities",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/transformer/mod.zig",
        .stub_module_path = "features/ai/transformer/stub.zig",
    },
    .{
        .feature = .federated,
        .description = "Federated AI learning",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/federated/mod.zig",
        .stub_module_path = "features/ai/federated/stub.zig",
    },
    .{
        .feature = .feedback,
        .description = "AI feedback capture and learning",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/feedback/mod.zig",
        .stub_module_path = "features/ai/feedback/stub.zig",
    },
    .{
        .feature = .compliance,
        .description = "AI compliance and governance checks",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/compliance/mod.zig",
        .stub_module_path = "features/ai/compliance/stub.zig",
    },
    .{
        .feature = .database,
        .description = "Vector database (WDBX)",
        .compile_flag_field = "feat_database",
        .parity_spec = .database,
        .real_module_path = "features/database/mod.zig",
        .stub_module_path = "features/database/stub.zig",
    },
    .{
        .feature = .profiles,
        .description = "Behavior profile routing and selection",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/profiles/mod.zig",
        .stub_module_path = "features/ai/profiles/stub.zig",
    },
    .{
        .feature = .profile,
        .description = "Multi-profile orchestration layer",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/profile/mod.zig",
        .stub_module_path = "features/ai/profile/stub.zig",
    },
    .{
        .feature = .context_engine,
        .description = "Context assembly and grounding",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/context_engine/mod.zig",
        .stub_module_path = "features/ai/context_engine/stub.zig",
    },
    .{
        .feature = .self_improve,
        .description = "Self-improvement routines",
        .compile_flag_field = "feat_ai",
        .parity_spec = .ai,
        .parent = .ai,
        .real_module_path = "features/ai/self_improve.zig",
        .stub_module_path = "features/ai/self_improve_stub.zig",
    },
    .{
        .feature = .network,
        .description = "Distributed compute network",
        .compile_flag_field = "feat_network",
        .parity_spec = .network,
        .real_module_path = "features/network/mod.zig",
        .stub_module_path = "features/network/stub.zig",
    },
    .{
        .feature = .observability,
        .description = "Metrics, tracing, profiling",
        .compile_flag_field = "feat_observability",
        .parity_spec = .observability,
        .real_module_path = "features/observability/mod.zig",
        .stub_module_path = "features/observability/stub.zig",
    },
    .{
        .feature = .web,
        .description = "Web/HTTP utilities",
        .compile_flag_field = "feat_web",
        .parity_spec = .web,
        .real_module_path = "features/web/mod.zig",
        .stub_module_path = "features/web/stub.zig",
    },
    .{
        .feature = .cloud,
        .description = "Cloud provider integration",
        .compile_flag_field = "feat_cloud",
        .parity_spec = .cloud,
        .real_module_path = "features/cloud/mod.zig",
        .stub_module_path = "features/cloud/stub.zig",
    },
    .{
        .feature = .analytics,
        .description = "Analytics event tracking",
        .compile_flag_field = "feat_analytics",
        .parity_spec = .analytics,
        .real_module_path = "features/analytics/mod.zig",
        .stub_module_path = "features/analytics/stub.zig",
    },
    .{
        .feature = .auth,
        .description = "Authentication and security",
        .compile_flag_field = "feat_auth",
        .parity_spec = .auth,
        .real_module_path = "features/auth/mod.zig",
        .stub_module_path = "features/auth/stub.zig",
    },
    .{
        .feature = .messaging,
        .description = "Event bus and messaging",
        .compile_flag_field = "feat_messaging",
        .parity_spec = .messaging,
        .real_module_path = "features/messaging/mod.zig",
        .stub_module_path = "features/messaging/stub.zig",
    },
    .{
        .feature = .cache,
        .description = "In-memory caching",
        .compile_flag_field = "feat_cache",
        .parity_spec = .cache,
        .real_module_path = "features/cache/mod.zig",
        .stub_module_path = "features/cache/stub.zig",
    },
    .{
        .feature = .storage,
        .description = "Unified file/object storage",
        .compile_flag_field = "feat_storage",
        .parity_spec = .storage,
        .real_module_path = "features/storage/mod.zig",
        .stub_module_path = "features/storage/stub.zig",
    },
    .{
        .feature = .search,
        .description = "Full-text search",
        .compile_flag_field = "feat_search",
        .parity_spec = .search,
        .real_module_path = "features/search/mod.zig",
        .stub_module_path = "features/search/stub.zig",
    },
    .{
        .feature = .mobile,
        .description = "Mobile platform support",
        .compile_flag_field = "feat_mobile",
        .parity_spec = .mobile,
        .real_module_path = "features/mobile/mod.zig",
        .stub_module_path = "features/mobile/stub.zig",
    },
    .{
        .feature = .gateway,
        .description = "API gateway (routing, rate limiting, circuit breaker)",
        .compile_flag_field = "feat_gateway",
        .parity_spec = .gateway,
        .real_module_path = "features/gateway/mod.zig",
        .stub_module_path = "features/gateway/stub.zig",
    },
    .{
        .feature = .pages,
        .description = "Dashboard/UI pages with URL routing",
        .compile_flag_field = "feat_pages",
        .parity_spec = .pages,
        .real_module_path = "features/observability/pages/mod.zig",
        .stub_module_path = "features/observability/pages/stub.zig",
    },
    .{
        .feature = .benchmarks,
        .description = "Performance benchmarking and timing",
        .compile_flag_field = "feat_benchmarks",
        .parity_spec = .benchmarks,
        .real_module_path = "features/benchmarks/mod.zig",
        .stub_module_path = "features/benchmarks/stub.zig",
    },
    .{
        .feature = .compute,
        .description = "Distributed compute mesh",
        .compile_flag_field = "feat_compute",
        .parity_spec = .compute,
        .real_module_path = "features/compute/mod.zig",
        .stub_module_path = "features/compute/stub.zig",
    },
    .{
        .feature = .documents,
        .description = "Native document parsing (HTML, PDF)",
        .compile_flag_field = "feat_documents",
        .parity_spec = .documents,
        .real_module_path = "features/documents/mod.zig",
        .stub_module_path = "features/documents/stub.zig",
    },
    .{
        .feature = .desktop,
        .description = "Native desktop OS extensions",
        .compile_flag_field = "feat_desktop",
        .parity_spec = .desktop,
        .real_module_path = "features/desktop/mod.zig",
        .stub_module_path = "features/desktop/stub.zig",
    },
    .{
        .feature = .tui,
        .description = "Terminal user interface",
        .compile_flag_field = "feat_tui",
        .parity_spec = .tui,
        .real_module_path = "features/tui/mod.zig",
        .stub_module_path = "features/tui/stub.zig",
    },
    .{
        .feature = .lsp,
        .description = "LSP (ZLS) service",
        .compile_flag_field = "feat_lsp",
        .parity_spec = .lsp,
        .real_module_path = "protocols/lsp/mod.zig",
        .stub_module_path = "protocols/lsp/stub.zig",
    },
    .{
        .feature = .mcp,
        .description = "MCP (Model Context Protocol) service",
        .compile_flag_field = "feat_mcp",
        .parity_spec = .mcp,
        .real_module_path = "protocols/mcp/mod.zig",
        .stub_module_path = "protocols/mcp/stub.zig",
    },
    .{
        .feature = .acp,
        .description = "Agent Communication Protocol",
        .compile_flag_field = "feat_acp",
        .parity_spec = .acp,
        .real_module_path = "protocols/acp/mod.zig",
        .stub_module_path = "protocols/acp/stub.zig",
    },
    .{
        .feature = .ha,
        .description = "High availability, replication, failover",
        .compile_flag_field = "feat_ha",
        .parity_spec = .ha,
        .real_module_path = "protocols/ha/mod.zig",
        .stub_module_path = "protocols/ha/stub.zig",
    },
    .{
        .feature = .connectors,
        .description = "External service adapters (LLM providers, Discord, etc.)",
        .compile_flag_field = "feat_connectors",
        .parity_spec = .connectors,
        .real_module_path = "connectors/mod.zig",
        .stub_module_path = "connectors/stub.zig",
    },
    .{
        .feature = .tasks,
        .description = "Task management and async job queues",
        .compile_flag_field = "feat_tasks",
        .parity_spec = .tasks,
        .real_module_path = "tasks/mod.zig",
        .stub_module_path = "tasks/stub.zig",
    },
    .{
        .feature = .inference,
        .description = "ML inference engine, scheduler, sampler",
        .compile_flag_field = "feat_inference",
        .parity_spec = .inference,
        .real_module_path = "inference/mod.zig",
        .stub_module_path = "inference/stub.zig",
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
    const ai_children = [_]Feature{
        .core,
        .llm,
        .embeddings,
        .agents,
        .training,
        .streaming,
        .explore,
        .abbey,
        .tools,
        .prompts,
        .memory,
        .reasoning,
        .constitution,
        .pipeline,
        .eval,
        .rag,
        .templates,
        .orchestration,
        .ai_documents,
        .ai_database,
        .vision,
        .multi_agent,
        .coordination,
        .models,
        .transformer,
        .federated,
        .feedback,
        .compliance,
        .profiles,
        .profile,
        .context_engine,
        .self_improve,
    };

    for (ai_children) |feature| {
        try std.testing.expectEqual(Feature.ai, parent(feature).?);
    }
}

test "profile and profiles remain distinct catalog entries" {
    try std.testing.expectEqualStrings("features/ai/profile/mod.zig", metadata(.profile).real_module_path);
    try std.testing.expectEqualStrings("features/ai/profiles/mod.zig", metadata(.profiles).real_module_path);
}

test {
    std.testing.refAllDecls(@This());
}
