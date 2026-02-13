//! AI Core Module — Agents, Tools, Prompts, Memory
//!
//! This module provides the foundational AI building blocks: agents, tool
//! registries, prompt builders, multi-agent coordination, memory, and model
//! discovery. It is always available when the `ai` feature flag is enabled.
//!
//! For LLM inference, embeddings, and vision see `ai_inference`.
//! For training pipelines see `ai_training`.
//! For advanced reasoning (Abbey, RAG, eval) see `ai_reasoning`.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../core/config/mod.zig");

// ============================================================================
// Sub-module re-exports (from features/ai/)
// ============================================================================

pub const agent = @import("../ai/agent.zig");
pub const agents = if (build_options.enable_ai)
    @import("../ai/agents/mod.zig")
else
    @import("../ai/agents/stub.zig");
pub const tools = @import("../ai/tools/mod.zig");
pub const prompts = @import("../ai/prompts/mod.zig");
pub const memory = @import("../ai/memory/mod.zig");
pub const multi_agent = if (build_options.enable_ai)
    @import("../ai/multi_agent/mod.zig")
else
    @import("../ai/multi_agent/stub.zig");
pub const core = @import("../ai/core/mod.zig");
pub const gpu_agent = @import("../ai/gpu_agent.zig");
pub const discovery = @import("../ai/discovery.zig");
pub const models = if (build_options.enable_ai)
    @import("../ai/models/mod.zig")
else
    @import("../ai/models/stub.zig");
pub const model_registry = @import("../ai/model_registry.zig");

// ============================================================================
// Convenience type re-exports
// ============================================================================

pub const Agent = agent.Agent;
pub const MultiAgentCoordinator = multi_agent.Coordinator;
pub const ModelRegistry = model_registry.ModelRegistry;
pub const ModelInfo = model_registry.ModelInfo;
pub const Tool = tools.Tool;
pub const ToolResult = tools.ToolResult;
pub const ToolRegistry = tools.ToolRegistry;
pub const TaskTool = tools.TaskTool;
pub const Subagent = tools.Subagent;
pub const DiscordTools = tools.DiscordTools;
pub const registerDiscordTools = tools.registerDiscordTools;
pub const OsTools = tools.OsTools;
pub const registerOsTools = tools.registerOsTools;
pub const PromptBuilder = prompts.PromptBuilder;
pub const Persona = prompts.Persona;
pub const PersonaType = prompts.PersonaType;
pub const PromptFormat = prompts.PromptFormat;
pub const GpuAgent = gpu_agent.GpuAgent;
pub const GpuAwareRequest = gpu_agent.GpuAwareRequest;
pub const GpuAwareResponse = gpu_agent.GpuAwareResponse;
pub const WorkloadType = gpu_agent.WorkloadType;
pub const GpuAgentPriority = gpu_agent.Priority;
pub const GpuAgentStats = gpu_agent.AgentStats;
pub const ModelDiscovery = discovery.ModelDiscovery;
pub const DiscoveredModel = discovery.DiscoveredModel;
pub const DiscoveryConfig = discovery.DiscoveryConfig;
pub const SystemCapabilities = discovery.SystemCapabilities;
pub const AdaptiveConfig = discovery.AdaptiveConfig;
pub const ModelRequirements = discovery.ModelRequirements;
pub const WarmupResult = discovery.WarmupResult;
pub const detectCapabilities = discovery.detectCapabilities;
pub const runWarmup = discovery.runWarmup;

// Core types
pub const Confidence = core.Confidence;
pub const ConfidenceLevel = core.ConfidenceLevel;
pub const EmotionalState = core.EmotionalState;
pub const EmotionType = core.EmotionType;

// ============================================================================
// Error
// ============================================================================

pub const Error = error{
    AiDisabled,
    AgentsDisabled,
    ModelNotFound,
    InvalidConfig,
};

// ============================================================================
// Context
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.AiConfig,
    agents_ctx: ?*agents.Context = null,
    model_discovery: ?*discovery.ModelDiscovery = null,
    capabilities: discovery.SystemCapabilities = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: config_module.AiConfig,
    ) !*Context {
        if (!isEnabled()) return error.AiDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
            .capabilities = discovery.detectCapabilities(),
        };

        if (cfg.auto_discover) {
            const disc = try allocator.create(discovery.ModelDiscovery);
            disc.* = discovery.ModelDiscovery.init(allocator, .{});
            disc.scanAll() catch |err| {
                std.log.debug(
                    "Model discovery scan failed (best effort): {t}",
                    .{err},
                );
            };
            ctx.model_discovery = disc;
        }

        if (cfg.agents) |agent_cfg| {
            ctx.agents_ctx = try agents.Context.init(allocator, agent_cfg);
        }

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.agents_ctx) |a| a.deinit();
        if (self.model_discovery) |disc| {
            disc.deinit();
            self.allocator.destroy(disc);
        }
        self.allocator.destroy(self);
    }

    pub fn getAgents(self: *Context) Error!*agents.Context {
        return self.agents_ctx orelse error.AgentsDisabled;
    }
};

// ============================================================================
// Module-level functions
// ============================================================================

pub fn isEnabled() bool {
    return build_options.enable_ai;
}

pub fn createRegistry(allocator: std.mem.Allocator) ModelRegistry {
    return ModelRegistry.init(allocator);
}

pub fn createAgent(allocator: std.mem.Allocator, name: []const u8) !Agent {
    if (!isEnabled()) return error.AiDisabled;
    return agent.Agent.init(allocator, .{ .name = name });
}

// ============================================================================
// Tests
// ============================================================================

test "ai_core module loads" {
    try std.testing.expect(@TypeOf(Agent) != void);
    try std.testing.expect(@TypeOf(ToolRegistry) != void);
    try std.testing.expect(@TypeOf(PromptBuilder) != void);
}
