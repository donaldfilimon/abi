//! AI Core Stub Module
//!
//! Provides API-compatible no-op implementations when AI core is disabled.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

pub const Error = error{
    AiDisabled,
    AgentsDisabled,
    ModelNotFound,
    InvalidConfig,
};

// Sub-module stubs
pub const core = @import("../ai/core/stub.zig");
pub const agents = @import("../ai/agents/stub.zig");
pub const tools = @import("../ai/tools/stub.zig");
pub const prompts = @import("../ai/prompts/stub.zig");
pub const memory = @import("../ai/memory/stub.zig");
pub const multi_agent = @import("../ai/multi_agent/stub.zig");
pub const models = @import("../ai/models/stub.zig");
pub const agent = @import("../ai/stubs/agent.zig");
pub const model_registry = @import("../ai/stubs/model_registry.zig");
pub const gpu_agent = @import("../ai/stubs/gpu_agent.zig");
pub const discovery = @import("../ai/stubs/discovery.zig");

// Re-exports
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
pub const Confidence = core.Confidence;
pub const ConfidenceLevel = core.ConfidenceLevel;
pub const EmotionalState = core.EmotionalState;
pub const EmotionType = core.EmotionType;

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.AiConfig,
    agents_ctx: ?*agents.Context = null,
    model_discovery: ?*discovery.ModelDiscovery = null,
    capabilities: discovery.SystemCapabilities = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        _: config_module.AiConfig,
    ) !*Context {
        _ = allocator;
        return error.AiDisabled;
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }

    pub fn getAgents(self: *Context) Error!*agents.Context {
        _ = self;
        return error.AiDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn createRegistry(allocator: std.mem.Allocator) ModelRegistry {
    return ModelRegistry.init(allocator);
}

pub fn createAgent(_: std.mem.Allocator, _: []const u8) !Agent {
    return error.AiDisabled;
}
