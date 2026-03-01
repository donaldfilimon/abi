//! AI Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

pub const Error = error{
    /// AI feature is disabled at compile time
    AiDisabled,
    /// LLM sub-feature is disabled
    LlmDisabled,
    /// Embeddings sub-feature is disabled
    EmbeddingsDisabled,
    /// Agents sub-feature is disabled
    AgentsDisabled,
    /// Training sub-feature is disabled
    TrainingDisabled,
    /// Model not found
    ModelNotFound,
    /// Inference failed
    InferenceFailed,
    /// Invalid configuration
    InvalidConfig,
};

// Sub-module stubs (each has its own stub.zig)
pub const core = @import("core/stub.zig");
pub const llm = @import("llm/stub.zig");
pub const embeddings = @import("embeddings/stub.zig");
pub const agents = @import("agents/stub.zig");
pub const training = @import("training/stub.zig");
pub const database = @import("database/stub.zig");
pub const documents = @import("documents/stub.zig");
pub const vision = @import("vision/stub.zig");
pub const orchestration = @import("orchestration/stub.zig");
pub const multi_agent = @import("multi_agent/stub.zig");
pub const models = @import("models/stub.zig");
pub const memory = @import("memory/stub.zig");
pub const streaming = @import("streaming/stub.zig");
pub const explore = @import("explore/stub.zig");
pub const personas = @import("personas/stub.zig");
pub const rag = @import("rag/stub.zig");
pub const templates = @import("templates/stub.zig");
pub const eval = @import("eval/stub.zig");
pub const federated = @import("federated/stub.zig");
pub const tools = @import("tools/stub.zig");
pub const transformer = @import("transformer/stub.zig");
pub const prompts = @import("prompts/stub.zig");
pub const abbey = @import("abbey/stub.zig");
pub const constitution = @import("constitution/stub.zig");

// Local stubs for single-file modules (merged into subdirectory stubs)
pub const agent = @import("agents/stub.zig");
pub const model_registry = @import("models/stub.zig");
pub const tool_agent = @import("tools/stub.zig");
pub const codebase_index = @import("explore/stub.zig");
pub const self_improve = @import("self_improve.zig");
pub const gpu_agent = @import("agents/stub.zig");
pub const discovery = @import("explore/stub.zig");

// NOTE(v0.4.0): Flat compatibility re-exports removed.
// Use canonical sub-module paths instead:
//   abi.features.ai.agent.Agent          (was abi.features.ai.Agent)
//   abi.features.ai.multi_agent.Coordinator (was abi.features.ai.MultiAgentCoordinator)
//   abi.features.ai.tools.ToolRegistry   (was abi.features.ai.ToolRegistry)
//   abi.features.ai.training.*           (was abi.features.ai.TrainingConfig, etc.)
//   abi.features.ai.orchestration.TaskType (was abi.features.ai.TaskType)
//   abi.features.ai.streaming.StreamToken (was abi.features.ai.StreamToken)
//   abi.features.ai.llm.InferenceConfig  (was abi.features.ai.LlmConfig)
//   abi.features.ai.self_improve.SelfImprover (was abi.features.ai.SelfImprover)
//   abi.features.ai.database.*           (was abi.features.ai.WdbxTokenDataset, etc.)

// Context
pub const Context = struct {
    pub const SubFeature = enum { llm, embeddings, agents, training, personas };

    pub fn SubFeatureContext(comptime feature: SubFeature) type {
        return switch (feature) {
            .llm => llm.Context,
            .embeddings => embeddings.Context,
            .agents => agents.Context,
            .training => training.Context,
            .personas => personas.Context,
        };
    }

    pub fn init(_: std.mem.Allocator, _: config_module.AiConfig) Error!*Context {
        return error.AiDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn get(_: *Context, comptime _: SubFeature) Error!*SubFeatureContext(SubFeature.llm) {
        return error.AiDisabled;
    }
    pub fn isSubFeatureEnabled(_: *Context, _: SubFeature) bool {
        return false;
    }
    pub fn getDiscoveredModels(_: *Context) []discovery.DiscoveredModel {
        return &.{};
    }
    pub fn discoveredModelCount(_: *Context) usize {
        return 0;
    }
    pub fn findBestModel(_: *Context, _: discovery.ModelRequirements) ?*discovery.DiscoveredModel {
        return null;
    }
    pub fn generateAdaptiveConfig(_: *Context, _: *const discovery.DiscoveredModel) discovery.AdaptiveConfig {
        return .{};
    }
    pub fn getCapabilities(_: *const Context) discovery.SystemCapabilities {
        return .{};
    }
    pub fn addModelPath(_: *Context, _: []const u8) !void {
        return error.AiDisabled;
    }
    pub fn addModelWithSize(_: *Context, _: []const u8, _: u64) !void {
        return error.AiDisabled;
    }
    pub fn clearDiscoveredModels(_: *Context) void {}
};

pub fn isEnabled() bool {
    return false;
}
pub fn isLlmEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return false;
}
pub fn init(_: std.mem.Allocator) Error!void {
    return error.AiDisabled;
}
pub fn deinit() void {}

pub fn createAgent(_: std.mem.Allocator, _: []const u8) !agent.Agent {
    return error.AiDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
