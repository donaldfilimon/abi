//! AI Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

pub const Error = error{ FeatureDisabled, ModelNotFound, InferenceFailed, InvalidConfig };

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

// Compatibility re-exports
pub const Agent = agent.Agent;
pub const MultiAgentCoordinator = multi_agent.Coordinator;
pub const ToolRegistry = tools.ToolRegistry;
pub const DiscordTools = tools.DiscordTools;
pub const registerDiscordTools = tools.registerDiscordTools;
pub const ToolAugmentedAgent = tool_agent.ToolAugmentedAgent;
pub const TrainingConfig = training.TrainingConfig;
pub const TrainingResult = training.TrainingResult;
pub const LlmTrainingConfig = training.LlmTrainingConfig;
pub const LlamaTrainer = training.LlamaTrainer;
pub const OptimizerType = training.OptimizerType;
pub const LearningRateSchedule = training.LearningRateSchedule;
pub const LoraConfig = training.LoraConfig;
pub const LoraModel = training.LoraModel;
pub const TrainableModel = training.TrainableModel;
pub const trainable_model = training.trainable_model;
pub const TrainableModelConfig = training.trainable_model.TrainableModelConfig;
pub const TrainableViTModel = training.TrainableViTModel;
pub const TrainableViTConfig = training.TrainableViTConfig;
pub const TrainableCLIPModel = training.TrainableCLIPModel;
pub const CLIPTrainingConfig = training.CLIPTrainingConfig;
pub const TokenizedDataset = training.TokenizedDataset;
pub const parseInstructionDataset = training.parseInstructionDataset;
pub const WdbxTokenDataset = database.WdbxTokenDataset;
pub const loadCheckpoint = training.loadCheckpoint;
pub const train = training.train;
pub const trainWithResult = training.trainWithResult;
pub const readTokenBinFile = database.readTokenBinFile;
pub const writeTokenBinFile = database.writeTokenBinFile;
pub const tokenBinToWdbx = database.tokenBinToWdbx;
pub const wdbxToTokenBin = database.wdbxToTokenBin;
pub const TaskType = orchestration.TaskType;
pub const StreamToken = streaming.StreamToken;
pub const LlmConfig = llm.InferenceConfig;
pub const SelfImprover = self_improve.SelfImprover;

// Context
pub const Context = struct {
    pub const SubFeature = enum { llm, embeddings, agents, training, personas };
    pub fn init(_: std.mem.Allocator, _: config_module.AiConfig) Error!*Context {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn getLlm(_: *Context) Error!*llm.Context {
        return error.FeatureDisabled;
    }
    pub fn getEmbeddings(_: *Context) Error!*embeddings.Context {
        return error.FeatureDisabled;
    }
    pub fn getAgents(_: *Context) Error!*agents.Context {
        return error.FeatureDisabled;
    }
    pub fn getTraining(_: *Context) Error!*training.Context {
        return error.FeatureDisabled;
    }
    pub fn getPersonas(_: *Context) Error!*personas.Context {
        return error.FeatureDisabled;
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
        return error.FeatureDisabled;
    }
    pub fn addModelWithSize(_: *Context, _: []const u8, _: u64) !void {
        return error.FeatureDisabled;
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
    return error.FeatureDisabled;
}
pub fn deinit() void {}

pub fn createAgent(_: std.mem.Allocator, _: []const u8) !Agent {
    return error.FeatureDisabled;
}
