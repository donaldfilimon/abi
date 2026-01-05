//! AI feature module with agents, transformers, training, and federated learning.
//!
//! Provides high-level interfaces for AI functionality including agent creation,
//! transformer models, training pipelines, and federated learning coordination.

const std = @import("std");
const build_options = @import("build_options");

pub const agent = @import("agent.zig");
pub const model_registry = @import("model_registry.zig");
pub const training = @import("training/mod.zig");
pub const federated = @import("federated/mod.zig");
pub const transformer = @import("transformer/mod.zig");
pub const streaming = @import("streaming.zig");
pub const tools = @import("tools/mod.zig");
pub const explore = @import("explore/mod.zig");

pub const Agent = agent.Agent;
pub const ModelRegistry = model_registry.ModelRegistry;
pub const ModelInfo = model_registry.ModelInfo;
pub const TrainingConfig = training.TrainingConfig;
pub const TrainingReport = training.TrainingReport;
pub const TrainingResult = training.TrainingResult;
pub const TrainError = training.TrainError;
pub const CheckpointStore = training.CheckpointStore;
pub const Checkpoint = training.Checkpoint;
pub const GradientAccumulator = training.GradientAccumulator;

pub const Tool = tools.Tool;
pub const ToolResult = tools.ToolResult;
pub const ToolRegistry = tools.ToolRegistry;
pub const TaskTool = tools.TaskTool;
pub const Subagent = tools.Subagent;

pub const ExploreAgent = explore.ExploreAgent;
pub const ExploreConfig = explore.ExploreConfig;
pub const ExploreLevel = explore.ExploreLevel;
pub const ExploreResult = explore.ExploreResult;
pub const Match = explore.Match;
pub const ExplorationStats = explore.ExplorationStats;
pub const QueryIntent = explore.QueryIntent;
pub const ParsedQuery = explore.ParsedQuery;
pub const QueryUnderstanding = explore.QueryUnderstanding;

pub const AiError = error{
    AiDisabled,
};

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    if (!isEnabled()) return AiError.AiDisabled;
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return build_options.enable_ai;
}

pub fn isInitialized() bool {
    return initialized;
}

pub fn createRegistry(allocator: std.mem.Allocator) ModelRegistry {
    return ModelRegistry.init(allocator);
}

pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingReport {
    return training.trainAndReport(allocator, config);
}

pub fn trainWithResult(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!TrainingResult {
    return training.trainWithResult(allocator, config);
}

pub fn createAgent(allocator: std.mem.Allocator, name: []const u8) !Agent {
    if (!isEnabled()) return AiError.AiDisabled;
    return agent.Agent.init(allocator, .{ .name = name });
}

pub fn createAgentWithConfig(allocator: std.mem.Allocator, config: agent.AgentConfig) !Agent {
    if (!isEnabled()) return AiError.AiDisabled;
    return agent.Agent.init(allocator, config);
}

pub fn processMessage(
    allocator: std.mem.Allocator,
    name: []const u8,
    message: []const u8,
) ![]u8 {
    if (!isEnabled()) return AiError.AiDisabled;
    var instance = try createAgent(allocator, name);
    defer instance.deinit();
    return instance.process(message, allocator);
}

pub fn createTransformer(config: transformer.TransformerConfig) transformer.TransformerModel {
    return transformer.TransformerModel.init(config);
}

pub fn inferText(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (!isEnabled()) return AiError.AiDisabled;
    var model = transformer.TransformerModel.init(.{});
    return model.infer(allocator, input);
}

pub fn embedText(allocator: std.mem.Allocator, input: []const u8) ![]f32 {
    if (!isEnabled()) return AiError.AiDisabled;
    var model = transformer.TransformerModel.init(.{});
    return model.embed(allocator, input);
}

pub fn encodeTokens(allocator: std.mem.Allocator, input: []const u8) ![]u32 {
    if (!isEnabled()) return AiError.AiDisabled;
    const model = transformer.TransformerModel.init(.{});
    return model.encode(allocator, input);
}

pub fn decodeTokens(allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
    if (!isEnabled()) return AiError.AiDisabled;
    const model = transformer.TransformerModel.init(.{});
    return model.decode(allocator, tokens);
}

test "ai module init gating" {
    if (!isEnabled()) return;
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}

test "ai convenience apis" {
    if (!isEnabled()) return;
    var instance = try createAgent(std.testing.allocator, "test");
    defer instance.deinit();

    const response = try processMessage(std.testing.allocator, "test", "hello");
    defer std.testing.allocator.free(response);
    try std.testing.expect(std.mem.indexOf(u8, response, "Echo") != null);

    const tokens = try encodeTokens(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(tokens);
    try std.testing.expect(tokens.len > 0);

    const embedding = try embedText(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(embedding);
    try std.testing.expect(embedding.len > 0);
}
