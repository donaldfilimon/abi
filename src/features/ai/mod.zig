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
pub const explore = if (build_options.enable_explore) @import("explore/mod.zig") else @import("explore/stub.zig");
pub const llm = if (build_options.enable_llm) @import("llm/mod.zig") else @import("llm/stub.zig");
pub const templates = @import("templates/mod.zig");
pub const embeddings = @import("embeddings/mod.zig");
pub const eval = @import("eval/mod.zig");
pub const memory = @import("memory/mod.zig");
pub const rag = @import("rag/mod.zig");
pub const enhanced_streaming = @import("streaming/mod.zig");

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

// LLM module exports
pub const LlmEngine = llm.Engine;
pub const LlmModel = llm.Model;
pub const LlmConfig = llm.InferenceConfig;
pub const GgufFile = llm.GgufFile;
pub const BpeTokenizer = llm.BpeTokenizer;

// Template module exports
pub const Template = templates.Template;
pub const TemplateRegistry = templates.TemplateRegistry;
pub const BuiltinTemplates = templates.BuiltinTemplates;
pub const renderTemplate = templates.renderTemplate;
pub const formatChatMessage = templates.formatChatMessage;
pub const ChatMessage = templates.ChatMessage;

// Embeddings module exports
pub const Embedder = embeddings.Embedder;
pub const EmbedderConfig = embeddings.EmbedderConfig;
pub const EmbeddingResult = embeddings.EmbeddingResult;
pub const EmbeddingCache = embeddings.EmbeddingCache;
pub const cosineSimilarity = embeddings.cosineSimilarity;
pub const euclideanDistance = embeddings.euclideanDistance;

// Evaluation module exports
pub const Evaluator = eval.Evaluator;
pub const EvalConfig = eval.EvalConfig;
pub const EvaluationResult = eval.EvaluationResult;
pub const EvaluationReport = eval.EvaluationReport;
pub const BleuScore = eval.BleuScore;
pub const RougeScore = eval.RougeScore;
pub const RougeType = eval.RougeType;
pub const PerplexityResult = eval.PerplexityResult;
pub const TokenMetrics = eval.TokenMetrics;
pub const TextStatistics = eval.TextStatistics;
pub const computeBleu = eval.computeBleu;
pub const computeRouge = eval.computeRouge;
pub const computePerplexity = eval.computePerplexity;
pub const computeF1 = eval.computeF1;
pub const computeExactMatch = eval.computeExactMatch;

// Memory module exports
pub const MemoryManager = memory.MemoryManager;
pub const MemoryConfig = memory.MemoryConfig;
pub const MemoryType = memory.MemoryType;
pub const MemoryStats = memory.MemoryStats;
pub const ShortTermMemory = memory.ShortTermMemory;
pub const SlidingWindowMemory = memory.SlidingWindowMemory;
pub const SummarizingMemory = memory.SummarizingMemory;
pub const LongTermMemory = memory.LongTermMemory;
pub const MemoryMessage = memory.Message;
pub const MessageRole = memory.MessageRole;
pub const createMemoryManager = memory.createMemoryManager;

// RAG module exports
pub const RagPipeline = rag.RagPipeline;
pub const RagConfig = rag.RagConfig;
pub const RagResponse = rag.RagResponse;
pub const RagContext = rag.RagContext;
pub const RagDocument = rag.Document;
pub const RagChunk = rag.Chunk;
pub const RagRetriever = rag.Retriever;
pub const RagChunker = rag.Chunker;
pub const createRagPipeline = rag.createPipeline;

// Enhanced streaming module exports
pub const SseEncoder = enhanced_streaming.SseEncoder;
pub const SseDecoder = enhanced_streaming.SseDecoder;
pub const SseEvent = enhanced_streaming.SseEvent;
pub const BackpressureController = enhanced_streaming.BackpressureController;
pub const TokenBuffer = enhanced_streaming.TokenBuffer;
pub const EnhancedStreamingGenerator = enhanced_streaming.EnhancedStreamingGenerator;
pub const StreamConfig = enhanced_streaming.StreamConfig;
pub const createSseStream = enhanced_streaming.createSseStream;

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
