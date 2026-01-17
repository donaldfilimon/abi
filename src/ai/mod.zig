//! AI Module
//!
//! Modular AI capabilities organized as independent sub-features:
//!
//! - **core**: Shared types, interfaces, and utilities (always available when AI enabled)
//! - **llm**: Local LLM inference (GGUF, transformer models)
//! - **embeddings**: Vector embeddings generation
//! - **agents**: AI agent runtime and tools
//! - **training**: Model training pipelines
//!
//! Each sub-feature can be independently enabled/disabled.
//!
//! ## Usage
//!
//! ```zig
//! const ai = @import("ai/mod.zig");
//!
//! // Initialize AI context
//! var ctx = try ai.Context.init(allocator, .{
//!     .llm = .{ .model_path = "./models/llama.gguf" },
//! });
//! defer ctx.deinit();
//!
//! // Use LLM
//! const response = try ctx.getLlm().generate("Hello, world!");
//! ```

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

// Re-export from features/ai for gradual migration
const features_ai = @import("../features/ai/mod.zig");

// Direct module re-exports for backward compatibility
pub const agent = features_ai.agent;
pub const model_registry = features_ai.model_registry;

// ============================================================================
// Sub-modules (conditionally compiled)
// ============================================================================

/// Core AI types and utilities (always available when AI enabled)
pub const core = struct {
    // Re-export common types
    pub const ModelInfo = features_ai.ModelInfo;
    pub const ModelRegistry = features_ai.ModelRegistry;
};

/// LLM inference module
pub const llm = if (build_options.enable_llm)
    @import("llm/mod.zig")
else
    @import("llm/stub.zig");

/// Embeddings generation module
pub const embeddings = if (build_options.enable_ai)
    @import("embeddings/mod.zig")
else
    @import("embeddings/stub.zig");

/// Agent runtime module
pub const agents = if (build_options.enable_ai)
    @import("agents/mod.zig")
else
    @import("agents/stub.zig");

/// Training pipelines module
pub const training = if (build_options.enable_ai)
    @import("training/mod.zig")
else
    @import("training/stub.zig");

// ============================================================================
// Re-exports from existing AI module (for compatibility)
// ============================================================================

pub const Agent = features_ai.Agent;
pub const ModelRegistry = features_ai.ModelRegistry;
pub const ModelInfo = features_ai.ModelInfo;
pub const TrainingConfig = features_ai.TrainingConfig;
pub const TrainingReport = features_ai.TrainingReport;
pub const TrainingResult = features_ai.TrainingResult;
pub const TrainError = features_ai.TrainError;
pub const OptimizerType = features_ai.OptimizerType;
pub const LearningRateSchedule = features_ai.LearningRateSchedule;
pub const CheckpointStore = features_ai.CheckpointStore;
pub const Checkpoint = features_ai.Checkpoint;
pub const LlmTrainingConfig = features_ai.LlmTrainingConfig;
pub const trainable_model = features_ai.trainable_model;
pub const TrainableModel = features_ai.TrainableModel;
pub const TrainableModelConfig = features_ai.TrainableModelConfig;
pub const LlamaTrainer = features_ai.LlamaTrainer;
pub const loadCheckpoint = features_ai.loadCheckpoint;
pub const saveCheckpoint = features_ai.saveCheckpoint;

// Tools
pub const Tool = features_ai.Tool;
pub const ToolResult = features_ai.ToolResult;
pub const ToolRegistry = features_ai.ToolRegistry;
pub const TaskTool = features_ai.TaskTool;
pub const Subagent = features_ai.Subagent;
pub const DiscordTools = features_ai.DiscordTools;
pub const registerDiscordTools = features_ai.registerDiscordTools;

// Transformer
pub const transformer = features_ai.transformer;
pub const TransformerConfig = transformer.TransformerConfig;
pub const TransformerModel = transformer.TransformerModel;

// Streaming
pub const streaming = features_ai.streaming;
pub const StreamingGenerator = streaming.StreamingGenerator;
pub const StreamToken = streaming.StreamToken;
pub const StreamState = streaming.StreamState;
pub const GenerationConfig = streaming.GenerationConfig;

// LLM Engine
pub const LlmEngine = features_ai.LlmEngine;
pub const LlmModel = features_ai.LlmModel;
pub const LlmConfig = features_ai.LlmConfig;
pub const GgufFile = features_ai.GgufFile;
pub const BpeTokenizer = features_ai.BpeTokenizer;

// Prompts
pub const prompts = features_ai.prompts;
pub const PromptBuilder = features_ai.PromptBuilder;
pub const Persona = features_ai.Persona;
pub const PersonaType = features_ai.PersonaType;
pub const PromptFormat = features_ai.PromptFormat;

// Abbey
pub const abbey = features_ai.abbey;
pub const Abbey = features_ai.Abbey;
pub const AbbeyConfig = features_ai.AbbeyConfig;
pub const AbbeyResponse = features_ai.AbbeyResponse;
pub const AbbeyStats = features_ai.AbbeyStats;
pub const ReasoningChain = features_ai.ReasoningChain;
pub const ReasoningStep = features_ai.ReasoningStep;
pub const Confidence = features_ai.Confidence;
pub const ConfidenceLevel = features_ai.ConfidenceLevel;
pub const EmotionalState = features_ai.EmotionalState;
pub const EmotionType = features_ai.EmotionType;
pub const ConversationContext = features_ai.ConversationContext;
pub const TopicTracker = features_ai.TopicTracker;

// Explore
pub const explore = features_ai.explore;
pub const ExploreAgent = features_ai.ExploreAgent;
pub const ExploreConfig = features_ai.ExploreConfig;
pub const ExploreLevel = features_ai.ExploreLevel;
pub const ExploreResult = features_ai.ExploreResult;
pub const Match = features_ai.Match;
pub const ExplorationStats = features_ai.ExplorationStats;
pub const QueryIntent = features_ai.QueryIntent;
pub const ParsedQuery = features_ai.ParsedQuery;
pub const QueryUnderstanding = features_ai.QueryUnderstanding;

// Memory
pub const memory = features_ai.memory;

// Federated
pub const federated = features_ai.federated;

// RAG
pub const rag = features_ai.rag;

// Templates
pub const templates = features_ai.templates;

// Eval
pub const eval = features_ai.eval;

// ============================================================================
// Errors
// ============================================================================

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

// ============================================================================
// Context - New unified interface for Framework integration
// ============================================================================

/// AI context for Framework integration.
/// Manages AI sub-features based on configuration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.AiConfig,

    // Sub-feature contexts (null if disabled)
    llm_ctx: ?*llm.Context = null,
    embeddings_ctx: ?*embeddings.Context = null,
    agents_ctx: ?*agents.Context = null,
    training_ctx: ?*training.Context = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.AiConfig) !*Context {
        if (!isEnabled()) return error.AiDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };

        // Initialize enabled sub-features
        errdefer ctx.deinitSubFeatures();

        if (cfg.llm) |llm_cfg| {
            ctx.llm_ctx = try llm.Context.init(allocator, llm_cfg);
        }

        if (cfg.embeddings) |emb_cfg| {
            ctx.embeddings_ctx = try embeddings.Context.init(allocator, emb_cfg);
        }

        if (cfg.agents) |agent_cfg| {
            ctx.agents_ctx = try agents.Context.init(allocator, agent_cfg);
        }

        if (cfg.training) |train_cfg| {
            ctx.training_ctx = try training.Context.init(allocator, train_cfg);
        }

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.deinitSubFeatures();
        self.allocator.destroy(self);
    }

    fn deinitSubFeatures(self: *Context) void {
        if (self.training_ctx) |t| {
            t.deinit();
            self.training_ctx = null;
        }
        if (self.agents_ctx) |a| {
            a.deinit();
            self.agents_ctx = null;
        }
        if (self.embeddings_ctx) |e| {
            e.deinit();
            self.embeddings_ctx = null;
        }
        if (self.llm_ctx) |l| {
            l.deinit();
            self.llm_ctx = null;
        }
    }

    /// Get LLM context (returns error if not enabled).
    pub fn getLlm(self: *Context) Error!*llm.Context {
        return self.llm_ctx orelse error.LlmDisabled;
    }

    /// Get embeddings context (returns error if not enabled).
    pub fn getEmbeddings(self: *Context) Error!*embeddings.Context {
        return self.embeddings_ctx orelse error.EmbeddingsDisabled;
    }

    /// Get agents context (returns error if not enabled).
    pub fn getAgents(self: *Context) Error!*agents.Context {
        return self.agents_ctx orelse error.AgentsDisabled;
    }

    /// Get training context (returns error if not enabled).
    pub fn getTraining(self: *Context) Error!*training.Context {
        return self.training_ctx orelse error.TrainingDisabled;
    }

    /// Check if a sub-feature is enabled.
    pub fn isSubFeatureEnabled(self: *Context, feature: SubFeature) bool {
        return switch (feature) {
            .llm => self.llm_ctx != null,
            .embeddings => self.embeddings_ctx != null,
            .agents => self.agents_ctx != null,
            .training => self.training_ctx != null,
        };
    }

    pub const SubFeature = enum {
        llm,
        embeddings,
        agents,
        training,
    };
};

// ============================================================================
// Module-level functions
// ============================================================================

/// Check if AI is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.enable_ai;
}

/// Check if LLM is enabled at compile time.
pub fn isLlmEnabled() bool {
    return build_options.enable_llm;
}

/// Check if AI module is initialized.
pub fn isInitialized() bool {
    return features_ai.isInitialized();
}

/// Initialize the AI module (legacy compatibility).
pub fn init(allocator: std.mem.Allocator) Error!void {
    if (!isEnabled()) return error.AiDisabled;
    features_ai.init(allocator) catch return error.AiDisabled;
}

/// Deinitialize the AI module (legacy compatibility).
pub fn deinit() void {
    features_ai.deinit();
}

// Legacy convenience functions
pub fn createRegistry(allocator: std.mem.Allocator) ModelRegistry {
    return features_ai.createRegistry(allocator);
}

pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingReport {
    return features_ai.train(allocator, config);
}

pub fn trainWithResult(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingResult {
    return features_ai.trainWithResult(allocator, config);
}

pub fn createAgent(allocator: std.mem.Allocator, name: []const u8) !Agent {
    return features_ai.createAgent(allocator, name);
}

pub fn createTransformer(config: TransformerConfig) TransformerModel {
    return features_ai.createTransformer(config);
}

pub fn inferText(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    return features_ai.inferText(allocator, input);
}

pub fn embedText(allocator: std.mem.Allocator, input: []const u8) ![]f32 {
    return features_ai.embedText(allocator, input);
}

pub fn encodeTokens(allocator: std.mem.Allocator, input: []const u8) ![]u32 {
    return features_ai.encodeTokens(allocator, input);
}

pub fn decodeTokens(allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
    return features_ai.decodeTokens(allocator, tokens);
}

// ============================================================================
// Tests
// ============================================================================

test "isEnabled returns build option" {
    try std.testing.expectEqual(build_options.enable_ai, isEnabled());
}

test "isLlmEnabled returns build option" {
    try std.testing.expectEqual(build_options.enable_llm, isLlmEnabled());
}
