//! AI Module - Public API
//!
//! This is the primary entry point for AI functionality in the ABI framework.
//! Import from here for Framework integration and the stable public API.
//!
//! ## Overview
//!
//! The AI module provides modular AI capabilities organized as independent sub-features,
//! each of which can be enabled or disabled independently:
//!
//! | Sub-feature | Description | Build Flag |
//! |-------------|-------------|------------|
//! | **core** | Shared types, interfaces, utilities | Always available |
//! | **llm** | Local LLM inference (GGUF models) | `-Denable-llm` |
//! | **embeddings** | Vector embeddings generation | `-Denable-ai` |
//! | **agents** | AI agent runtime and tools | `-Denable-ai` |
//! | **training** | Model training pipelines | `-Denable-ai` |
//! | **personas** | Multi-persona AI assistant | `-Denable-ai` |
//! | **vision** | Image processing and analysis | `-Denable-vision` |
//!
//! ## Quick Start
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Initialize framework with AI enabled
//! var fw = try abi.Framework.init(allocator, .{
//!     .ai = .{
//!         .llm = .{ .model_path = "./models/llama-7b.gguf" },
//!         .embeddings = .{ .dimension = 768 },
//!     },
//! });
//! defer fw.deinit();
//!
//! // Access AI context
//! const ai_ctx = try fw.getAi();
//!
//! // Use LLM
//! const llm = try ai_ctx.getLlm();
//! // ... perform inference ...
//! ```
//!
//! ## Standalone Usage
//!
//! ```zig
//! const ai = abi.ai;
//!
//! // Initialize AI context directly
//! var ctx = try ai.Context.init(allocator, .{
//!     .llm = .{ .model_path = "./models/llama.gguf" },
//! });
//! defer ctx.deinit();
//!
//! // Check which sub-features are enabled
//! if (ctx.isSubFeatureEnabled(.llm)) {
//!     const llm = try ctx.getLlm();
//!     // ... use LLM ...
//! }
//! ```
//!
//! ## Sub-module Access
//!
//! Access sub-modules directly through the namespace:
//! - `abi.ai.llm` - LLM inference engine
//! - `abi.ai.embeddings` - Embedding generation
//! - `abi.ai.agents` - Agent runtime
//! - `abi.ai.training` - Training pipelines
//! - `abi.ai.personas` - Multi-persona system
//! - `abi.ai.vision` - Vision processing

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../core/config/mod.zig");

// ============================================================================
// Direct module imports (no longer using implementation_mod.zig bridge)
// ============================================================================

// Direct imports for AI submodules
pub const agent = @import("agents/agent.zig");
pub const model_registry = @import("models/registry.zig");
pub const transformer = @import("transformer/mod.zig");
pub const streaming = @import("streaming/mod.zig");
pub const tools = @import("tools/mod.zig");
pub const prompts = @import("prompts/mod.zig");
pub const abbey = @import("abbey/mod.zig");
pub const memory = @import("memory/mod.zig");
pub const federated = @import("federated/mod.zig");
pub const personas = if (build_options.enable_ai) @import("personas/mod.zig") else @import("personas/stub.zig");
pub const rag = if (build_options.enable_ai) @import("rag/mod.zig") else @import("rag/stub.zig");
pub const templates = if (build_options.enable_ai) @import("templates/mod.zig") else @import("templates/stub.zig");
pub const eval = if (build_options.enable_ai) @import("eval/mod.zig") else @import("eval/stub.zig");
pub const explore = if (build_options.enable_explore) @import("explore/mod.zig") else @import("explore/stub.zig");
pub const orchestration = if (build_options.enable_ai) @import("orchestration/mod.zig") else @import("orchestration/stub.zig");
pub const constitution = if (build_options.enable_ai) @import("constitution/mod.zig") else @import("constitution/stub.zig");

// Tool-augmented agent with tool execution loop
pub const tool_agent = @import("tools/tool_agent.zig");

// Codebase self-awareness via indexing and RAG
pub const codebase_index = @import("explore/codebase_index.zig");

// Self-improvement and performance tracking
pub const self_improve = @import("self_improve.zig");

// GPU-aware agent (always available, uses stubs when GPU disabled)
pub const gpu_agent = @import("agents/gpu_agent.zig");

// Model auto-discovery and adaptive configuration
pub const discovery = @import("explore/discovery.zig");

// Model management utilities
pub const models = if (build_options.enable_ai) @import("models/mod.zig") else @import("models/stub.zig");

// ============================================================================
// Sub-modules (conditionally compiled)
// ============================================================================

/// Core AI types and utilities (always available when AI enabled)
pub const core = @import("core/mod.zig");

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

/// AI Database module
pub const database = if (build_options.enable_ai)
    @import("database/mod.zig")
else
    @import("database/stub.zig");

// ---------------------------------------------------------------------------
// Multi‑Agent coordination (requires AI feature)
// ---------------------------------------------------------------------------
pub const multi_agent = if (build_options.enable_ai)
    @import("multi_agent/mod.zig")
else
    @import("multi_agent/stub.zig");

/// Vision/image processing module
pub const vision = if (build_options.enable_vision)
    @import("vision/mod.zig")
else
    @import("vision/stub.zig");

/// Document understanding and processing module
pub const documents = if (build_options.enable_ai)
    @import("documents/mod.zig")
else
    @import("documents/stub.zig");

// NOTE(v0.4.0): Flat compatibility re-exports removed.
// Use canonical sub-module paths instead:
//   abi.ai.agent.Agent          (was abi.ai.Agent)
//   abi.ai.multi_agent.Coordinator (was abi.ai.MultiAgentCoordinator)
//   abi.ai.tools.ToolRegistry   (was abi.ai.ToolRegistry)
//   abi.ai.training.*           (was abi.ai.TrainingConfig, etc.)
//   abi.ai.orchestration.TaskType (was abi.ai.TaskType)
//   abi.ai.streaming.StreamToken (was abi.ai.StreamToken)
//   abi.ai.llm.InferenceConfig  (was abi.ai.LlmConfig)
//   abi.ai.self_improve.SelfImprover (was abi.ai.SelfImprover)
//   abi.ai.database.*           (was abi.ai.WdbxTokenDataset, etc.)

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
// Context - Unified interface for Framework integration
// ============================================================================

/// AI context for Framework integration.
///
/// The Context struct manages all AI sub-features (LLM, embeddings, agents, training,
/// personas) based on the provided configuration. Each sub-feature is independently
/// initialized and can be accessed through type-safe getter methods.
///
/// ## Thread Safety
///
/// The Context itself is not thread-safe. If you need to access AI features from
/// multiple threads, use external synchronization.
///
/// ## Memory Management
///
/// The Context allocates memory for each enabled sub-feature context. All memory
/// is released when `deinit()` is called.
///
/// ## Example
///
/// ```zig
/// var ctx = try ai.Context.init(allocator, .{
///     .llm = .{ .model_path = "./models/llama.gguf" },
///     .embeddings = .{ .dimension = 768 },
/// });
/// defer ctx.deinit();
///
/// // Access sub-features
/// const llm = try ctx.getLlm();
/// const emb = try ctx.getEmbeddings();
/// ```
pub const Context = struct {
    /// Memory allocator for context resources.
    allocator: std.mem.Allocator,
    /// Configuration for AI sub-features.
    config: config_module.AiConfig,

    // Sub-feature contexts (null if disabled)
    /// LLM inference context, or null if not enabled.
    llm_ctx: ?*llm.Context = null,
    /// Embeddings generation context, or null if not enabled.
    embeddings_ctx: ?*embeddings.Context = null,
    /// Agent runtime context, or null if not enabled.
    agents_ctx: ?*agents.Context = null,
    /// Training pipeline context, or null if not enabled.
    training_ctx: ?*training.Context = null,
    /// Multi-persona system context, or null if not enabled.
    personas_ctx: ?*personas.Context = null,

    // Auto-discovery and adaptive configuration
    /// Model discovery system for automatic model detection.
    model_discovery: ?*discovery.ModelDiscovery = null,
    /// Detected system capabilities.
    capabilities: discovery.SystemCapabilities = .{},

    /// Initialize the AI context with the given configuration.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Memory allocator for context resources
    /// - `cfg`: AI configuration specifying which sub-features to enable
    ///
    /// ## Returns
    ///
    /// A pointer to the initialized Context.
    ///
    /// ## Errors
    ///
    /// - `error.AiDisabled`: AI is disabled at compile time
    /// - `error.OutOfMemory`: Memory allocation failed
    /// - Sub-feature specific errors during initialization
    pub fn init(allocator: std.mem.Allocator, cfg: config_module.AiConfig) !*Context {
        if (!isEnabled()) return error.AiDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
            .capabilities = discovery.detectCapabilities(),
        };

        // Initialize model discovery if auto_discover is enabled
        if (cfg.auto_discover) {
            const disc = try allocator.create(discovery.ModelDiscovery);
            disc.* = discovery.ModelDiscovery.init(allocator, .{});
            disc.scanAll() catch |err| {
                std.log.debug("Model discovery scan failed (best effort): {t}", .{err});
            };
            ctx.model_discovery = disc;
        }

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

        if (cfg.personas) |personas_cfg| {
            ctx.personas_ctx = try personas.Context.init(allocator, personas_cfg);
        }

        return ctx;
    }

    /// Deinitialize the context and release all resources.
    ///
    /// This releases memory for all initialized sub-feature contexts and
    /// the model discovery system (if enabled). The Context pointer becomes
    /// invalid after this call.
    pub fn deinit(self: *Context) void {
        self.deinitSubFeatures();
        self.allocator.destroy(self);
    }

    fn deinitSubFeatures(self: *Context) void {
        if (self.model_discovery) |disc| {
            disc.deinit();
            self.allocator.destroy(disc);
            self.model_discovery = null;
        }
        if (self.personas_ctx) |p| {
            p.deinit();
            self.personas_ctx = null;
        }
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

    /// Get personas context (returns error if not enabled).
    pub fn getPersonas(self: *Context) Error!*personas.Context {
        return self.personas_ctx orelse error.AiDisabled;
    }

    /// Check if a sub-feature is enabled.
    pub fn isSubFeatureEnabled(self: *Context, feature: SubFeature) bool {
        return switch (feature) {
            .llm => self.llm_ctx != null,
            .embeddings => self.embeddings_ctx != null,
            .agents => self.agents_ctx != null,
            .training => self.training_ctx != null,
            .personas => self.personas_ctx != null,
        };
    }

    /// Get discovered models (returns empty slice if discovery not enabled).
    pub fn getDiscoveredModels(self: *Context) []discovery.DiscoveredModel {
        if (self.model_discovery) |disc| {
            return disc.getModels();
        }
        return &.{};
    }

    /// Get number of discovered models.
    pub fn discoveredModelCount(self: *Context) usize {
        if (self.model_discovery) |disc| {
            return disc.modelCount();
        }
        return 0;
    }

    /// Find best model matching requirements.
    pub fn findBestModel(self: *Context, requirements: discovery.ModelRequirements) ?*discovery.DiscoveredModel {
        if (self.model_discovery) |disc| {
            return disc.findBestModel(requirements);
        }
        return null;
    }

    /// Generate adaptive configuration for a model.
    pub fn generateAdaptiveConfig(self: *Context, model: *const discovery.DiscoveredModel) discovery.AdaptiveConfig {
        if (self.model_discovery) |disc| {
            return disc.generateConfig(model);
        }
        // Return defaults if no discovery
        return discovery.AdaptiveConfig{
            .num_threads = self.capabilities.recommendedThreads(),
            .batch_size = 1,
        };
    }

    /// Get system capabilities.
    pub fn getCapabilities(self: *const Context) discovery.SystemCapabilities {
        return self.capabilities;
    }

    /// Add a model path to the discovery system.
    /// Use this to register known model files.
    pub fn addModelPath(self: *Context, path: []const u8) !void {
        if (self.model_discovery) |disc| {
            try disc.addModelPath(path);
        }
    }

    /// Add a model path with known file size.
    pub fn addModelWithSize(self: *Context, path: []const u8, size_bytes: u64) !void {
        if (self.model_discovery) |disc| {
            try disc.addModelWithSize(path, size_bytes);
        }
    }

    /// Clear all discovered models.
    pub fn clearDiscoveredModels(self: *Context) void {
        if (self.model_discovery) |disc| {
            for (disc.discovered_models.items) |*model| {
                model.deinit(self.allocator);
            }
            disc.discovered_models.clearRetainingCapacity();
        }
    }

    pub const SubFeature = enum {
        llm,
        embeddings,
        agents,
        training,
        personas,
    };
};

// ============================================================================
// Module state
// ============================================================================

var initialized: bool = false;

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
    return initialized;
}

/// Initialize the AI module (legacy compatibility).
pub fn init(allocator: std.mem.Allocator) Error!void {
    _ = allocator;
    if (!isEnabled()) return error.AiDisabled;
    initialized = true;
}

/// Deinitialize the AI module (legacy compatibility).
pub fn deinit() void {
    initialized = false;
}

/// Legacy compatibility helper for direct agent creation.
pub fn createAgent(allocator: std.mem.Allocator, name: []const u8) !agent.Agent {
    if (!isEnabled()) return error.AiDisabled;
    return agent.Agent.init(allocator, .{ .name = name });
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

test "ai module init gating" {
    if (!isEnabled()) return;
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}

test {
    std.testing.refAllDecls(@This());
}
