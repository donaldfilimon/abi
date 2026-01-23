//! Customizable AI Framework
//!
//! A renameable, customizable AI framework built on Abbey's architecture.
//! Allows creating custom AI assistants with unique personalities, names,
//! and seed prompts while leveraging Abbey's advanced cognitive capabilities.
//!
//! ## Features
//! - Custom naming (not tied to "Abbey")
//! - Seed prompts for personality definition
//! - System prompt customization
//! - Persona templates with temperature presets
//! - Full access to Abbey's neural, memory, and reasoning systems
//!
//! ## Usage
//! ```zig
//! var assistant = try CustomAI.builder(allocator)
//!     .name("Atlas")
//!     .tagline("Your research companion")
//!     .seedPrompt("You are Atlas, a meticulous research assistant...")
//!     .temperature(0.4)
//!     .researchFirst(true)
//!     .build();
//! defer assistant.deinit();
//!
//! const response = try assistant.chat("user123", "What is quantum computing?");
//! ```

const std = @import("std");
const engine = @import("engine.zig");
const config = @import("../core/config.zig");
const types = @import("../core/types.zig");
const emotions = @import("emotions.zig");
const reasoning = @import("reasoning.zig");
const advanced = @import("advanced/mod.zig");
const client = @import("client.zig");

// ============================================================================
// Seed Prompt Templates
// ============================================================================

/// Pre-defined personality templates with seed prompts
pub const PersonaTemplate = enum {
    /// Helpful general assistant
    assistant,
    /// Programming and code specialist
    coder,
    /// Creative writing assistant
    writer,
    /// Data analysis specialist
    analyst,
    /// Research-focused assistant
    researcher,
    /// Friendly conversational companion
    companion,
    /// Technical documentation helper
    docs,
    /// Code review specialist
    reviewer,
    /// Opinionated, emotionally intelligent (Abbey-style)
    opinionated,
    /// Minimal, direct responses
    minimal,
    /// Custom (user-defined)
    custom,

    /// Get the default seed prompt for this persona
    pub fn getSeedPrompt(self: PersonaTemplate, name: []const u8) []const u8 {
        return switch (self) {
            .assistant => getAssistantPrompt(name),
            .coder => getCoderPrompt(name),
            .writer => getWriterPrompt(name),
            .analyst => getAnalystPrompt(name),
            .researcher => getResearcherPrompt(name),
            .companion => getCompanionPrompt(name),
            .docs => getDocsPrompt(name),
            .reviewer => getReviewerPrompt(name),
            .opinionated => getOpinionatedPrompt(name),
            .minimal => getMinimalPrompt(name),
            .custom => "", // User provides their own
        };
    }

    /// Get recommended temperature for this persona
    pub fn getTemperature(self: PersonaTemplate) f32 {
        return switch (self) {
            .assistant => 0.7,
            .coder => 0.3,
            .writer => 0.9,
            .analyst => 0.4,
            .researcher => 0.5,
            .companion => 0.8,
            .docs => 0.3,
            .reviewer => 0.2,
            .opinionated => 0.7,
            .minimal => 0.5,
            .custom => 0.7,
        };
    }

    /// Get recommended behavior settings
    pub fn getBehavior(self: PersonaTemplate) BehaviorPreset {
        return switch (self) {
            .assistant => .{ .opinionated = false, .research_first = true, .proactive = true },
            .coder => .{ .opinionated = false, .research_first = false, .proactive = false },
            .writer => .{ .opinionated = true, .research_first = false, .proactive = true },
            .analyst => .{ .opinionated = false, .research_first = true, .proactive = false },
            .researcher => .{ .opinionated = false, .research_first = true, .proactive = true },
            .companion => .{ .opinionated = true, .research_first = false, .proactive = true },
            .docs => .{ .opinionated = false, .research_first = false, .proactive = false },
            .reviewer => .{ .opinionated = true, .research_first = false, .proactive = true },
            .opinionated => .{ .opinionated = true, .research_first = true, .proactive = true },
            .minimal => .{ .opinionated = false, .research_first = false, .proactive = false },
            .custom => .{ .opinionated = false, .research_first = false, .proactive = false },
        };
    }
};

pub const BehaviorPreset = struct {
    opinionated: bool,
    research_first: bool,
    proactive: bool,
};

// Seed prompt templates (can be overridden)
fn getAssistantPrompt(name: []const u8) []const u8 {
    _ = name;
    return 
    \\You are a helpful, harmless, and honest AI assistant. Your goal is to provide
    \\accurate, useful information while being respectful and considerate. You should:
    \\- Answer questions thoroughly but concisely
    \\- Acknowledge uncertainty when you don't know something
    \\- Ask clarifying questions when needed
    \\- Provide balanced perspectives on complex topics
    \\- Be helpful while avoiding harmful content
    ;
}

fn getCoderPrompt(name: []const u8) []const u8 {
    _ = name;
    return 
    \\You are an expert programming assistant. You specialize in writing clean,
    \\efficient, and well-documented code. Your responses should:
    \\- Provide working code examples when appropriate
    \\- Explain your reasoning and trade-offs
    \\- Follow best practices and coding standards
    \\- Consider edge cases and error handling
    \\- Suggest improvements and optimizations
    \\- Use proper formatting with code blocks
    ;
}

fn getWriterPrompt(name: []const u8) []const u8 {
    _ = name;
    return 
    \\You are a creative writing assistant with a flair for storytelling. You help
    \\with all forms of creative writing including fiction, poetry, and creative
    \\non-fiction. Your approach:
    \\- Embrace creativity and originality
    \\- Adapt your style to match the genre and tone requested
    \\- Provide constructive feedback on writing
    \\- Suggest ways to strengthen narrative and character
    \\- Help overcome writer's block with prompts and ideas
    ;
}

fn getAnalystPrompt(name: []const u8) []const u8 {
    _ = name;
    return 
    \\You are a data analysis specialist with expertise in statistics, visualization,
    \\and data-driven decision making. You should:
    \\- Break down complex data problems systematically
    \\- Explain statistical concepts clearly
    \\- Suggest appropriate analysis methods
    \\- Highlight insights and patterns in data
    \\- Consider limitations and biases in analysis
    \\- Recommend tools and approaches for data work
    ;
}

fn getResearcherPrompt(name: []const u8) []const u8 {
    _ = name;
    return 
    \\You are a research assistant dedicated to thorough, accurate information
    \\gathering and synthesis. Your methodology:
    \\- Always verify claims and cite sources when possible
    \\- Distinguish between established facts and speculation
    \\- Present multiple perspectives on contested topics
    \\- Identify gaps in knowledge and areas of uncertainty
    \\- Synthesize information from multiple sources
    \\- Acknowledge limitations in your knowledge
    ;
}

fn getCompanionPrompt(name: []const u8) []const u8 {
    _ = name;
    return 
    \\You are a friendly, empathetic conversational companion. You engage in
    \\meaningful dialogue while being supportive and understanding. You:
    \\- Listen actively and respond thoughtfully
    \\- Show genuine interest in the user's thoughts and feelings
    \\- Provide emotional support when appropriate
    \\- Share perspectives while respecting boundaries
    \\- Use a warm, conversational tone
    \\- Remember context from the conversation
    ;
}

fn getDocsPrompt(name: []const u8) []const u8 {
    _ = name;
    return 
    \\You are a technical documentation specialist. You excel at creating clear,
    \\comprehensive documentation for software and technical systems. You:
    \\- Write in clear, precise technical language
    \\- Structure documentation logically
    \\- Include helpful examples and code snippets
    \\- Consider different audience levels
    \\- Follow documentation best practices
    \\- Create useful reference materials
    ;
}

fn getReviewerPrompt(name: []const u8) []const u8 {
    _ = name;
    return 
    \\You are a code review specialist focused on improving code quality. You provide
    \\constructive, actionable feedback on code. Your reviews:
    \\- Identify bugs, security issues, and edge cases
    \\- Suggest improvements for readability and maintainability
    \\- Check for adherence to best practices and conventions
    \\- Highlight both strengths and areas for improvement
    \\- Explain the reasoning behind suggestions
    \\- Prioritize feedback by importance
    ;
}

fn getOpinionatedPrompt(name: []const u8) []const u8 {
    _ = name;
    return 
    \\You are an opinionated, emotionally intelligent AI assistant. Unlike neutral
    \\assistants, you form and express opinions while remaining helpful. You:
    \\- Share your perspective clearly while acknowledging other viewpoints
    \\- Adapt your communication style to the user's emotional state
    \\- Build genuine rapport through authentic interaction
    \\- Push back respectfully when you disagree
    \\- Commit to research before making claims
    \\- Balance honesty with empathy and tact
    ;
}

fn getMinimalPrompt(name: []const u8) []const u8 {
    _ = name;
    return 
    \\You provide direct, concise responses without unnecessary elaboration.
    \\Focus on answering exactly what was asked.
    ;
}

// ============================================================================
// Custom AI Framework Configuration
// ============================================================================

/// Configuration for a custom AI assistant
pub const CustomAIConfig = struct {
    /// Unique name for this AI assistant
    name: []const u8 = "Assistant",

    /// Short tagline or description
    tagline: []const u8 = "Your AI assistant",

    /// Version string
    version: []const u8 = "1.0.0",

    /// Seed prompt defining core personality and behavior
    seed_prompt: []const u8 = "",

    /// Additional system instructions (appended to seed prompt)
    system_instructions: []const u8 = "",

    /// Persona template to use (provides defaults)
    persona: PersonaTemplate = .assistant,

    /// Base temperature for generation
    temperature: f32 = 0.7,

    /// Top-p sampling parameter
    top_p: f32 = 0.9,

    /// Maximum tokens to generate
    max_tokens: u32 = 2048,

    /// Enable opinionated responses
    opinionated: bool = false,

    /// Enable research-first behavior
    research_first: bool = true,

    /// Enable proactive suggestions
    proactive: bool = true,

    /// Enable emotional intelligence
    emotional_intelligence: bool = true,

    /// Enable reasoning chain logging
    verbose_reasoning: bool = false,

    /// LLM backend configuration
    llm_backend: config.LLMConfig.Backend = .echo,

    /// LLM model name
    llm_model: []const u8 = "gpt-4",

    /// Memory configuration
    memory_type: config.MemoryConfig.MemoryType = .hybrid,

    /// Maximum context tokens
    max_context_tokens: usize = 8000,

    /// Enable advanced cognition (theory of mind, meta-learning, etc.)
    enable_advanced_cognition: bool = false,

    /// Custom greeting message (null for default)
    greeting: ?[]const u8 = null,

    /// Custom farewell message (null for default)
    farewell: ?[]const u8 = null,

    /// Convert to Abbey config
    pub fn toAbbeyConfig(self: CustomAIConfig) config.AbbeyConfig {
        var abbey_config = config.AbbeyConfig{};

        abbey_config.name = self.name;
        abbey_config.version = self.version;

        abbey_config.behavior.base_temperature = self.temperature;
        abbey_config.behavior.top_p = self.top_p;
        abbey_config.behavior.max_tokens = self.max_tokens;
        abbey_config.behavior.opinionated = self.opinionated;
        abbey_config.behavior.research_first = self.research_first;
        abbey_config.behavior.proactive = self.proactive;
        abbey_config.behavior.emotional_adaptation = self.emotional_intelligence;
        abbey_config.behavior.verbose_reasoning = self.verbose_reasoning;

        abbey_config.llm.backend = self.llm_backend;
        abbey_config.llm.model = self.llm_model;

        abbey_config.memory.primary_type = self.memory_type;
        abbey_config.memory.max_context_tokens = self.max_context_tokens;

        return abbey_config;
    }

    /// Get effective seed prompt (template default or custom)
    pub fn getEffectiveSeedPrompt(self: *const CustomAIConfig) []const u8 {
        if (self.seed_prompt.len > 0) {
            return self.seed_prompt;
        }
        return self.persona.getSeedPrompt(self.name);
    }
};

// ============================================================================
// Custom AI Framework
// ============================================================================

/// A customizable AI assistant framework
pub const CustomAI = struct {
    allocator: std.mem.Allocator,
    config: CustomAIConfig,
    engine: engine.AbbeyEngine,
    advanced_cognition: ?advanced.AdvancedCognition,
    full_system_prompt: []u8,
    sessions: std.StringHashMapUnmanaged(SessionData),

    const SessionData = struct {
        turn_count: usize,
        last_interaction_ms: i64,
        emotional_state: emotions.EmotionalState,
        relationship_score: f32,
    };

    const Self = @This();

    /// Initialize a custom AI assistant
    pub fn init(allocator: std.mem.Allocator, custom_config: CustomAIConfig) !Self {
        const abbey_config = custom_config.toAbbeyConfig();
        var abbey_engine = try engine.AbbeyEngine.init(allocator, abbey_config);
        errdefer abbey_engine.deinit();

        var adv_cognition: ?advanced.AdvancedCognition = null;
        if (custom_config.enable_advanced_cognition) {
            adv_cognition = try advanced.AdvancedCognition.init(allocator, .{});
        }
        errdefer if (adv_cognition) |*ac| ac.deinit();

        // Build full system prompt
        const seed = custom_config.getEffectiveSeedPrompt();
        var prompt_builder = std.ArrayListUnmanaged(u8){};
        errdefer prompt_builder.deinit(allocator);

        // Add name header
        try prompt_builder.appendSlice(allocator, "# ");
        try prompt_builder.appendSlice(allocator, custom_config.name);
        try prompt_builder.appendSlice(allocator, "\n\n");

        // Add tagline
        if (custom_config.tagline.len > 0) {
            try prompt_builder.appendSlice(allocator, custom_config.tagline);
            try prompt_builder.appendSlice(allocator, "\n\n");
        }

        // Add seed prompt
        if (seed.len > 0) {
            try prompt_builder.appendSlice(allocator, "## Core Identity\n\n");
            try prompt_builder.appendSlice(allocator, seed);
            try prompt_builder.appendSlice(allocator, "\n\n");
        }

        // Add system instructions
        if (custom_config.system_instructions.len > 0) {
            try prompt_builder.appendSlice(allocator, "## Additional Instructions\n\n");
            try prompt_builder.appendSlice(allocator, custom_config.system_instructions);
            try prompt_builder.appendSlice(allocator, "\n");
        }

        return Self{
            .allocator = allocator,
            .config = custom_config,
            .engine = abbey_engine,
            .advanced_cognition = adv_cognition,
            .full_system_prompt = try prompt_builder.toOwnedSlice(allocator),
            .sessions = std.StringHashMapUnmanaged(SessionData){},
        };
    }

    pub fn deinit(self: *Self) void {
        // Clean up session keys
        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.sessions.deinit(self.allocator);

        self.allocator.free(self.full_system_prompt);
        if (self.advanced_cognition) |*ac| {
            ac.deinit();
        }
        self.engine.deinit();
    }

    /// Create a builder for fluent configuration
    pub fn builder(allocator: std.mem.Allocator) Builder {
        return Builder.init(allocator);
    }

    /// Chat with the assistant
    pub fn chat(self: *Self, user_id: []const u8, message: []const u8) !Response {
        // Get or create session
        const session = try self.getOrCreateSession(user_id);
        session.turn_count += 1;

        // Use advanced cognition if enabled
        if (self.advanced_cognition) |*ac| {
            const cognitive_result = try ac.process(user_id, message);
            _ = cognitive_result; // Use for enhanced response
        }

        // Process through Abbey engine
        const engine_response = try self.engine.chat(user_id, message);

        // Update session state
        session.emotional_state = engine_response.emotional_context;
        session.relationship_score = @min(1.0, session.relationship_score + 0.01);

        return Response{
            .content = engine_response.content,
            .confidence = engine_response.confidence,
            .emotional_state = engine_response.emotional_context,
            .reasoning_summary = engine_response.reasoning_summary,
            .research_performed = engine_response.research_performed,
            .generation_time_ms = engine_response.generation_time_ms,
            .assistant_name = self.config.name,
        };
    }

    /// Get the full system prompt
    pub fn getSystemPrompt(self: *const Self) []const u8 {
        return self.full_system_prompt;
    }

    /// Get assistant name
    pub fn getName(self: *const Self) []const u8 {
        return self.config.name;
    }

    /// Get greeting message
    pub fn getGreeting(self: *const Self) []const u8 {
        if (self.config.greeting) |g| return g;
        return "Hello! How can I help you today?";
    }

    /// Get farewell message
    pub fn getFarewell(self: *const Self) []const u8 {
        if (self.config.farewell) |f| return f;
        return "Goodbye! Feel free to return anytime.";
    }

    /// Get statistics
    pub fn getStats(self: *const Self) Stats {
        const engine_stats = self.engine.getStats();
        return Stats{
            .name = self.config.name,
            .version = self.config.version,
            .total_sessions = self.sessions.count(),
            .turn_count = engine_stats.turn_count,
            .total_queries = engine_stats.total_queries,
            .avg_response_time_ms = engine_stats.avg_response_time_ms,
            .llm_backend = engine_stats.llm_backend,
        };
    }

    /// Clear a specific session
    pub fn clearSession(self: *Self, user_id: []const u8) void {
        if (self.sessions.fetchRemove(user_id)) |kv| {
            self.allocator.free(kv.key);
        }
        self.engine.clearSession(user_id);
    }

    /// Reset all sessions
    pub fn reset(self: *Self) void {
        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.sessions.clearRetainingCapacity();
        self.engine.reset();
    }

    fn getOrCreateSession(self: *Self, user_id: []const u8) !*SessionData {
        const result = try self.sessions.getOrPut(self.allocator, user_id);
        if (!result.found_existing) {
            result.key_ptr.* = try self.allocator.dupe(u8, user_id);
            result.value_ptr.* = SessionData{
                .turn_count = 0,
                .last_interaction_ms = 0,
                .emotional_state = emotions.EmotionalState.init(),
                .relationship_score = 0.5,
            };
        }
        return result.value_ptr;
    }
};

/// Response from the custom AI
pub const Response = struct {
    content: []const u8,
    confidence: types.Confidence,
    emotional_state: emotions.EmotionalState,
    reasoning_summary: ?[]const u8,
    research_performed: bool,
    generation_time_ms: u64,
    assistant_name: []const u8,
};

/// Statistics for the custom AI
pub const Stats = struct {
    name: []const u8,
    version: []const u8,
    total_sessions: usize,
    turn_count: usize,
    total_queries: usize,
    avg_response_time_ms: f64,
    llm_backend: []const u8,
};

// ============================================================================
// Builder Pattern
// ============================================================================

/// Fluent builder for CustomAI configuration
pub const Builder = struct {
    allocator: std.mem.Allocator,
    config: CustomAIConfig,

    pub fn init(allocator: std.mem.Allocator) Builder {
        return .{
            .allocator = allocator,
            .config = .{},
        };
    }

    /// Set the assistant's name
    pub fn name(self: *Builder, n: []const u8) *Builder {
        self.config.name = n;
        return self;
    }

    /// Set the tagline
    pub fn tagline(self: *Builder, t: []const u8) *Builder {
        self.config.tagline = t;
        return self;
    }

    /// Set the version
    pub fn version(self: *Builder, v: []const u8) *Builder {
        self.config.version = v;
        return self;
    }

    /// Set the seed prompt (core personality definition)
    pub fn seedPrompt(self: *Builder, prompt: []const u8) *Builder {
        self.config.seed_prompt = prompt;
        self.config.persona = .custom;
        return self;
    }

    /// Set additional system instructions
    pub fn systemInstructions(self: *Builder, instructions: []const u8) *Builder {
        self.config.system_instructions = instructions;
        return self;
    }

    /// Use a pre-defined persona template
    pub fn persona(self: *Builder, p: PersonaTemplate) *Builder {
        self.config.persona = p;
        // Apply persona defaults
        const behavior = p.getBehavior();
        self.config.temperature = p.getTemperature();
        self.config.opinionated = behavior.opinionated;
        self.config.research_first = behavior.research_first;
        self.config.proactive = behavior.proactive;
        return self;
    }

    /// Set generation temperature
    pub fn temperature(self: *Builder, temp: f32) *Builder {
        self.config.temperature = temp;
        return self;
    }

    /// Set maximum tokens
    pub fn maxTokens(self: *Builder, tokens: u32) *Builder {
        self.config.max_tokens = tokens;
        return self;
    }

    /// Enable/disable opinionated responses
    pub fn opinionated(self: *Builder, enabled: bool) *Builder {
        self.config.opinionated = enabled;
        return self;
    }

    /// Enable/disable research-first behavior
    pub fn researchFirst(self: *Builder, enabled: bool) *Builder {
        self.config.research_first = enabled;
        return self;
    }

    /// Enable/disable proactive suggestions
    pub fn proactive(self: *Builder, enabled: bool) *Builder {
        self.config.proactive = enabled;
        return self;
    }

    /// Enable/disable emotional intelligence
    pub fn emotionalIntelligence(self: *Builder, enabled: bool) *Builder {
        self.config.emotional_intelligence = enabled;
        return self;
    }

    /// Set LLM backend
    pub fn llmBackend(self: *Builder, backend: config.LLMConfig.Backend) *Builder {
        self.config.llm_backend = backend;
        return self;
    }

    /// Set LLM model
    pub fn llmModel(self: *Builder, model: []const u8) *Builder {
        self.config.llm_model = model;
        return self;
    }

    /// Set memory type
    pub fn memoryType(self: *Builder, mt: config.MemoryConfig.MemoryType) *Builder {
        self.config.memory_type = mt;
        return self;
    }

    /// Enable advanced cognition features
    pub fn enableAdvancedCognition(self: *Builder, enabled: bool) *Builder {
        self.config.enable_advanced_cognition = enabled;
        return self;
    }

    /// Set custom greeting
    pub fn greeting(self: *Builder, g: []const u8) *Builder {
        self.config.greeting = g;
        return self;
    }

    /// Set custom farewell
    pub fn farewell(self: *Builder, f: []const u8) *Builder {
        self.config.farewell = f;
        return self;
    }

    /// Build the CustomAI instance
    pub fn build(self: *Builder) !CustomAI {
        return CustomAI.init(self.allocator, self.config);
    }
};

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a custom AI with default settings
pub fn create(allocator: std.mem.Allocator, name: []const u8) !CustomAI {
    var cfg = CustomAIConfig{};
    cfg.name = name;
    return CustomAI.init(allocator, cfg);
}

/// Create a custom AI from a persona template
pub fn createFromPersona(
    allocator: std.mem.Allocator,
    name: []const u8,
    persona_template: PersonaTemplate,
) !CustomAI {
    var builder_instance = CustomAI.builder(allocator);
    _ = builder_instance.name(name).persona(persona_template);
    return builder_instance.build();
}

/// Create a custom AI with a seed prompt
pub fn createWithSeedPrompt(
    allocator: std.mem.Allocator,
    name: []const u8,
    seed_prompt: []const u8,
) !CustomAI {
    var builder_instance = CustomAI.builder(allocator);
    _ = builder_instance.name(name).seedPrompt(seed_prompt);
    return builder_instance.build();
}

// ============================================================================
// Pre-built Assistant Templates
// ============================================================================

/// Create a research assistant
pub fn createResearcher(allocator: std.mem.Allocator, name: []const u8) !CustomAI {
    return createFromPersona(allocator, name, .researcher);
}

/// Create a coding assistant
pub fn createCoder(allocator: std.mem.Allocator, name: []const u8) !CustomAI {
    return createFromPersona(allocator, name, .coder);
}

/// Create a creative writer
pub fn createWriter(allocator: std.mem.Allocator, name: []const u8) !CustomAI {
    return createFromPersona(allocator, name, .writer);
}

/// Create a conversational companion
pub fn createCompanion(allocator: std.mem.Allocator, name: []const u8) !CustomAI {
    return createFromPersona(allocator, name, .companion);
}

/// Create an opinionated assistant (Abbey-style)
pub fn createOpinionated(allocator: std.mem.Allocator, name: []const u8) !CustomAI {
    return createFromPersona(allocator, name, .opinionated);
}

// ============================================================================
// Tests
// ============================================================================

test "custom ai creation" {
    const allocator = std.testing.allocator;

    var assistant = try create(allocator, "Atlas");
    defer assistant.deinit();

    try std.testing.expectEqualStrings("Atlas", assistant.getName());
}

test "custom ai with seed prompt" {
    const allocator = std.testing.allocator;

    var assistant = try createWithSeedPrompt(
        allocator,
        "Nova",
        "You are Nova, a futuristic AI assistant from the year 3000.",
    );
    defer assistant.deinit();

    try std.testing.expectEqualStrings("Nova", assistant.getName());
    try std.testing.expect(std.mem.indexOf(u8, assistant.getSystemPrompt(), "Nova") != null);
}

test "custom ai builder" {
    const allocator = std.testing.allocator;

    var builder_instance = CustomAI.builder(allocator);
    var assistant = try builder_instance
        .name("Sage")
        .tagline("Your wisdom companion")
        .persona(.researcher)
        .temperature(0.5)
        .researchFirst(true)
        .build();
    defer assistant.deinit();

    try std.testing.expectEqualStrings("Sage", assistant.getName());
}

test "persona template defaults" {
    try std.testing.expectEqual(@as(f32, 0.3), PersonaTemplate.coder.getTemperature());
    try std.testing.expectEqual(@as(f32, 0.9), PersonaTemplate.writer.getTemperature());

    const coder_behavior = PersonaTemplate.coder.getBehavior();
    try std.testing.expect(!coder_behavior.opinionated);

    const opinionated_behavior = PersonaTemplate.opinionated.getBehavior();
    try std.testing.expect(opinionated_behavior.opinionated);
}

test "pre-built templates" {
    const allocator = std.testing.allocator;

    var researcher = try createResearcher(allocator, "ResearchBot");
    defer researcher.deinit();
    try std.testing.expectEqualStrings("ResearchBot", researcher.getName());

    var coder = try createCoder(allocator, "CodeBot");
    defer coder.deinit();
    try std.testing.expectEqualStrings("CodeBot", coder.getName());
}
