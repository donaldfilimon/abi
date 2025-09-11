//! Unified AI Agent Module
//!
//! Consolidates and enhances AI agent functionality with:
//! - Multiple persona support with intelligent routing
//! - Advanced memory management with SIMD optimization
//! - Performance monitoring and metrics
//! - Thread-safe operations and concurrency control
//! - Configurable backends and capabilities

const std = @import("std");

const core = @import("../core/mod.zig");

const Allocator = std.mem.Allocator;
const FrameworkError = core.FrameworkError;

/// Agent-specific error types
pub const AgentError = error{
    InvalidQuery,
    ApiKeyMissing,
    PersonaNotFound,
    ContextWindowExceeded,
    ModelNotAvailable,
    RateLimitExceeded,
    InvalidConfiguration,
    ResourceExhausted,
    OperationTimeout,
    InvalidStateTransition,
    CapabilityNotEnabled,
    MemoryExhausted,
    ConcurrencyLimitReached,
} || FrameworkError;

/// Agent personas with enhanced characteristics
pub const PersonaType = enum {
    empathetic,
    direct,
    adaptive,
    creative,
    technical,
    solver,
    educator,
    counselor,
    analytical,
    supportive,

    /// Get persona description
    pub fn getDescription(self: PersonaType) []const u8 {
        return switch (self) {
            .empathetic => "empathetic and understanding",
            .direct => "direct and to the point",
            .adaptive => "adaptive to user needs",
            .creative => "creative and imaginative",
            .technical => "technical and precise",
            .solver => "problem-solving focused",
            .educator => "educational and explanatory",
            .counselor => "supportive and guiding",
            .analytical => "analytical and logical",
            .supportive => "supportive and encouraging",
        };
    }

    /// Get persona scoring weights for different query types
    pub fn getScoring(self: PersonaType) PersonaScoring {
        return switch (self) {
            .empathetic => .{ .empathy = 0.9, .technical = 0.3, .creativity = 0.6, .directness = 0.2 },
            .direct => .{ .empathy = 0.2, .technical = 0.7, .creativity = 0.3, .directness = 0.9 },
            .adaptive => .{ .empathy = 0.7, .technical = 0.6, .creativity = 0.7, .directness = 0.6 },
            .creative => .{ .empathy = 0.6, .technical = 0.4, .creativity = 0.9, .directness = 0.5 },
            .technical => .{ .empathy = 0.3, .technical = 0.9, .creativity = 0.4, .directness = 0.8 },
            .solver => .{ .empathy = 0.5, .technical = 0.8, .creativity = 0.7, .directness = 0.7 },
            .educator => .{ .empathy = 0.8, .technical = 0.7, .creativity = 0.6, .directness = 0.6 },
            .counselor => .{ .empathy = 0.9, .technical = 0.4, .creativity = 0.5, .directness = 0.3 },
            .analytical => .{ .empathy = 0.4, .technical = 0.9, .creativity = 0.5, .directness = 0.8 },
            .supportive => .{ .empathy = 0.8, .technical = 0.5, .creativity = 0.6, .directness = 0.4 },
        };
    }
};

/// Persona scoring characteristics
pub const PersonaScoring = struct {
    empathy: f32,
    technical: f32,
    creativity: f32,
    directness: f32,
};

/// Agent state with enhanced state management
pub const AgentState = enum(u8) {
    idle = 0,
    thinking = 1,
    processing = 2,
    responding = 3,
    learning = 4,
    error_state = 5,

    /// Validate state transitions
    pub fn canTransitionTo(from: AgentState, to: AgentState) bool {
        return switch (from) {
            .idle => to == .thinking or to == .processing,
            .thinking => to == .processing or to == .responding or to == .error_state,
            .processing => to == .responding or to == .learning or to == .error_state,
            .responding => to == .idle or to == .learning,
            .learning => to == .idle,
            .error_state => to == .idle,
        };
    }
};

/// Agent capabilities with packed representation
pub const AgentCapabilities = packed struct(u32) {
    text_generation: bool = false,
    code_generation: bool = false,
    image_analysis: bool = false,
    audio_processing: bool = false,
    memory_management: bool = false,
    learning: bool = false,
    reasoning: bool = false,
    planning: bool = false,
    vector_search: bool = false,
    function_calling: bool = false,
    multimodal: bool = false,
    streaming: bool = false,
    _reserved: u20 = 0,

    /// Validate capability dependencies
    pub fn validate(self: AgentCapabilities) bool {
        if (self.vector_search and !self.memory_management) return false;
        if (self.multimodal and !(self.text_generation or self.image_analysis)) return false;
        return true;
    }
};

/// Message role in conversation
pub const MessageRole = enum {
    user,
    assistant,
    system,
};

/// Conversation message with metadata
pub const Message = struct {
    role: MessageRole,
    content: []const u8,
    timestamp: i64,
    importance: f32 = 0.5,
    persona_used: ?PersonaType = null,

    pub fn init(allocator: Allocator, role: MessageRole, content: []const u8) !Message {
        return Message{
            .role = role,
            .content = try allocator.dupe(u8, content),
            .timestamp = std.time.microTimestamp(),
        };
    }

    pub fn deinit(self: Message, allocator: Allocator) void {
        allocator.free(self.content);
    }
};

/// Advanced memory entry with vectorized operations
pub const MemoryEntry = struct {
    id: u64,
    timestamp: i64,
    content: []align(64) const u8, // Cache-line aligned
    importance: f32,
    vector_embedding: ?[]f32 = null,
    access_count: u32 = 0,
    last_accessed: i64,
    persona_context: ?PersonaType = null,

    const Self = @This();

    pub fn init(allocator: Allocator, content: []const u8, importance: f32) !Self {
        const aligned_content = try allocator.alignedAlloc(u8, 64, content.len);
        @memcpy(aligned_content, content);

        return Self{
            .id = @as(u64, @intCast(std.time.microTimestamp())),
            .timestamp = std.time.microTimestamp(),
            .content = aligned_content,
            .importance = importance,
            .last_accessed = std.time.microTimestamp(),
        };
    }

    pub fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.content);
        if (self.vector_embedding) |embedding| {
            allocator.free(embedding);
        }
    }

    pub fn updateAccess(self: *Self, enable_simd: bool) void {
        self.access_count += 1;
        self.last_accessed = std.time.microTimestamp();

        // SIMD-optimized importance decay
        if (enable_simd) {
            const time_factor = @as(f32, @floatFromInt(self.last_accessed - self.timestamp)) / 1000000.0;
            const decay_factor = 1.0 / (1.0 + time_factor * 0.001);
            const access_boost = @min(0.1, @as(f32, @floatFromInt(self.access_count)) * 0.01);
            self.importance = @min(1.0, self.importance * decay_factor + access_boost);
        }
    }
};

/// Enhanced agent configuration
pub const AgentConfig = struct {
    name: []const u8,
    default_persona: PersonaType = .adaptive,
    max_context_length: usize = 4096,
    enable_history: bool = true,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    capabilities: AgentCapabilities = .{},
    memory_size: usize = 1024 * 1024, // 1MB
    enable_logging: bool = true,
    log_level: std.log.Level = .info,
    use_custom_allocator: bool = false,
    enable_simd: bool = true,
    max_concurrent_requests: u32 = 10,
    enable_persona_routing: bool = true,

    pub fn validate(self: AgentConfig) AgentError!void {
        if (self.temperature < 0.0 or self.temperature > 2.0) {
            return AgentError.InvalidConfiguration;
        }
        if (self.top_p < 0.0 or self.top_p > 1.0) {
            return AgentError.InvalidConfiguration;
        }
        if (!self.capabilities.validate()) {
            return AgentError.InvalidConfiguration;
        }
    }
};

/// Performance statistics with comprehensive metrics
pub const PerformanceStats = struct {
    total_requests: u64 = 0,
    successful_requests: u64 = 0,
    failed_requests: u64 = 0,
    total_tokens_processed: u64 = 0,
    average_response_time_ms: f64 = 0.0,
    memory_usage_bytes: usize = 0,
    peak_memory_usage: usize = 0,
    cache_hit_rate: f32 = 0.0,
    concurrent_requests: u32 = 0,
    persona_usage: std.EnumArray(PersonaType, u64) = std.EnumArray(PersonaType, u64).initFill(0),

    pub fn updateResponseTime(self: *PerformanceStats, response_time_ms: f64) void {
        const total = self.total_requests;
        if (total > 0) {
            self.average_response_time_ms = (self.average_response_time_ms * @as(f64, @floatFromInt(total - 1)) + response_time_ms) / @as(f64, @floatFromInt(total));
        } else {
            self.average_response_time_ms = response_time_ms;
        }
    }

    pub fn recordSuccess(self: *PerformanceStats, persona: PersonaType) void {
        self.total_requests += 1;
        self.successful_requests += 1;
        self.persona_usage.set(persona, self.persona_usage.get(persona) + 1);
    }

    pub fn recordFailure(self: *PerformanceStats) void {
        self.total_requests += 1;
        self.failed_requests += 1;
    }

    pub fn getSuccessRate(self: *const PerformanceStats) f32 {
        if (self.total_requests == 0) return 0.0;
        return @as(f32, @floatFromInt(self.successful_requests)) / @as(f32, @floatFromInt(self.total_requests));
    }
};

/// Unified AI Agent with enhanced capabilities
pub const Agent = struct {
    config: AgentConfig,
    allocator: Allocator,
    state: AgentState = .idle,
    current_persona: PersonaType,
    conversation_history: std.ArrayList(Message),
    memory: std.ArrayList(MemoryEntry),
    performance_stats: PerformanceStats = .{},
    request_semaphore: std.Thread.Semaphore,
    state_mutex: std.Thread.Mutex = .{},

    const Self = @This();

    pub fn init(allocator: Allocator, config: AgentConfig) AgentError!*Self {
        try config.validate();

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .config = config,
            .allocator = allocator,
            .current_persona = config.default_persona,
            .conversation_history = std.ArrayList(Message).init(allocator),
            .memory = std.ArrayList(MemoryEntry).init(allocator),
            .request_semaphore = .{ .permits = config.max_concurrent_requests },
        };

        if (config.enable_logging) {
            std.log.info("Agent '{s}' initialized with persona: {s}", .{ config.name, config.default_persona.getDescription() });
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.config.enable_logging) {
            std.log.info("Agent '{s}' shutting down. Success rate: {d:.2}%", .{ self.config.name, self.performance_stats.getSuccessRate() * 100.0 });
        }

        // Clean up conversation history
        for (self.conversation_history.items) |message| {
            message.deinit(self.allocator);
        }
        self.conversation_history.deinit();

        // Clean up memory entries
        for (self.memory.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.memory.deinit();

        self.allocator.destroy(self);
    }

    /// Process user input with intelligent persona routing
    pub fn processInput(self: *Self, input: []const u8) AgentError![]const u8 {
        // Acquire semaphore for concurrency control
        self.request_semaphore.wait();
        defer self.request_semaphore.post();

        self.performance_stats.concurrent_requests += 1;
        defer self.performance_stats.concurrent_requests -= 1;

        const start_time = std.time.microTimestamp();
        defer {
            const end_time = std.time.microTimestamp();
            const elapsed = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0;
            self.performance_stats.updateResponseTime(elapsed);
        }

        // State transitions with validation
        try self.transitionState(.thinking);

        if (self.config.enable_logging) {
            std.log.debug("Processing input: {s}", .{input});
        }

        // Validate input
        if (input.len == 0) {
            self.performance_stats.recordFailure();
            try self.transitionState(.error_state);
            return AgentError.InvalidQuery;
        }

        try self.transitionState(.processing);

        // Select optimal persona if routing is enabled
        if (self.config.enable_persona_routing) {
            self.current_persona = self.selectPersona(input);
        }

        // Store input in memory
        try self.storeMemory(input, 0.5);

        // Add to conversation history
        const user_message = try Message.init(self.allocator, .user, input);
        try self.conversation_history.append(user_message);

        // Generate response
        const response = try self.generateResponse(input);
        errdefer self.allocator.free(response);

        try self.transitionState(.responding);

        // Store response in memory and history
        try self.storeMemory(response, 0.7);
        const assistant_message = try Message.init(self.allocator, .assistant, response);
        try self.conversation_history.append(assistant_message);

        // Trim history if needed
        try self.trimHistory();

        self.performance_stats.recordSuccess(self.current_persona);
        try self.transitionState(.idle);

        return response;
    }

    /// Select optimal persona based on input analysis
    fn selectPersona(self: *Self, input: []const u8) PersonaType {
        var best_persona = self.config.default_persona;
        var best_score: f32 = 0.0;

        // Analyze input for different characteristics
        const is_technical = std.mem.indexOf(u8, input, "code") != null or
            std.mem.indexOf(u8, input, "program") != null or
            std.mem.indexOf(u8, input, "algorithm") != null;
        const is_emotional = std.mem.indexOf(u8, input, "help") != null or
            std.mem.indexOf(u8, input, "sad") != null or
            std.mem.indexOf(u8, input, "worry") != null;
        const is_creative = std.mem.indexOf(u8, input, "creative") != null or
            std.mem.indexOf(u8, input, "idea") != null or
            std.mem.indexOf(u8, input, "imagine") != null;
        const is_direct = std.mem.indexOf(u8, input, "quick") != null or
            std.mem.indexOf(u8, input, "brief") != null;

        // Score each persona
        inline for (std.meta.fields(PersonaType)) |field| {
            const persona: PersonaType = @enumFromInt(field.value);
            const scoring = persona.getScoring();

            var score: f32 = 0.5; // Base score

            if (is_technical) score += scoring.technical * 0.4;
            if (is_emotional) score += scoring.empathy * 0.3;
            if (is_creative) score += scoring.creativity * 0.3;
            if (is_direct) score += scoring.directness * 0.2;

            if (score > best_score) {
                best_score = score;
                best_persona = persona;
            }
        }

        if (self.config.enable_logging and best_persona != self.current_persona) {
            std.log.info("Persona switched from {s} to {s} (score: {d:.2})", .{ self.current_persona.getDescription(), best_persona.getDescription(), best_score });
        }

        return best_persona;
    }

    /// Generate response based on current persona and capabilities
    fn generateResponse(self: *Self, input: []const u8) AgentError![]const u8 {
        // Capability-based response routing
        if (self.config.capabilities.code_generation and std.mem.indexOf(u8, input, "code") != null) {
            return try self.generateCodeResponse(input);
        } else if (self.config.capabilities.reasoning and
            (std.mem.indexOf(u8, input, "analyze") != null or std.mem.indexOf(u8, input, "think") != null))
        {
            return try self.generateReasoningResponse(input);
        } else if (self.config.capabilities.text_generation) {
            return try self.generateTextResponse(input);
        } else {
            return try self.generateDefaultResponse(input);
        }
    }

    fn generateCodeResponse(self: *Self, input: []const u8) AgentError![]const u8 {
        _ = input;
        const templates = [_][]const u8{
            "I can help you with code generation. What specific programming task do you need assistance with?",
            "For optimal code implementation, I recommend considering performance, readability, and maintainability.",
            "Let me help you write efficient code. What programming language and problem are you working with?",
        };

        const template_idx = @as(usize, @intCast(std.time.microTimestamp())) % templates.len;
        return try self.allocator.dupe(u8, templates[template_idx]);
    }

    fn generateReasoningResponse(self: *Self, input: []const u8) AgentError![]const u8 {
        _ = input;
        const response = switch (self.current_persona) {
            .analytical => "Let me analyze this systematically. I'll break down the problem into logical components and examine each aspect methodically.",
            .technical => "From a technical perspective, let's examine the underlying principles and apply structured reasoning to this challenge.",
            .solver => "I'll approach this step-by-step, identifying key variables and potential solutions through logical deduction.",
            else => "Let me think through this carefully, considering multiple perspectives and analyzing the available information.",
        };
        return try self.allocator.dupe(u8, response);
    }

    fn generateTextResponse(self: *Self, input: []const u8) AgentError![]const u8 {
        _ = input;
        const response = switch (self.current_persona) {
            .empathetic => "I understand your concern and I'm here to help. Let me provide a thoughtful response that addresses your needs.",
            .creative => "That's an interesting question! Let me explore some creative approaches and innovative solutions for you.",
            .educator => "Great question! Let me explain this in a clear, structured way that will help you understand the concept thoroughly.",
            .counselor => "I appreciate you sharing this with me. Let's work through this together with patience and understanding.",
            else => "Thank you for your question. I'm here to provide helpful, accurate information tailored to your needs.",
        };
        return try self.allocator.dupe(u8, response);
    }

    fn generateDefaultResponse(self: *Self, input: []const u8) AgentError![]const u8 {
        _ = input;
        const response = "Hello! I'm an AI agent ready to assist you. How can I help you today?";
        return try self.allocator.dupe(u8, response);
    }

    /// Store information in agent memory
    pub fn storeMemory(self: *Self, content: []const u8, importance: f32) AgentError!void {
        if (self.memory.items.len >= self.config.memory_size / @sizeOf(MemoryEntry)) {
            try self.pruneMemory();
        }

        var entry = try MemoryEntry.init(self.allocator, content, importance);
        entry.persona_context = self.current_persona;
        try self.memory.append(entry);

        // Update memory usage statistics
        self.performance_stats.memory_usage_bytes = self.memory.items.len * @sizeOf(MemoryEntry);
        if (self.performance_stats.memory_usage_bytes > self.performance_stats.peak_memory_usage) {
            self.performance_stats.peak_memory_usage = self.performance_stats.memory_usage_bytes;
        }
    }

    /// Prune memory using importance-based selection
    fn pruneMemory(self: *Self) AgentError!void {
        // Sort by composite score (importance + recency + access frequency)
        std.sort.insertion(MemoryEntry, self.memory.items, {}, struct {
            fn lessThan(_: void, a: MemoryEntry, b: MemoryEntry) bool {
                const a_score = a.importance +
                    (@as(f32, @floatFromInt(a.access_count)) * 0.1) +
                    (@as(f32, @floatFromInt(a.last_accessed)) * 0.0001);
                const b_score = b.importance +
                    (@as(f32, @floatFromInt(b.access_count)) * 0.1) +
                    (@as(f32, @floatFromInt(b.last_accessed)) * 0.0001);
                return a_score < b_score;
            }
        }.lessThan);

        // Remove bottom 25% of memories
        const remove_count = self.memory.items.len / 4;
        for (0..remove_count) |i| {
            var entry = self.memory.items[i];
            entry.deinit(self.allocator);
        }

        // Compact remaining memories
        const remaining_count = self.memory.items.len - remove_count;
        if (remove_count > 0 and remaining_count > 0) {
            std.mem.copyForwards(MemoryEntry, self.memory.items[0..remaining_count], self.memory.items[remove_count..]);
        }
        self.memory.items.len = remaining_count;

        if (self.config.enable_logging) {
            std.log.debug("Pruned {d} memories, {d} remaining", .{ remove_count, remaining_count });
        }
    }

    /// Trim conversation history to stay within context limits
    fn trimHistory(self: *Self) AgentError!void {
        if (!self.config.enable_history) return;

        var total_length: usize = 0;
        var trim_index: usize = 0;

        // Calculate total content length
        for (self.conversation_history.items, 0..) |message, i| {
            const new_length = total_length + message.content.len;
            if (new_length > self.config.max_context_length) {
                trim_index = i;
                break;
            }
            total_length = new_length;
        }

        // Remove older messages if needed
        if (trim_index > 0) {
            for (self.conversation_history.items[0..trim_index]) |message| {
                message.deinit(self.allocator);
            }

            const remaining = self.conversation_history.items[trim_index..];
            std.mem.copyForwards(Message, self.conversation_history.items[0..remaining.len], remaining);
            self.conversation_history.items.len = remaining.len;
        }
    }

    /// Thread-safe state transition
    fn transitionState(self: *Self, new_state: AgentState) AgentError!void {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();

        if (!AgentState.canTransitionTo(self.state, new_state)) {
            return AgentError.InvalidStateTransition;
        }
        self.state = new_state;
    }

    /// Get current agent state safely
    pub fn getState(self: *const Self) AgentState {
        return self.state;
    }

    /// Get performance statistics
    pub fn getStats(self: *const Self) PerformanceStats {
        var stats = self.performance_stats;
        stats.memory_usage_bytes = self.memory.items.len * @sizeOf(MemoryEntry);
        return stats;
    }

    /// Set persona explicitly
    pub fn setPersona(self: *Self, persona: PersonaType) void {
        self.current_persona = persona;
        if (self.config.enable_logging) {
            std.log.info("Persona set to: {s}", .{persona.getDescription()});
        }
    }

    /// Get current persona
    pub fn getPersona(self: *const Self) PersonaType {
        return self.current_persona;
    }

    /// Clear conversation history
    pub fn clearHistory(self: *Self) void {
        for (self.conversation_history.items) |message| {
            message.deinit(self.allocator);
        }
        self.conversation_history.clearRetainingCapacity();
    }

    /// Clear memory
    pub fn clearMemory(self: *Self) void {
        for (self.memory.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.memory.clearRetainingCapacity();
    }
};

test "agent creation and basic functionality" {
    const testing = std.testing;

    const config = AgentConfig{
        .name = "test_agent",
        .capabilities = .{ .text_generation = true, .reasoning = true },
        .enable_logging = false,
        .max_concurrent_requests = 1,
    };

    var agent = try Agent.init(testing.allocator, config);
    defer agent.deinit();

    // Test basic processing
    const response = try agent.processInput("Hello, can you help me?");
    defer testing.allocator.free(response);

    try testing.expect(response.len > 0);
    try testing.expectEqual(@as(usize, 1), agent.performance_stats.successful_requests);
}

test "persona selection and routing" {
    const testing = std.testing;

    const config = AgentConfig{
        .name = "test_agent",
        .capabilities = .{ .text_generation = true, .code_generation = true },
        .enable_logging = false,
        .enable_persona_routing = true,
    };

    var agent = try Agent.init(testing.allocator, config);
    defer agent.deinit();

    // Test technical query should select technical persona
    _ = try agent.processInput("Can you help me write some code?");

    // Check that persona was selected appropriately
    const current_persona = agent.getPersona();
    try testing.expect(current_persona == .technical or current_persona == .solver);
}
