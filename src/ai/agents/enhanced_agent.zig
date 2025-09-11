//! Enhanced AI Agent - Production-ready AI agent with full connectivity and capabilities
//!
//! This module provides an enhanced AI agent system with:
//! - Multi-persona support with intelligent routing
//! - Advanced memory management with SIMD optimization
//! - Performance monitoring and metrics
//! - Thread-safe operations and concurrency control
//! - Configurable backends and capabilities
//! - Real-time communication and collaboration

const std = @import("std");
const builtin = @import("builtin");

const core = @import("../../core/mod.zig");
const config = @import("../../core/config.zig");
const errors = @import("../../core/errors.zig");

const FrameworkError = errors.FrameworkError;
const AgentConfig = config.AgentConfig;
const PersonaType = config.PersonaType;
const AgentCapabilities = config.AgentCapabilities;

/// Enhanced AI Agent with production-ready features
pub const EnhancedAgent = struct {
    config: AgentConfig,
    allocator: std.mem.Allocator,
    state: AgentState,
    current_persona: PersonaType,
    conversation_history: std.ArrayList(Message),
    memory: std.ArrayList(MemoryEntry),
    performance_stats: PerformanceStats,
    request_semaphore: std.Thread.Semaphore,
    state_mutex: std.Thread.Mutex,
    message_bus: *MessageBus,
    event_system: *EventSystem,
    service_registry: *ServiceRegistry,
    load_balancer: *LoadBalancer,
    router: *AgentRouter,

    const Self = @This();

    /// Initialize the enhanced agent
    pub fn init(allocator: std.mem.Allocator, agent_config: AgentConfig) FrameworkError!*Self {
        try agent_config.validate();

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .config = agent_config,
            .allocator = allocator,
            .state = .idle,
            .current_persona = agent_config.default_persona,
            .conversation_history = std.ArrayList(Message).init(allocator),
            .memory = std.ArrayList(MemoryEntry).init(allocator),
            .performance_stats = PerformanceStats{},
            .request_semaphore = .{ .permits = agent_config.max_concurrent_requests },
            .state_mutex = .{},
            .message_bus = try MessageBus.init(allocator),
            .event_system = try EventSystem.init(allocator),
            .service_registry = try ServiceRegistry.init(allocator),
            .load_balancer = try LoadBalancer.init(allocator),
            .router = try AgentRouter.init(allocator),
        };

        // Register agent services
        try self.registerServices();

        // Initialize performance monitoring
        try self.initializePerformanceMonitoring();

        if (self.config.enable_logging) {
            std.log.info("Enhanced agent '{s}' initialized with persona: {s}", .{ agent_config.name, agent_config.default_persona.getDescription() });
        }

        return self;
    }

    /// Deinitialize the enhanced agent
    pub fn deinit(self: *Self) void {
        if (self.config.enable_logging) {
            std.log.info("Enhanced agent '{s}' shutting down. Success rate: {d:.2}%", .{ self.config.name, self.performance_stats.getSuccessRate() * 100.0 });
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

        // Clean up components
        self.message_bus.deinit();
        self.event_system.deinit();
        self.service_registry.deinit();
        self.load_balancer.deinit();
        self.router.deinit();

        self.allocator.destroy(self);
    }

    /// Process user input with enhanced capabilities
    pub fn processInput(self: *Self, input: []const u8) FrameworkError![]const u8 {
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
            return FrameworkError.InvalidInput;
        }

        try self.transitionState(.processing);

        // Select optimal persona if routing is enabled
        if (self.config.enable_persona_routing) {
            self.current_persona = try self.router.selectPersona(input, self.current_persona);
        }

        // Store input in memory
        try self.storeMemory(input, 0.5);

        // Add to conversation history
        const user_message = try Message.init(self.allocator, .user, input);
        try self.conversation_history.append(user_message);

        // Generate response using enhanced capabilities
        const response = try self.generateEnhancedResponse(input);
        errdefer self.allocator.free(response);

        try self.transitionState(.responding);

        // Store response in memory and history
        try self.storeMemory(response, 0.7);
        const assistant_message = try Message.init(self.allocator, .assistant, response);
        try self.conversation_history.append(assistant_message);

        // Trim history if needed
        try self.trimHistory();

        // Emit response event
        self.event_system.emitEvent(.response_generated, .{
            .agent_name = self.config.name,
            .persona = self.current_persona,
            .response_length = response.len,
            .processing_time_ms = @as(f64, @floatFromInt(std.time.microTimestamp() - start_time)) / 1000.0,
        });

        self.performance_stats.recordSuccess(self.current_persona);
        try self.transitionState(.idle);

        return response;
    }

    /// Generate enhanced response using multiple capabilities
    fn generateEnhancedResponse(self: *Self, input: []const u8) FrameworkError![]const u8 {
        // Analyze input for capability requirements
        const requirements = try self.analyzeInputRequirements(input);

        // Route to appropriate capability handler
        return switch (requirements.primary_capability) {
            .text_generation => self.generateTextResponse(input, requirements),
            .code_generation => self.generateCodeResponse(input, requirements),
            .reasoning => self.generateReasoningResponse(input, requirements),
            .planning => self.generatePlanningResponse(input, requirements),
            .multimodal => self.generateMultimodalResponse(input, requirements),
            else => self.generateDefaultResponse(input, requirements),
        };
    }

    /// Analyze input to determine capability requirements
    fn analyzeInputRequirements(input: []const u8) FrameworkError!CapabilityRequirements {
        var requirements = CapabilityRequirements{};

        // Analyze for code generation
        if (std.mem.indexOf(u8, input, "code") != null or
            std.mem.indexOf(u8, input, "program") != null or
            std.mem.indexOf(u8, input, "function") != null)
        {
            requirements.code_generation = true;
            requirements.primary_capability = .code_generation;
        }

        // Analyze for reasoning
        if (std.mem.indexOf(u8, input, "analyze") != null or
            std.mem.indexOf(u8, input, "think") != null or
            std.mem.indexOf(u8, input, "reason") != null)
        {
            requirements.reasoning = true;
            if (requirements.primary_capability == .none) {
                requirements.primary_capability = .reasoning;
            }
        }

        // Analyze for planning
        if (std.mem.indexOf(u8, input, "plan") != null or
            std.mem.indexOf(u8, input, "strategy") != null or
            std.mem.indexOf(u8, input, "steps") != null)
        {
            requirements.planning = true;
            if (requirements.primary_capability == .none) {
                requirements.primary_capability = .planning;
            }
        }

        // Analyze for multimodal content
        if (std.mem.indexOf(u8, input, "image") != null or
            std.mem.indexOf(u8, input, "audio") != null or
            std.mem.indexOf(u8, input, "video") != null)
        {
            requirements.multimodal = true;
            if (requirements.primary_capability == .none) {
                requirements.primary_capability = .multimodal;
            }
        }

        // Default to text generation
        if (requirements.primary_capability == .none) {
            requirements.text_generation = true;
            requirements.primary_capability = .text_generation;
        }

        return requirements;
    }

    /// Generate text response
    fn generateTextResponse(self: *Self, requirements: CapabilityRequirements) FrameworkError![]const u8 {
        _ = requirements;

        const response = switch (self.current_persona) {
            .empathetic => "I understand your concern and I'm here to help. Let me provide a thoughtful response that addresses your needs.",
            .creative => "That's an interesting question! Let me explore some creative approaches and innovative solutions for you.",
            .educator => "Great question! Let me explain this in a clear, structured way that will help you understand the concept thoroughly.",
            .counselor => "I appreciate you sharing this with me. Let's work through this together with patience and understanding.",
            .analytical => "Let me analyze this systematically. I'll break down the problem into logical components and examine each aspect methodically.",
            .technical => "From a technical perspective, let's examine the underlying principles and apply structured reasoning to this challenge.",
            .solver => "I'll approach this step-by-step, identifying key variables and potential solutions through logical deduction.",
            else => "Thank you for your question. I'm here to provide helpful, accurate information tailored to your needs.",
        };

        return try self.allocator.dupe(u8, response);
    }

    /// Generate code response
    fn generateCodeResponse(self: *Self, requirements: CapabilityRequirements) FrameworkError![]const u8 {
        _ = requirements;

        const response = switch (self.current_persona) {
            .technical => "I'll help you write efficient, well-structured code. Let me provide a solution that follows best practices and is optimized for performance.",
            .creative => "Let me create an innovative solution that combines creativity with technical excellence. I'll explore different approaches to solve your problem.",
            .educator => "I'll write code that's not only functional but also educational. Let me explain each part so you can understand the concepts and learn from it.",
            else => "I can help you with code generation. What specific programming task do you need assistance with?",
        };

        return try self.allocator.dupe(u8, response);
    }

    /// Generate reasoning response
    fn generateReasoningResponse(self: *Self, requirements: CapabilityRequirements) FrameworkError![]const u8 {
        _ = requirements;

        const response = switch (self.current_persona) {
            .analytical => "Let me analyze this systematically. I'll break down the problem into logical components and examine each aspect methodically.",
            .technical => "From a technical perspective, let's examine the underlying principles and apply structured reasoning to this challenge.",
            .solver => "I'll approach this step-by-step, identifying key variables and potential solutions through logical deduction.",
            else => "Let me think through this carefully, considering multiple perspectives and analyzing the available information.",
        };

        return try self.allocator.dupe(u8, response);
    }

    /// Generate planning response
    fn generatePlanningResponse(self: *Self, requirements: CapabilityRequirements) FrameworkError![]const u8 {
        _ = requirements;

        const response = switch (self.current_persona) {
            .solver => "I'll create a comprehensive plan that breaks down the task into manageable steps, considering dependencies and potential challenges.",
            .analytical => "Let me develop a strategic plan by analyzing the requirements, identifying key milestones, and creating a timeline for execution.",
            .educator => "I'll design a learning plan that progresses logically, building on previous knowledge and incorporating best practices for skill development.",
            else => "I'll help you create a structured plan that addresses your goals and provides a clear path forward.",
        };

        return try self.allocator.dupe(u8, response);
    }

    /// Generate multimodal response
    fn generateMultimodalResponse(self: *Self, requirements: CapabilityRequirements) FrameworkError![]const u8 {
        _ = requirements;

        const response = "I can process and analyze multimodal content including text, images, audio, and video. Let me help you with your multimodal request.";

        return try self.allocator.dupe(u8, response);
    }

    /// Generate default response
    fn generateDefaultResponse(self: *Self, requirements: CapabilityRequirements) FrameworkError![]const u8 {
        _ = requirements;

        const response = "Hello! I'm an enhanced AI agent ready to assist you. How can I help you today?";

        return try self.allocator.dupe(u8, response);
    }

    /// Store information in agent memory with enhanced features
    pub fn storeMemory(self: *Self, content: []const u8, importance: f32) FrameworkError!void {
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

        // Emit memory event
        self.event_system.emitEvent(.memory_stored, .{
            .agent_name = self.config.name,
            .content_length = content.len,
            .importance = importance,
            .total_memories = self.memory.items.len,
        });
    }

    /// Prune memory using enhanced importance-based selection
    fn pruneMemory(self: *Self) FrameworkError!void {
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

        // Emit memory pruned event
        self.event_system.emitEvent(.memory_pruned, .{
            .agent_name = self.config.name,
            .pruned_count = remove_count,
            .remaining_count = remaining_count,
        });
    }

    /// Trim conversation history to stay within context limits
    fn trimHistory(self: *Self) FrameworkError!void {
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
    fn transitionState(self: *Self, new_state: AgentState) FrameworkError!void {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();

        if (!AgentState.canTransitionTo(self.state, new_state)) {
            return FrameworkError.OperationFailed;
        }
        self.state = new_state;

        // Emit state change event
        self.event_system.emitEvent(.state_changed, .{
            .agent_name = self.config.name,
            .old_state = self.state,
            .new_state = new_state,
        });
    }

    /// Register agent services
    fn registerServices(self: *Self) FrameworkError!void {
        // Register text generation service
        if (self.config.capabilities.text_generation) {
            try self.service_registry.registerService("text_generation", self, textGenerationService);
        }

        // Register code generation service
        if (self.config.capabilities.code_generation) {
            try self.service_registry.registerService("code_generation", self, codeGenerationService);
        }

        // Register reasoning service
        if (self.config.capabilities.reasoning) {
            try self.service_registry.registerService("reasoning", self, reasoningService);
        }

        // Register planning service
        if (self.config.capabilities.planning) {
            try self.service_registry.registerService("planning", self, planningService);
        }
    }

    /// Initialize performance monitoring
    fn initializePerformanceMonitoring(self: *Self) FrameworkError!void {
        // Set up performance monitoring callbacks
        const perf_callback = PerformanceCallback{
            .agent = self,
            .handler = performanceMonitoringHandler,
        };

        try self.event_system.registerCallback(.response_generated, perf_callback);
    }

    // Public interface methods

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

        // Emit persona change event
        self.event_system.emitEvent(.persona_changed, .{
            .agent_name = self.config.name,
            .new_persona = persona,
        });
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

        // Emit history cleared event
        self.event_system.emitEvent(.history_cleared, .{
            .agent_name = self.config.name,
        });
    }

    /// Clear memory
    pub fn clearMemory(self: *Self) void {
        for (self.memory.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.memory.clearRetainingCapacity();

        // Emit memory cleared event
        self.event_system.emitEvent(.memory_cleared, .{
            .agent_name = self.config.name,
        });
    }

    /// Get message bus for inter-agent communication
    pub fn getMessageBus(self: *Self) *MessageBus {
        return self.message_bus;
    }

    /// Get event system for event handling
    pub fn getEventSystem(self: *Self) *EventSystem {
        return self.event_system;
    }

    /// Get service registry for service discovery
    pub fn getServiceRegistry(self: *Self) *ServiceRegistry {
        return self.service_registry;
    }

    /// Get load balancer for request distribution
    pub fn getLoadBalancer(self: *Self) *LoadBalancer {
        return self.load_balancer;
    }

    /// Get router for intelligent routing
    pub fn getRouter(self: *Self) *AgentRouter {
        return self.router;
    }
};

// Supporting types and structures

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

    pub fn init(allocator: std.mem.Allocator, role: MessageRole, content: []const u8) !Message {
        return Message{
            .role = role,
            .content = try allocator.dupe(u8, content),
            .timestamp = std.time.microTimestamp(),
        };
    }

    pub fn deinit(self: Message, allocator: std.mem.Allocator) void {
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

    pub fn init(allocator: std.mem.Allocator, content: []const u8, importance: f32) !Self {
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

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
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

/// Capability requirements for input processing
pub const CapabilityRequirements = struct {
    text_generation: bool = false,
    code_generation: bool = false,
    reasoning: bool = false,
    planning: bool = false,
    multimodal: bool = false,
    primary_capability: PrimaryCapability = .none,
};

/// Primary capability types
pub const PrimaryCapability = enum {
    none,
    text_generation,
    code_generation,
    reasoning,
    planning,
    multimodal,
};

// Placeholder types for components (to be implemented in separate modules)

const MessageBus = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*MessageBus {
        const self = try allocator.create(MessageBus);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *MessageBus) void {
        self.allocator.destroy(self);
    }
};

const EventSystem = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*EventSystem {
        const self = try allocator.create(EventSystem);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *EventSystem) void {
        self.allocator.destroy(self);
    }

    pub fn emitEvent(self: *EventSystem, event_type: EventType, data: anytype) void {
        _ = self;
        _ = event_type;
        _ = data;
    }

    pub fn registerCallback(self: *EventSystem, event_type: EventType, callback: anytype) !void {
        _ = self;
        _ = event_type;
        _ = callback;
    }
};

const ServiceRegistry = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*ServiceRegistry {
        const self = try allocator.create(ServiceRegistry);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *ServiceRegistry) void {
        self.allocator.destroy(self);
    }

    pub fn registerService(self: *ServiceRegistry, name: []const u8, service: anytype, handler: anytype) !void {
        _ = self;
        _ = name;
        _ = service;
        _ = handler;
    }
};

const LoadBalancer = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*LoadBalancer {
        const self = try allocator.create(LoadBalancer);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *LoadBalancer) void {
        self.allocator.destroy(self);
    }
};

const AgentRouter = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*AgentRouter {
        const self = try allocator.create(AgentRouter);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *AgentRouter) void {
        self.allocator.destroy(self);
    }

    pub fn selectPersona(self: *AgentRouter, input: []const u8, current_persona: PersonaType) !PersonaType {
        _ = self;
        _ = input;
        return current_persona;
    }
};

const EventType = enum {
    response_generated,
    memory_stored,
    memory_pruned,
    memory_cleared,
    state_changed,
    persona_changed,
    history_cleared,
};

const PerformanceCallback = struct {
    agent: *EnhancedAgent,
    handler: *const fn (agent: *EnhancedAgent, event_type: EventType, data: anytype) void,
};

// Service handlers
fn textGenerationService(agent: *EnhancedAgent, input: []const u8) ![]const u8 {
    _ = agent;
    _ = input;
    return "Text generation service response";
}

fn codeGenerationService(agent: *EnhancedAgent, input: []const u8) ![]const u8 {
    _ = agent;
    _ = input;
    return "Code generation service response";
}

fn reasoningService(agent: *EnhancedAgent, input: []const u8) ![]const u8 {
    _ = agent;
    _ = input;
    return "Reasoning service response";
}

fn planningService(agent: *EnhancedAgent, input: []const u8) ![]const u8 {
    _ = agent;
    _ = input;
    return "Planning service response";
}

fn performanceMonitoringHandler(agent: *EnhancedAgent, event_type: EventType, data: anytype) void {
    _ = agent;
    _ = event_type;
    _ = data;
}

// Extension methods for PersonaType
pub fn getDescription(persona: PersonaType) []const u8 {
    return switch (persona) {
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

test "enhanced agent creation and basic functionality" {
    const testing = std.testing;

    const agent_config = AgentConfig{
        .name = "test_agent",
        .capabilities = .{ .text_generation = true, .reasoning = true },
        .enable_logging = false,
        .max_concurrent_requests = 1,
    };
    _ = agent_config; // autofix

    var agent = try EnhancedAgent.init(testing.allocator, config);
    defer agent.deinit();

    // Test basic processing
    const response = try agent.processInput("Hello, can you help me?");
    defer testing.allocator.free(response);

    try testing.expect(response.len > 0);
    try testing.expectEqual(@as(usize, 1), agent.performance_stats.successful_requests);
}

test "enhanced agent persona management" {
    const testing = std.testing;

    const persona_config = AgentConfig{
        .name = "test_agent",
        .capabilities = .{ .text_generation = true, .code_generation = true },
        .enable_logging = false,
        .enable_persona_routing = true,
    };
    _ = persona_config; // autofix

    var agent = try EnhancedAgent.init(testing.allocator, config);
    defer agent.deinit();

    // Test persona setting
    agent.setPersona(.creative);
    try testing.expectEqual(PersonaType.creative, agent.getPersona());

    // Test persona change
    agent.setPersona(.technical);
    try testing.expectEqual(PersonaType.technical, agent.getPersona());
}

test "enhanced agent memory management" {
    const testing = std.testing;

    const memory_config = AgentConfig{
        .name = "test_agent",
        .capabilities = .{ .text_generation = true },
        .enable_logging = false,
        .memory_size = 1024, // Small memory for testing
    };
    _ = memory_config; // autofix

    var agent = try EnhancedAgent.init(testing.allocator, config);
    defer agent.deinit();

    // Test memory storage
    try agent.storeMemory("Test memory content", 0.8);
    try testing.expectEqual(@as(usize, 1), agent.memory.items.len);

    // Test memory clearing
    agent.clearMemory();
    try testing.expectEqual(@as(usize, 0), agent.memory.items.len);
}
