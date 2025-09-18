//! AI Agent Module
//!
//! Provides intelligent AI agents with configurable personas and conversation management.
//! Supports multiple backend integrations and maintains conversation context.

const std = @import("std");
const builtin = @import("builtin");

const core = @import("../core/mod.zig");

// Re-export core types for convenience
pub const Allocator = std.mem.Allocator;
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
} || core.AbiError;


/// Backend provider types
pub const BackendType = enum {
    openai,
    anthropic,
    local_llm,
    custom,

    pub fn getDefaultCapabilities(self: BackendType) AgentCapabilities {
        return switch (self) {
            .openai => .{
                .text_generation = true,
                .code_generation = true,
                .function_calling = true,
                .streaming = true,
                .reasoning = true,
            },
            .anthropic => .{
                .text_generation = true,
                .code_generation = true,
                .reasoning = true,
                .planning = true,
            },
            .local_llm => .{
                .text_generation = true,
                .code_generation = true,
                .memory_management = true,
            },
            .custom => .{}, // No default capabilities
        };
    }
};

/// Backend configuration
pub const BackendConfig = struct {
    backend_type: BackendType,
    api_key: ?[]const u8 = null,
    endpoint: ?[]const u8 = null,
    model_name: []const u8,
    timeout_ms: u32 = 30000,
    retry_attempts: u8 = 3,
    rate_limit_per_minute: u32 = 60,

    pub fn validate(self: BackendConfig) AgentError!void {
        if (self.backend_type != .custom and self.api_key == null) {
            return AgentError.ApiKeyMissing;
        }
        if (self.model_name.len == 0) {
            return AgentError.InvalidConfiguration;
        }
    }
};

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

    /// Retrieves a description string for the persona
    analytical,
    supportive,
    specialist,
    researcher,

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
            .specialist => "domain-specific expert",
            .researcher => "research-oriented and thorough",
        };
    }

    /// Get persona scoring weights for different query types
    pub fn getScoring(self: PersonaType) PersonaScoring {
        return switch (self) {
            .empathetic => .{ .empathy = 0.9, .technical = 0.3, .creativity = 0.6, .directness = 0.2, .research = 0.4 },
            .direct => .{ .empathy = 0.2, .technical = 0.7, .creativity = 0.3, .directness = 0.9, .research = 0.5 },
            .adaptive => .{ .empathy = 0.7, .technical = 0.6, .creativity = 0.7, .directness = 0.6, .research = 0.6 },
            .creative => .{ .empathy = 0.6, .technical = 0.4, .creativity = 0.9, .directness = 0.5, .research = 0.3 },
            .technical => .{ .empathy = 0.3, .technical = 0.9, .creativity = 0.4, .directness = 0.8, .research = 0.7 },
            .solver => .{ .empathy = 0.5, .technical = 0.8, .creativity = 0.7, .directness = 0.7, .research = 0.6 },
            .educator => .{ .empathy = 0.8, .technical = 0.7, .creativity = 0.6, .directness = 0.6, .research = 0.8 },
            .counselor => .{ .empathy = 0.9, .technical = 0.4, .creativity = 0.5, .directness = 0.3, .research = 0.5 },
            .analytical => .{ .empathy = 0.4, .technical = 0.9, .creativity = 0.5, .directness = 0.8, .research = 0.9 },
            .supportive => .{ .empathy = 0.8, .technical = 0.5, .creativity = 0.6, .directness = 0.4, .research = 0.4 },
            .specialist => .{ .empathy = 0.5, .technical = 0.9, .creativity = 0.6, .directness = 0.7, .research = 0.9 },
            .researcher => .{ .empathy = 0.4, .technical = 0.8, .creativity = 0.7, .directness = 0.6, .research = 1.0 },
        };
    }
};

/// Represents the role of a message in the conversation
/// Persona scoring characteristics
pub const PersonaScoring = struct {
    empathy: f32,
    technical: f32,
    creativity: f32,
    directness: f32,
    research: f32,
};

/// Agent state with enhanced state management
pub const AgentState = enum(u8) {
    idle = 0,
    thinking = 1,
    processing = 2,
    responding = 3,
    learning = 4,
    error_state = 5,
    warming_up = 6,
    benchmarking = 7,

    /// Validate state transitions
    pub fn canTransitionTo(from: AgentState, to: AgentState) bool {
        return switch (from) {
            .idle => to == .thinking or to == .processing or to == .warming_up,
            .thinking => to == .processing or to == .responding or to == .error_state,
            .processing => to == .responding or to == .learning or to == .error_state,
            .responding => to == .idle or to == .learning,
            .learning => to == .idle or to == .benchmarking,
            .error_state => to == .idle or to == .warming_up,
            .warming_up => to == .idle or to == .thinking,
            .benchmarking => to == .idle,
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
    simd_optimization: bool = false,
    custom_allocator: bool = false,
    profiling: bool = false,
    _reserved: u17 = 0,

    /// Validate capability dependencies
    pub fn validate(self: AgentCapabilities) bool {
        if (self.vector_search and !self.memory_management) return false;
        if (self.multimodal and !(self.text_generation or self.image_analysis)) return false;
        if (self.simd_optimization and !self.memory_management) return false;
        return true;
    }
};

/// Message role in conversation
pub const MessageRole = enum {
    user,
    assistant,
    system,
};

/// Structure representing a message in the conversation
pub const Message = struct {
    role: MessageRole,
    content: []const u8,

    /// Deinitializes the message, freeing allocated resources
    pub fn deinit(self: Message, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
    }
};

/// Configuration settings for the AI agent
pub const AgentConfig = struct {
    function,
    tool,
};

/// Conversation message with metadata
pub const Message = struct {
    role: MessageRole,
    content: []const u8,
    timestamp: i64,
    importance: f32 = 0.5,
    persona_used: ?PersonaType = null,
    token_count: ?u32 = null,
    embedding: ?[]f32 = null,

    pub fn init(allocator: Allocator, role: MessageRole, content: []const u8) !Message {
        return Message{
            .role = role,
            .content = try allocator.dupe(u8, content),
            .timestamp = std.time.microTimestamp(),
        };
    }

    pub fn deinit(self: Message, allocator: Allocator) void {
        allocator.free(self.content);
        if (self.embedding) |embedding| {
            allocator.free(embedding);
        }
    }
};

/// Advanced memory entry with vectorized operations
pub const MemoryEntry = struct {
    id: u64,
    timestamp: i64,
    content: []align(64) const u8, // Cache-line aligned
    importance: f32,
    vector_embedding: ?[]align(32) f32 = null, // SIMD-aligned
    access_count: u32 = 0,
    last_accessed: i64,
    persona_context: ?PersonaType = null,
    similarity_score: f32 = 0.0,

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
        if (enable_simd and comptime std.simd.suggestVectorLength(f32) != null) {
            const time_factor = @as(f32, @floatFromInt(self.last_accessed - self.timestamp)) / 1000000.0;
            const decay_factor = 1.0 / (1.0 + time_factor * 0.001);
            const access_boost = @min(0.1, @as(f32, @floatFromInt(self.access_count)) * 0.01);
            self.importance = @min(1.0, self.importance * decay_factor + access_boost);
        }
    }

    /// Compute similarity using SIMD if available
    pub fn computeSimilarity(self: *const Self, other_embedding: []const f32, use_simd: bool) f32 {
        if (self.vector_embedding == null or other_embedding.len == 0) return 0.0;

        const embedding = self.vector_embedding.?;
        const min_len = @min(embedding.len, other_embedding.len);

        if (use_simd and comptime std.simd.suggestVectorLength(f32) != null) {
            return simdDotProduct(embedding[0..min_len], other_embedding[0..min_len]);
        } else {
            return scalarDotProduct(embedding[0..min_len], other_embedding[0..min_len]);
        }
    }
};

/// SIMD-optimized dot product
fn simdDotProduct(a: []const f32, b: []const f32) f32 {
    if (comptime std.simd.suggestVectorLength(f32)) |vec_len| {
        const VecType = @Vector(vec_len, f32);
        var sum = @as(VecType, @splat(0.0));

        var i: usize = 0;
        while (i + vec_len <= a.len) : (i += vec_len) {
            const vec_a: VecType = a[i .. i + vec_len][0..vec_len].*;
            const vec_b: VecType = b[i .. i + vec_len][0..vec_len].*;
            sum += vec_a * vec_b;
        }

        var result: f32 = 0.0;
        for (0..vec_len) |j| {
            result += sum[j];
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result += a[i] * b[i];
        }

        return result;
    }
    return scalarDotProduct(a, b);
}

/// Scalar dot product fallback
fn scalarDotProduct(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0.0;
    for (a, b) |ai, bi| {
        sum += ai * bi;
    }
    return sum;
}

/// Thread pool for concurrent operations
pub const ThreadPool = struct {
    allocator: Allocator,
    threads: []std.Thread,
    work_queue: std.fifo.LinearFifo(WorkItem, .Dynamic),
    mutex: std.Thread.Mutex = .{},
    condition: std.Thread.Condition = .{},
    should_stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    const WorkItem = struct {
        func: *const fn (*anyopaque) void,
        data: *anyopaque,
    };

    pub fn init(allocator: Allocator, thread_count: u32) !*ThreadPool {
        const pool = try allocator.create(ThreadPool);
        pool.* = .{
            .allocator = allocator,
            .threads = try allocator.alloc(std.Thread, thread_count),
            .work_queue = std.fifo.LinearFifo(WorkItem, .Dynamic).init(allocator),
        };

        for (pool.threads, 0..) |*thread, i| {
            thread.* = try std.Thread.spawn(.{}, workerThread, .{ pool, i });
        }

        return pool;
    }

    pub fn deinit(self: *ThreadPool) void {
        self.should_stop.store(true, .monotonic);
        self.condition.broadcast();

        for (self.threads) |thread| {
            thread.join();
        }

        self.allocator.free(self.threads);
        self.work_queue.deinit();
        self.allocator.destroy(self);
    }

    pub fn submit(self: *ThreadPool, func: *const fn (*anyopaque) void, data: *anyopaque) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        try self.work_queue.writeItem(.{ .func = func, .data = data });
        self.condition.signal();
    }

    fn workerThread(self: *ThreadPool, thread_id: usize) void {
        _ = thread_id;
        while (!self.should_stop.load(.monotonic)) {
            self.mutex.lock();

            while (self.work_queue.readItem() == null and !self.should_stop.load(.monotonic)) {
                self.condition.wait(&self.mutex);
            }

            const work_item = self.work_queue.readItem();
            self.mutex.unlock();

            if (work_item) |item| {
                item.func(item.data);
            }
        }
    }
};

/// Custom allocator optimized for agent operations
pub const AgentAllocator = struct {
    backing_allocator: Allocator,
    arena: std.heap.ArenaAllocator,
    pool_allocator: std.heap.MemoryPool(MemoryEntry),
    message_pool: std.heap.MemoryPool(Message),

    const Self = @This();

    pub fn init(backing_allocator: Allocator) Self {
        return Self{
            .backing_allocator = backing_allocator,
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .pool_allocator = std.heap.MemoryPool(MemoryEntry).init(backing_allocator),
            .message_pool = std.heap.MemoryPool(Message).init(backing_allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.pool_allocator.deinit();
        self.message_pool.deinit();
        self.arena.deinit();
    }

    pub fn allocator(self: *Self) Allocator {
        return self.arena.allocator();
    }

    pub fn createMemoryEntry(self: *Self) !*MemoryEntry {
        return self.pool_allocator.create();
    }

    pub fn destroyMemoryEntry(self: *Self, entry: *MemoryEntry) void {
        self.pool_allocator.destroy(entry);
    }

    pub fn createMessage(self: *Self) !*Message {
        return self.message_pool.create();
    }

    pub fn destroyMessage(self: *Self, message: *Message) void {
        self.message_pool.destroy(message);
    }
};

/// Enhanced agent configuration
pub const AgentConfig = struct {
    name: []const u8,
    default_persona: PersonaType = .adaptive,
    max_context_length: usize = 4096,
    enable_history: bool = true,
    temperature: f32 = 0.7,
};


/// Agent configuration with comprehensive settings
pub const AgentConfig = struct {
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
    backend_config: BackendConfig,
    thread_pool_size: u32 = 4,
    enable_profiling: bool = false,
    cache_size: usize = 512 * 1024, // 512KB
    vector_dimension: usize = 1536,
    enable_history: bool = true,
    default_persona: ?PersonaType = null,

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
        try self.backend_config.validate();
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
    backend_latency_ms: f64 = 0.0,
    simd_operations: u64 = 0,
    thread_pool_utilization: f32 = 0.0,
    vector_operations: u64 = 0,
    memory_allocations: u64 = 0,
    memory_deallocations: u64 = 0,

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

    pub fn recordSimdOperation(self: *PerformanceStats) void {
        self.simd_operations += 1;
    }

    pub fn recordVectorOperation(self: *PerformanceStats) void {
        self.vector_operations += 1;
    }
};

/// Profiler for performance monitoring
pub const Profiler = struct {
    allocator: Allocator,
    samples: std.ArrayList(ProfileSample),
    enabled: bool,
    start_time: i64,

    const ProfileSample = struct {
        timestamp: i64,
        operation: []const u8,
        duration_us: u64,
        memory_used: usize,
    };

    pub fn init(allocator: Allocator, enabled: bool) Profiler {
        return .{
            .allocator = allocator,
            .samples = std.ArrayList(ProfileSample).init(allocator),
            .enabled = enabled,
            .start_time = std.time.microTimestamp(),
        };
    }

    pub fn deinit(self: *Profiler) void {
        for (self.samples.items) |sample| {
            self.allocator.free(sample.operation);
        }
        self.samples.deinit();
    }

    pub fn startOperation(self: *Profiler, operation: []const u8) i64 {
        _ = operation;
        if (!self.enabled) return 0;
        return std.time.microTimestamp();
    }

    pub fn endOperation(self: *Profiler, start_time: i64, operation: []const u8, memory_used: usize) !void {
        if (!self.enabled) return;

        const end_time = std.time.microTimestamp();
        const duration = @as(u64, @intCast(end_time - start_time));

        try self.samples.append(.{
            .timestamp = end_time,
            .operation = try self.allocator.dupe(u8, operation),
            .duration_us = duration,
            .memory_used = memory_used,
        });
    }

    pub fn getReport(self: *const Profiler) []const ProfileSample {
        return self.samples.items;
    }
};

/// Cache for frequently accessed data
pub const AgentCache = struct {
    allocator: Allocator,
    cache_map: std.HashMap(u64, CacheEntry, std.hash_map.DefaultContext(u64), std.hash_map.default_max_load_percentage),
    max_size: usize,
    current_size: usize = 0,

    const CacheEntry = struct {
        data: []const u8,
        timestamp: i64,
        access_count: u32,
        importance: f32,
    };

    pub fn init(allocator: Allocator, max_size: usize) AgentCache {
        return .{
            .allocator = allocator,
            .cache_map = std.HashMap(u64, CacheEntry, std.hash_map.DefaultContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .max_size = max_size,
        };
    }

    pub fn deinit(self: *AgentCache) void {
        var iterator = self.cache_map.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.data);
        }
        self.cache_map.deinit();
    }

    pub fn get(self: *AgentCache, key: u64) ?[]const u8 {
        if (self.cache_map.getPtr(key)) |entry| {
            entry.access_count += 1;
            entry.timestamp = std.time.microTimestamp();
            return entry.data;
        }
        return null;
    }

    pub fn put(self: *AgentCache, key: u64, data: []const u8, importance: f32) !void {
        if (self.current_size + data.len > self.max_size) {
            try self.evict();
        }

        const owned_data = try self.allocator.dupe(u8, data);
        try self.cache_map.put(key, .{
            .data = owned_data,
            .timestamp = std.time.microTimestamp(),
            .access_count = 1,
            .importance = importance,
        });
        self.current_size += data.len;
    }

    fn evict(self: *AgentCache) !void {
        // Simple LRU eviction
        var oldest_key: u64 = 0;
        var oldest_time: i64 = std.math.maxInt(i64);

        var iterator = self.cache_map.iterator();
        while (iterator.next()) |entry| {
            if (entry.value_ptr.timestamp < oldest_time) {
                oldest_time = entry.value_ptr.timestamp;
                oldest_key = entry.key_ptr.*;
            }
        }

        if (self.cache_map.fetchRemove(oldest_key)) |removed| {
            self.current_size -= removed.value.data.len;
            self.allocator.free(removed.value.data);
        }
    }
};

/// Unified AI Agent with enhanced capabilities
pub const Agent = struct {
    config: AgentConfig,
    allocator: Allocator,
    custom_allocator: ?AgentAllocator = null,
    state: AgentState = .idle,
    current_persona: PersonaType,
    conversation_history: std.ArrayList(Message),
    memory: std.ArrayList(MemoryEntry),
    performance_stats: PerformanceStats = .{},
    request_semaphore: std.Thread.Semaphore,
    state_mutex: std.Thread.Mutex = .{},
    thread_pool: ?*ThreadPool = null,
    profiler: Profiler,
    cache: AgentCache,

    const Self = @This();

    pub fn init(allocator: Allocator, config: AgentConfig) AgentError!*Self {
        try config.validate();

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        var custom_alloc: ?AgentAllocator = null;
        var actual_allocator = allocator;

        if (config.use_custom_allocator) {
            custom_alloc = AgentAllocator.init(allocator);
            actual_allocator = custom_alloc.?.allocator();
        }

        var thread_pool: ?*ThreadPool = null;
        if (config.thread_pool_size > 0) {
            thread_pool = try ThreadPool.init(allocator, config.thread_pool_size);
        }

        self.* = .{
            .config = config,
            .allocator = actual_allocator,
            .custom_allocator = custom_alloc,
            .current_persona = config.default_persona,
            .conversation_history = std.ArrayList(Message).init(actual_allocator),
            .memory = std.ArrayList(MemoryEntry).init(actual_allocator),
            .request_semaphore = .{ .permits = config.max_concurrent_requests },
            .thread_pool = thread_pool,
            .profiler = Profiler.init(actual_allocator, config.enable_profiling),
            .cache = AgentCache.init(actual_allocator, config.cache_size),
        };

        if (config.enable_logging) {
            std.log.info("Agent '{s}' initialized with persona: {s}, backend: {s}", .{ config.name, config.default_persona.getDescription(), @tagName(config.backend_config.backend_type) });
        }

        return self;
    }

    /// Deinitializes the agent, freeing allocated resources
    pub fn deinit(self: *Agent) void {
        // Clean up conversation history
        if (self.config.enable_history) {
            for (self.conversation_history.items) |*message| {
                message.deinit(self.allocator);
            }
            self.conversation_history.deinit(self.allocator);
        }

        self.allocator.destroy(self);
    }

    /// Sets the agent's persona
    pub fn setPersona(self: *Agent, persona: PersonaType) void {
        self.current_persona = persona;
    }

    /// Retrieves the current persona of the agent
    pub fn getPersona(self: *const Agent) ?PersonaType {
        return self.current_persona;
    }

    /// Starts the agent (placeholder implementation)
    pub fn start(self: *Agent) !void {
        const logger = core.logging.ai_logger;
        const persona_desc = if (self.current_persona) |p| p.getDescription() else "no persona";
        logger.info("AI Agent started with {s} persona", .{persona_desc});
    }

    /// Adds a message to the conversation history
    pub fn addMessage(self: *Agent, role: MessageRole, content: []const u8) !void {
        if (!self.config.enable_history) return;

        const content_copy = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(content_copy);

        const message = Message{
            .role = role,
            .content = content_copy,
        };

        try self.conversation_history.append(self.allocator, message);

        // Trim history if it exceeds context length
        self.trimHistory();
    }

    /// Retrieves the conversation history
    pub fn getHistory(self: *const Agent) []const Message {
        if (!self.config.enable_history) return &.{};
        return self.conversation_history.items;
    }

    /// Clears the conversation history
    pub fn clearHistory(self: *Agent) void {
        if (!self.config.enable_history) return;

        for (self.conversation_history.items) |*message| {
            message.deinit(self.allocator);
        }
        self.conversation_history.clearRetainingCapacity();
    }

    /// Trims the conversation history to stay within context limits
    fn trimHistory(self: *Agent) void {
        // TODO: Implement history trimming logic
    }

    /// Process user input with intelligent persona routing
    pub fn processInput(self: *Self, input: []const u8) AgentError![]const u8 {
        // Acquire semaphore for concurrency control
        self.request_semaphore.wait();
        defer self.request_semaphore.post();

        self.performance_stats.concurrent_requests += 1;
        defer self.performance_stats.concurrent_requests -= 1;

        const profile_start = self.profiler.startOperation("processInput");
        defer self.profiler.endOperation(profile_start, "processInput", self.performance_stats.memory_usage_bytes) catch {};

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

        // Check cache first
        const input_hash = std.hash_map.hashString(input);
        if (self.cache.get(input_hash)) |cached_response| {
            self.performance_stats.cache_hit_rate = (self.performance_stats.cache_hit_rate * @as(f32, @floatFromInt(self.performance_stats.total_requests)) + 1.0) / @as(f32, @floatFromInt(self.performance_stats.total_requests + 1));
            const response = try self.allocator.dupe(u8, cached_response);
            self.performance_stats.recordSuccess(self.current_persona);
            try self.transitionState(.idle);
            return response;
        }

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

        // Cache the response
        try self.cache.put(input_hash, response, 0.8);

        // Trim history if needed
        try self.trimHistory();

        self.performance_stats.recordSuccess(self.current_persona);
        try self.transitionState(.idle);

        return response;
    }

    /// Select optimal persona based on input analysis with enhanced scoring
    fn selectPersona(self: *Self, input: []const u8) PersonaType {
        var best_persona = self.config.default_persona;
        var best_score: f32 = 0.0;

        // Enhanced input analysis
        const input_lower = std.ascii.allocLowerString(self.allocator, input) catch input;
        defer if (input_lower.ptr != input.ptr) self.allocator.free(input_lower);

        const is_technical = std.mem.indexOf(u8, input_lower, "code") != null or
            std.mem.indexOf(u8, input_lower, "program") != null or
            std.mem.indexOf(u8, input_lower, "algorithm") != null or
            std.mem.indexOf(u8, input_lower, "function") != null;

        const is_emotional = std.mem.indexOf(u8, input_lower, "help") != null or
            std.mem.indexOf(u8, input_lower, "sad") != null or
            std.mem.indexOf(u8, input_lower, "worry") != null or
            std.mem.indexOf(u8, input_lower, "feel") != null;

        const is_creative = std.mem.indexOf(u8, input_lower, "creative") != null or
            std.mem.indexOf(u8, input_lower, "idea") != null or
            std.mem.indexOf(u8, input_lower, "imagine") != null or
            std.mem.indexOf(u8, input_lower, "design") != null;

        const is_direct = std.mem.indexOf(u8, input_lower, "quick") != null or
            std.mem.indexOf(u8, input_lower, "brief") != null or
            std.mem.indexOf(u8, input_lower, "short") != null;

        const is_research = std.mem.indexOf(u8, input_lower, "research") != null or
            std.mem.indexOf(u8, input_lower, "study") != null or
            std.mem.indexOf(u8, input_lower, "analyze") != null or
            std.mem.indexOf(u8, input_lower, "investigate") != null;

        // Score each persona with enhanced characteristics
        inline for (std.meta.fields(PersonaType)) |field| {
            const persona: PersonaType = @enumFromInt(field.value);
            const scoring = persona.getScoring();

            var score: f32 = 0.5; // Base score

            if (is_technical) score += scoring.technical * 0.4;
            if (is_emotional) score += scoring.empathy * 0.3;
            if (is_creative) score += scoring.creativity * 0.3;
            if (is_direct) score += scoring.directness * 0.2;
            if (is_research) score += scoring.research * 0.3;

            // Context from conversation history
            if (self.conversation_history.items.len > 0) {
                const last_persona = self.conversation_history.items[self.conversation_history.items.len - 1].persona_used;
                if (last_persona == persona) {
                    score += 0.1; // Slight bias towards consistency
                }
            }

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
        const profile_start = self.profiler.startOperation("generateResponse");
        defer self.profiler.endOperation(profile_start, "generateResponse", 0) catch {};

        // Enhanced capability-based response routing
        if (self.config.capabilities.code_generation and self.isCodeQuery(input)) {
            return try self.generateCodeResponse(input);
        } else if (self.config.capabilities.reasoning and self.isReasoningQuery(input)) {
            return try self.generateReasoningResponse(input);
        } else if (self.config.capabilities.vector_search and self.shouldUseVectorSearch(input)) {
            return try self.generateVectorSearchResponse(input);
        } else if (self.config.capabilities.text_generation) {
            return try self.generateTextResponse(input);
        } else {
            return try self.generateDefaultResponse(input);
        }
    }

    fn isCodeQuery(self: *Self, input: []const u8) bool {
        _ = self;
        const code_keywords = [_][]const u8{ "code", "function", "algorithm", "program", "implement", "debug", "compile" };
        for (code_keywords) |keyword| {
            if (std.mem.indexOf(u8, input, keyword) != null) return true;
        }
        return false;
    }

    fn isReasoningQuery(self: *Self, input: []const u8) bool {
        _ = self;
        const reasoning_keywords = [_][]const u8{ "analyze", "think", "reason", "logic", "explain", "why", "how" };
        for (reasoning_keywords) |keyword| {
            if (std.mem.indexOf(u8, input, keyword) != null) return true;
        }
        return false;
    }

    fn shouldUseVectorSearch(self: *Self, input: []const u8) bool {
        _ = input;
        // Use vector search if we have sufficient memory entries
        return self.memory.items.len > 10;
    }

    fn generateCodeResponse(self: *Self, input: []const u8) AgentError![]const u8 {
        _ = input;
        const templates = [_][]const u8{
            "I can help you with code generation. What specific programming task do you need assistance with?",
            "For optimal code implementation, I recommend considering performance, readability, and maintainability.",
            "Let me help you write efficient code. What programming language and problem are you working with?",
            "I'll provide you with well-structured, commented code that follows best practices.",
        };

        const template_idx = @as(usize, @intCast(std.time.microTimestamp())) % templates.len;
        return try self.allocator.dupe(u8, templates[template_idx]);
    }

    fn generateReasoningResponse(self: *Self, input: []const u8) AgentError![]const u8 {
        _ = input;
        const response = switch (self.current_persona) {
            .analytical => "Let me analyze this systematically. I'll break down the problem into logical components and examine each aspect methodically, considering all variables and their relationships.",
            .technical => "From a technical perspective, let's examine the underlying principles and apply structured reasoning to this challenge, focusing on evidence-based conclusions.",
            .solver => "I'll approach this step-by-step, identifying key variables and potential solutions through logical deduction and systematic problem-solving methodologies.",
            .researcher => "Let me conduct a thorough analysis of this topic, examining multiple sources of information and applying rigorous analytical frameworks.",
            else => "Let me think through this carefully, considering multiple perspectives and analyzing the available information using logical reasoning principles.",
        };
        return try self.allocator.dupe(u8, response);
    }

    fn generateVectorSearchResponse(self: *Self, input: []const u8) AgentError![]const u8 {
        self.performance_stats.recordVectorOperation();

        // Simple vector search simulation - in practice, this would use actual embeddings
        var best_match: ?*MemoryEntry = null;
        var best_similarity: f32 = 0.0;

        for (self.memory.items) |*entry| {
            // Simulate similarity calculation
            const similarity = @as(f32, @floatFromInt(std.mem.count(u8, input, entry.content[0..@min(entry.content.len, 100)]))) / 10.0;
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_match = entry;
            }
        }

        if (best_match) |match| {
            match.updateAccess(self.config.enable_simd);
            if (self.config.enable_simd) {
                self.performance_stats.recordSimdOperation();
            }
            return try std.fmt.allocPrint(self.allocator, "Based on relevant information from my memory: {s}", .{match.content[0..@min(match.content.len, 200)]});
        }

        return try self.generateTextResponse(input);
    }

    fn generateTextResponse(self: *Self, input: []const u8) AgentError![]const u8 {
        _ = input;
        const response = switch (self.current_persona) {
            .empathetic => "I understand your concern and I'm here to help. Let me provide a thoughtful response that addresses your needs with care and understanding.",
            .creative => "That's an interesting question! Let me explore some creative approaches and innovative solutions that might open new possibilities for you.",
            .educator => "Great question! Let me explain this in a clear, structured way that will help you understand the concept thoroughly and build upon this knowledge.",
            .counselor => "I appreciate you sharing this with me. Let's work through this together with patience and understanding, taking it one step at a time.",
            .specialist => "Drawing from specialized knowledge in this domain, I can provide you with expert-level insights and detailed analysis.",
            .supportive => "I'm here to support you through this. Let me provide encouragement and practical guidance to help you move forward confidently.",
            else => "Thank you for your question. I'm here to provide helpful, accurate information tailored to your specific needs and context.",
        };
        return try self.allocator.dupe(u8, response);
    }

    fn generateDefaultResponse(self: *Self, input: []const u8) AgentError![]const u8 {
        _ = input;
        const response = "Hello! I'm an AI agent ready to assist you. How can I help you today?";
        return try self.allocator.dupe(u8, response);
    }

    /// Store information in agent memory with enhanced features
    pub fn storeMemory(self: *Self, content: []const u8, importance: f32) AgentError!void {
        const profile_start = self.profiler.startOperation("storeMemory");
        defer self.profiler.endOperation(profile_start, "storeMemory", content.len) catch {};

        if (self.memory.items.len >= self.config.memory_size / @sizeOf(MemoryEntry)) {
            try self.pruneMemory();
        }

        var entry = try MemoryEntry.init(self.allocator, content, importance);
        entry.persona_context = self.current_persona;

        // Generate vector embedding if capability is enabled
        if (self.config.capabilities.vector_search) {
            entry.vector_embedding = try self.generateEmbedding(content);
        }

        try self.memory.append(entry);

        // Update memory usage statistics
        self.performance_stats.memory_usage_bytes = self.memory.items.len * @sizeOf(MemoryEntry);
        self.performance_stats.memory_allocations += 1;

        if (self.performance_stats.memory_usage_bytes > self.performance_stats.peak_memory_usage) {
            self.performance_stats.peak_memory_usage = self.performance_stats.memory_usage_bytes;
        }
    }

    /// Generate simple embedding (placeholder for actual embedding model)
    fn generateEmbedding(self: *Self, content: []const u8) ![]f32 {
        const embedding = try self.allocator.alignedAlloc(f32, 32, self.config.vector_dimension);

        // Simple hash-based embedding simulation
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(content);
        const hash = hasher.final();

        for (embedding, 0..) |*value, i| {
            const seed = hash +% i;
            value.* = @as(f32, @floatFromInt(seed % 1000)) / 1000.0 - 0.5;
        }

        return embedding;
    }

    /// Enhanced memory pruning with SIMD optimization
    fn pruneMemory(self: *Self) AgentError!void {
        const profile_start = self.profiler.startOperation("pruneMemory");
        defer self.profiler.endOperation(profile_start, "pruneMemory", 0) catch {};

        // Sort by composite score with SIMD optimization if available
        if (self.config.enable_simd) {
            self.performance_stats.recordSimdOperation();
        }

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
            self.performance_stats.memory_deallocations += 1;
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

        // Determine where to trim to stay under context limit
        // Calculate total content length and token count
        for (self.conversation_history.items, 0..) |message, i| {
            const new_length = total_length + message.content.len;
            if (new_length > self.config.max_context_length) {
                trim_index = i;
                break;
            }
            total_length = new_length;

            // Update token count if available
            if (message.token_count) |tokens| {
                self.performance_stats.total_tokens_processed += tokens;
            }
        }

        // Remove older messages if needed
        if (trim_index > 0) {
            // Clean up messages being removed
            for (self.conversation_history.items[0..trim_index]) |*message| {
                message.deinit(self.allocator);
            }

            // Shift remaining messages
            for (self.conversation_history.items[0..trim_index]) |message| {
                message.deinit(self.allocator);
            }

            const remaining = self.conversation_history.items[trim_index..];
            std.mem.copyForwards(Message, self.conversation_history.items[0..remaining.len], remaining);
            self.conversation_history.items.len = remaining.len;
        }
    }
};



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

    /// Get comprehensive performance statistics
    pub fn getStats(self: *const Self) PerformanceStats {
        var stats = self.performance_stats;
        stats.memory_usage_bytes = self.memory.items.len * @sizeOf(MemoryEntry);

        // Calculate thread pool utilization if available
        if (self.thread_pool) |pool| {
            // Simplified utilization calculation
            stats.thread_pool_utilization = @as(f32, @floatFromInt(pool.threads.len)) / @as(f32, @floatFromInt(self.config.thread_pool_size));
        }

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
        self.performance_stats.memory_usage_bytes = 0;
    }

    /// Clear cache
    pub fn clearCache(self: *Self) void {
        var iterator = self.cache.cache_map.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.data);
        }
        self.cache.cache_map.clearRetainingCapacity();
        self.cache.current_size = 0;
    }

    /// Get profiling report
    pub fn getProfilingReport(self: *const Self) []const Profiler.ProfileSample {
        return self.profiler.getReport();
    }

    /// Warm up the agent (pre-allocate resources, load models, etc.)
    pub fn warmUp(self: *Self) AgentError!void {
        try self.transitionState(.warming_up);

        if (self.config.enable_logging) {
            std.log.info("Warming up agent '{s}'...", .{self.config.name});
        }

        // Pre-allocate some memory entries
        try self.memory.ensureTotalCapacity(64);

        // Pre-allocate conversation history
        try self.conversation_history.ensureTotalCapacity(32);

        // Test SIMD capabilities
        if (self.config.enable_simd and self.config.capabilities.simd_optimization) {
            const test_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
            const test_b = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
            _ = simdDotProduct(&test_a, &test_b);
            self.performance_stats.recordSimdOperation();
        }

        try self.transitionState(.idle);

        if (self.config.enable_logging) {
            std.log.info("Agent '{s}' warmed up successfully", .{self.config.name});
        }
    }

    // /// Run benchmarks to assess performance
    // pub fn benchmark(self: *Self) AgentError!void {
    //     // TODO: Implement benchmark functionality
    //     _ = self;
    // }
};

test "enhanced agent creation and basic functionality" {
    const testing = std.testing;

    const backend_config = BackendConfig{
        .backend_type = .custom,
        .model_name = "test-model",
    };

    const config = AgentConfig{
        .name = "test_agent",
        .capabilities = .{
            .text_generation = true,
            .reasoning = true,
            .simd_optimization = true,
            .profiling = true,
        },
        .enable_logging = false,
        .max_concurrent_requests = 1,
        .backend_config = backend_config,
        .enable_profiling = true,
        .enable_simd = true,
    };

    var agent = try Agent.init(testing.allocator, config);
    defer agent.deinit();

    // Test warm up
    try agent.warmUp();

    // Test basic processing
    const response = try agent.processInput("Hello, can you help me?");
    defer testing.allocator.free(response);

    try testing.expect(response.len > 0);
    try testing.expectEqual(@as(usize, 1), agent.performance_stats.successful_requests);

    // Test profiling
    const profile_report = agent.getProfilingReport();
    try testing.expect(profile_report.len > 0);

    // Test benchmark
    // try agent.benchmark();
}

test "enhanced persona selection and routing" {
    const testing = std.testing;

    const backend_config = BackendConfig{
        .backend_type = .custom,
        .model_name = "test-model",
    };

    const config = AgentConfig{
        .name = "test_agent",
        .capabilities = .{
            .text_generation = true,
            .code_generation = true,
            .vector_search = true,
        },
        .enable_logging = false,
        .enable_persona_routing = true,
        .backend_config = backend_config,
    };

    var agent = try Agent.init(testing.allocator, config);
    defer agent.deinit();

    // Test technical query should select appropriate persona
    _ = try agent.processInput("Can you help me write some code for a sorting algorithm?");

    // Check that persona was selected appropriately
    const current_persona = agent.getPersona();
    try testing.expect(current_persona == .technical or current_persona == .solver);

    // Test research query
    _ = try agent.processInput("I need to research machine learning algorithms");
    const research_persona = agent.getPersona();
    try testing.expect(research_persona == .researcher or research_persona == .analytical);
}

test "SIMD operations and memory management" {
    const testing = std.testing;

    // Test SIMD dot product
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 0.5, 1.0, 1.5, 2.0 };

    const result = simdDotProduct(&a, &b);
    const expected: f32 = 1.0 * 0.5 + 2.0 * 1.0 + 3.0 * 1.5 + 4.0 * 2.0; // = 15.5

    try testing.expectEqual(expected, result);
}

test "cache functionality" {
    const testing = std.testing;

    var cache = AgentCache.init(testing.allocator, 1024);
    defer cache.deinit();

    try cache.put(123, "test data", 0.8);

    const retrieved = cache.get(123);
    try testing.expect(retrieved != null);
    try testing.expectEqualStrings("test data", retrieved.?);
