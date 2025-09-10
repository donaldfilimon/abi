//! Enhanced AI Agent Module
//! Modern AI agent implementation with advanced features and performance optimizations

const std = @import("std");
const Allocator = std.mem.Allocator;

// Create a scoped logger for the AI agent
const log = std.log.scoped(.ai_agent);

/// Agent state management
pub const AgentState = enum {
    idle,
    thinking,
    processing,
    responding,
    learning,
    err,
};

/// Agent capabilities and features
pub const AgentCapabilities = packed struct {
    text_generation: bool = false,
    code_generation: bool = false,
    image_analysis: bool = false,
    audio_processing: bool = false,
    memory_management: bool = false,
    learning: bool = false,
    reasoning: bool = false,
    planning: bool = false,
    _reserved: u24 = 0,
};

/// Agent configuration
pub const AgentConfig = struct {
    name: []const u8,
    model_path: ?[]const u8 = null,
    max_context_length: usize = 4096,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    capabilities: AgentCapabilities = .{},
    memory_size: usize = 1024 * 1024, // 1MB
    enable_logging: bool = true,
    log_level: std.log.Level = .info,
};

/// Memory entry for agent
pub const MemoryEntry = struct {
    id: u64,
    timestamp: i64,
    content: []const u8,
    importance: f32, // 0.0 to 1.0
    tags: std.StringHashMap([]const u8),

    pub fn init(allocator: Allocator, content: []const u8, importance: f32) !MemoryEntry {
        return .{
            .id = @intCast(std.time.microTimestamp()),
            .timestamp = std.time.microTimestamp(),
            .content = try allocator.dupe(u8, content),
            .importance = importance,
            .tags = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *MemoryEntry, allocator: Allocator) void {
        allocator.free(self.content);
        var it = self.tags.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.tags.deinit();
    }

    pub fn addTag(self: *MemoryEntry, allocator: Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        const value_copy = try allocator.dupe(u8, value);
        try self.tags.put(key_copy, value_copy);
    }
};

/// Enhanced AI Agent
pub const EnhancedAgent = struct {
    config: AgentConfig,
    allocator: Allocator,
    state: AgentState,
    memory: std.ArrayList(MemoryEntry),
    context: std.ArrayList(u8),
    capabilities: AgentCapabilities,
    performance_stats: PerformanceStats,

    const Self = @This();

    /// Performance tracking
    pub const PerformanceStats = struct {
        total_requests: u64 = 0,
        total_tokens_processed: u64 = 0,
        average_response_time_ms: f64 = 0.0,
        memory_usage_bytes: usize = 0,
        error_count: u64 = 0,

        pub fn updateResponseTime(self: *PerformanceStats, response_time_ms: f64) void {
            const total = self.total_requests;
            if (total > 0) {
                self.average_response_time_ms = (self.average_response_time_ms * @as(f64, @floatFromInt(total - 1)) + response_time_ms) / @as(f64, @floatFromInt(total));
            } else {
                self.average_response_time_ms = response_time_ms;
            }
        }
    };

    /// Initialize enhanced agent
    pub fn init(allocator: Allocator, config: AgentConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .config = config,
            .allocator = allocator,
            .state = .idle,
            .memory = std.ArrayList(MemoryEntry).init(allocator),
            .context = std.ArrayList(u8).init(allocator),
            .capabilities = config.capabilities,
            .performance_stats = .{},
        };

        if (config.enable_logging) {
            log.info("Enhanced Agent '{s}' initialized", .{config.name});
        }

        return self;
    }

    /// Deinitialize agent
    pub fn deinit(self: *Self) void {
        if (self.config.enable_logging) {
            log.info("Enhanced Agent '{s}' shutting down", .{self.config.name});
        }

        // Free memory entries
        for (self.memory.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.memory.deinit();

        // Free context
        self.context.deinit();

        self.allocator.destroy(self);
    }

    /// Process user input and generate response
    pub fn processInput(self: *Self, input: []const u8) ![]const u8 {
        const start_time = std.time.microTimestamp();
        defer {
            const end_time = std.time.microTimestamp();
            const elapsed = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0; // Convert to milliseconds
            self.performance_stats.updateResponseTime(elapsed);
            self.performance_stats.total_requests += 1;
        }

        self.state = .thinking;
        if (self.config.enable_logging) {
            log.debug("Processing input: {s}", .{input});
        }

        // Store in memory
        try self.storeMemory(input, 0.5);

        // Update context
        try self.updateContext(input);

        // Generate response based on capabilities
        const response = try self.generateResponse(input);

        self.state = .responding;
        if (self.config.enable_logging) {
            log.debug("Generated response: {s}", .{response});
        }

        // Store response in memory
        try self.storeMemory(response, 0.7);

        self.state = .idle;
        return response;
    }

    /// Store information in agent memory
    pub fn storeMemory(self: *Self, content: []const u8, importance: f32) !void {
        if (self.memory.items.len >= self.config.memory_size / 1024) {
            // Remove least important memories
            try self.pruneMemory();
        }

        const entry = try MemoryEntry.init(self.allocator, content, importance);
        try self.memory.append(entry);

        if (self.config.enable_logging) {
            log.debug("Stored memory (importance: {d:.2})", .{importance});
        }
    }

    /// Update agent context
    fn updateContext(self: *Self, input: []const u8) !void {
        // Simple context management - append new input
        try self.context.appendSlice(input);
        try self.context.append('\n');

        // Limit context length
        if (self.context.items.len > self.config.max_context_length) {
            const excess = self.context.items.len - self.config.max_context_length;
            std.mem.copyForwards(u8, self.context.items, self.context.items[excess..]);
            self.context.items.len -= excess;
        }
    }

    /// Generate response based on input and capabilities
    fn generateResponse(self: *Self, input: []const u8) ![]const u8 {
        // Simple response generation - in a real implementation,
        // this would integrate with an LLM or other AI model

        if (self.capabilities.code_generation and std.mem.indexOf(u8, input, "code") != null) {
            return try self.generateCodeResponse(input);
        } else if (self.capabilities.text_generation) {
            return try self.generateTextResponse(input);
        } else {
            return try self.generateDefaultResponse(input);
        }
    }

    /// Generate code-related response
    fn generateCodeResponse(self: *Self, input: []const u8) ![]const u8 {
        _ = input;
        const response = "I can help you with code generation. Please provide more specific requirements.";
        return try self.allocator.dupe(u8, response);
    }

    /// Generate text response
    fn generateTextResponse(self: *Self, input: []const u8) ![]const u8 {
        _ = input;
        const response = "I can help you with text generation and analysis. What would you like me to help with?";
        return try self.allocator.dupe(u8, response);
    }

    /// Generate default response
    fn generateDefaultResponse(self: *Self, input: []const u8) ![]const u8 {
        _ = input;
        const response = "Hello! I'm an enhanced AI agent. How can I assist you today?";
        return try self.allocator.dupe(u8, response);
    }

    /// Prune less important memories
    fn pruneMemory(self: *Self) !void {
        // Sort by importance and remove least important
        std.mem.sort(MemoryEntry, self.memory.items, {}, struct {
            fn lessThan(_: void, a: MemoryEntry, b: MemoryEntry) bool {
                return a.importance < b.importance;
            }
        }.lessThan);

        // Remove bottom 20% of memories
        const remove_count = self.memory.items.len / 5;
        for (0..remove_count) |i| {
            var entry = self.memory.items[i];
            entry.deinit(self.allocator);
        }

        // Shift remaining items
        for (remove_count..self.memory.items.len) |i| {
            self.memory.items[i - remove_count] = self.memory.items[i];
        }
        self.memory.items.len -= remove_count;

        if (self.config.enable_logging) {
            log.debug("Pruned {d} memories", .{remove_count});
        }
    }

    /// Get agent statistics
    pub fn getStats(self: *const Self) PerformanceStats {
        var stats = self.performance_stats;
        stats.memory_usage_bytes = self.memory.items.len * @sizeOf(MemoryEntry);
        return stats;
    }

    /// Search memory by content
    pub fn searchMemory(self: *const Self, query: []const u8) ![]MemoryEntry {
        var results = std.ArrayList(MemoryEntry).init(self.allocator);

        for (self.memory.items) |entry| {
            if (std.mem.indexOf(u8, entry.content, query) != null) {
                try results.append(entry);
            }
        }

        return results.toOwnedSlice();
    }

    /// Learn from interaction
    pub fn learn(self: *Self, input: []const u8, feedback: f32) !void {
        self.state = .learning;

        // Adjust memory importance based on feedback
        for (self.memory.items) |*entry| {
            if (std.mem.indexOf(u8, entry.content, input) != null) {
                entry.importance = @min(1.0, entry.importance + feedback * 0.1);
            }
        }

        if (self.config.enable_logging) {
            log.info("Learned from interaction (feedback: {d:.2})", .{feedback});
        }

        self.state = .idle;
    }
};

test "enhanced agent basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = AgentConfig{
        .name = "test_agent",
        .capabilities = .{ .text_generation = true, .code_generation = true },
        .enable_logging = false,
    };

    var agent = try EnhancedAgent.init(allocator, config);
    defer agent.deinit();

    // Test basic processing
    const response = try agent.processInput("Hello, can you help me with code?");
    defer allocator.free(response);
    try testing.expect(response.len > 0);

    // Test memory storage
    try agent.storeMemory("Important information", 0.9);
    try testing.expect(agent.memory.items.len == 3); // input + response + manual storage

    // Test statistics
    const stats = agent.getStats();
    try testing.expect(stats.total_requests == 1);
    try testing.expect(stats.memory_usage_bytes > 0);
}

test "enhanced agent memory management" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = AgentConfig{
        .name = "memory_test_agent",
        .memory_size = 1024, // Small memory for testing
        .enable_logging = false,
    };

    var agent = try EnhancedAgent.init(allocator, config);
    defer agent.deinit();

    // Add many memories to trigger pruning
    for (0..20) |i| {
        const content = try std.fmt.allocPrint(allocator, "Memory {d}", .{i});
        defer allocator.free(content);
        try agent.storeMemory(content, 0.1 + @as(f32, @floatFromInt(i)) * 0.05);
    }

    // Memory should be pruned
    try testing.expect(agent.memory.items.len < 20);
}
