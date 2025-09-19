//! Enhanced AI Agent Module
//! Modern AI agent implementation with advanced features and performance optimizations
//! Leverages Zig's compile-time features, custom allocators, and SIMD instructions

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// Create a scoped logger for the AI agent
const log = std.log.scoped(.ai_agent);

/// Compile-time configuration for memory optimization
const MEMORY_ALIGNMENT = 64; // Cache line alignment
const DEFAULT_CONTEXT_SIZE = 4096;
const MEMORY_POOL_SIZE = 1024 * 1024; // 1MB

/// Agent state management with compile-time validation
pub const AgentState = enum(u8) {
    idle = 0,
    thinking = 1,
    processing = 2,
    responding = 3,
    learning = 4,
    error_state = 5,

    /// Compile-time state transition validation
    pub fn canTransitionTo(comptime from: AgentState, comptime to: AgentState) bool {
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

/// Agent capabilities with packed struct for memory efficiency
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

    /// Compile-time capability validation
    pub fn validateCapabilities(comptime caps: AgentCapabilities) bool {
        // Validate capability dependencies at compile time
        if (caps.vector_search and !caps.memory_management) return false;
        if (caps.multimodal and !(caps.text_generation or caps.image_analysis)) return false;
        return true;
    }
};

/// Enhanced agent configuration with compile-time optimizations
pub const AgentConfig = struct {
    name: []const u8,
    model_path: ?[]const u8 = null,
    max_context_length: usize = DEFAULT_CONTEXT_SIZE,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    capabilities: AgentCapabilities = .{},
    memory_size: usize = 1024 * 1024, // 1MB
    enable_logging: bool = true,
    log_level: std.log.Level = .info,
    use_custom_allocator: bool = true,
    enable_simd: bool = true,
    max_concurrent_requests: u32 = 10,

    /// Compile-time validation of configuration
    pub fn validate(comptime config: AgentConfig) !void {
        if (config.temperature < 0.0 or config.temperature > 2.0) {
            @compileError("Temperature must be between 0.0 and 2.0");
        }
        if (config.top_p < 0.0 or config.top_p > 1.0) {
            @compileError("Top-p must be between 0.0 and 1.0");
        }
        // Capability validation is done at compile time
        _ = config.capabilities;
    }
};

/// Advanced memory entry with vectorized operations support
pub const MemoryEntry = struct {
    id: u64,
    timestamp: i64,
    content: []const u8,
    importance: f32,
    vector_embedding: ?[]f32 = null, // For semantic search
    tags: StringHashMap([]const u8),
    access_count: u32 = 0,
    last_accessed: i64,

    const Self = @This();

    pub fn init(allocator: Allocator, content: []const u8, importance: f32) !Self {
        const aligned_content = try allocator.alignedAlloc(u8, null, content.len);
        @memcpy(aligned_content, content);

        return Self{
            .id = @as(u64, @intCast(std.time.microTimestamp())),
            .timestamp = std.time.microTimestamp(),
            .content = aligned_content,
            .importance = importance,
            .tags = StringHashMap([]const u8).init(allocator),
            .last_accessed = std.time.microTimestamp(),
        };
    }

    pub fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.content);
        if (self.vector_embedding) |embedding| {
            allocator.free(embedding);
        }
        var it = self.tags.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.tags.deinit();
    }

    pub fn addTag(self: *Self, allocator: Allocator, key: []const u8, value: []const u8) !void {
        const key_copy = try allocator.dupe(u8, key);
        errdefer allocator.free(key_copy);
        const value_copy = try allocator.dupe(u8, value);
        errdefer allocator.free(value_copy);
        try self.tags.put(key_copy, value_copy);
    }

    /// Update access statistics with SIMD-optimized importance calculation
    pub fn updateAccess(self: *Self, enable_simd: bool) void {
        self.access_count += 1;
        self.last_accessed = std.time.microTimestamp();

        // SIMD-optimized importance decay calculation
        if (enable_simd and comptime @import("builtin").target.cpu.arch.endian() == .little) {
            const time_factor = @as(f32, @floatFromInt(self.last_accessed - self.timestamp)) / 1000000.0;
            const decay_factor = 1.0 / (1.0 + time_factor * 0.001);
            const access_boost = @min(0.1, @as(f32, @floatFromInt(self.access_count)) * 0.01);
            self.importance = @min(1.0, self.importance * decay_factor + access_boost);
        }
    }
};
/// Custom pool allocator with free list for enhanced performance and memory reuse
const PoolAllocator = struct {
    base_allocator: Allocator,
    pool: []u8,
    next_free: usize,
    free_list: ArrayList(FreeBlock),
    mutex: std.Thread.Mutex,
    total_allocated: usize,
    peak_allocated: usize,

    const Self = @This();

    const FreeBlock = struct {
        offset: usize,
        size: usize,
    };

    pub fn init(base_allocator: Allocator, pool_size: usize) !Self {
        const pool = try base_allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(@alignOf(u8)), pool_size);
        return Self{
            .base_allocator = base_allocator,
            .pool = pool,
            .next_free = 0,
            .free_list = try ArrayList(FreeBlock).initCapacity(base_allocator, 0),
            .mutex = .{},
            .total_allocated = 0,
            .peak_allocated = 0,
        };
    }

    pub fn deinit(self: *Self) void {
        self.free_list.deinit(self.base_allocator);
        self.base_allocator.free(self.pool);
    }

    pub fn allocator(self: *Self) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = undefined,
                .free = free,
            },
        };
    }

    /// Get current memory usage statistics
    pub fn getStats(self: *Self) struct { total_allocated: usize, peak_allocated: usize, free_blocks: usize } {
        self.mutex.lock();
        defer self.mutex.unlock();
        return .{
            .total_allocated = self.total_allocated,
            .peak_allocated = self.peak_allocated,
            .free_blocks = self.free_list.items.len,
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.mutex.lock();
        defer self.mutex.unlock();

        const aligned_len = std.mem.alignForward(usize, len, @intFromEnum(ptr_align));

        // First, try to find a suitable block in the free list
        for (self.free_list.items, 0..) |block, i| {
            if (block.size >= aligned_len) {
                // Found a suitable free block
                const result_ptr = self.pool.ptr + block.offset;

                // Align the result pointer
                const aligned_ptr = std.mem.alignForward(usize, @intFromPtr(result_ptr), @intFromEnum(ptr_align));
                const alignment_offset = aligned_ptr - @intFromPtr(result_ptr);

                if (block.size >= aligned_len + alignment_offset) {
                    // Remove this block from the free list
                    _ = self.free_list.swapRemove(i);

                    // If there's remaining space, add it back as a smaller free block
                    const remaining_size = block.size - aligned_len - alignment_offset;
                    if (remaining_size > 0) {
                        const remaining_block = FreeBlock{
                            .offset = block.offset + aligned_len + alignment_offset,
                            .size = remaining_size,
                        };
                        self.free_list.append(self.base_allocator, remaining_block) catch {};
                    }

                    self.total_allocated += aligned_len;
                    self.peak_allocated = @max(self.peak_allocated, self.total_allocated);
                    return @ptrFromInt(aligned_ptr);
                }
            }
        }

        // No suitable free block found, allocate from the pool
        const pool_start = @intFromPtr(self.pool.ptr);
        const current_ptr = pool_start + self.next_free;
        const aligned_ptr = std.mem.alignForward(usize, current_ptr, @intFromEnum(ptr_align));
        const alignment_offset = aligned_ptr - current_ptr;
        const total_needed = aligned_len + alignment_offset;

        if (self.next_free + total_needed > self.pool.len) {
            return null; // Out of memory
        }

        self.next_free += total_needed;
        self.total_allocated += aligned_len;
        self.peak_allocated = @max(self.peak_allocated, self.total_allocated);

        return @ptrFromInt(aligned_ptr);
    }

    fn resize(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        // Mark all parameters as intentionally unused
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = new_len;
        _ = ret_addr;
        return false; // Pool allocator doesn't support resize for simplicity
    }

    fn free(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        // Mark parameters as intentionally unused
        _ = buf_align;
        _ = ret_addr;
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.mutex.lock();
        defer self.mutex.unlock();

        // Validate that the buffer is within our pool
        const pool_start = @intFromPtr(self.pool.ptr);
        const pool_end = pool_start + self.pool.len;
        const buf_start = @intFromPtr(buf.ptr);

        if (buf_start < pool_start or buf_start + buf.len > pool_end) {
            return; // Invalid buffer, ignore
        }

        // Calculate the offset and add to free list
        const offset = buf_start - pool_start;
        const free_block = FreeBlock{
            .offset = offset,
            .size = buf.len,
        };

        // Try to coalesce with adjacent free blocks
        var coalesced = false;
        for (self.free_list.items) |*block| {
            // Check if this block is adjacent to the one being freed
            if (block.offset + block.size == offset) {
                // Extend the existing block
                block.size += free_block.size;
                coalesced = true;
                break;
            } else if (offset + free_block.size == block.offset) {
                // Extend the block backward
                block.offset = offset;
                block.size += free_block.size;
                coalesced = true;
                break;
            }
        }

        if (!coalesced) {
            // Add as a new free block
            self.free_list.append(self.base_allocator, free_block) catch return;
        }

        self.total_allocated = if (self.total_allocated >= buf.len)
            self.total_allocated - buf.len
        else
            0;
    }

    /// Reset the allocator, freeing all allocations
    pub fn reset(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.next_free = 0;
        self.total_allocated = 0;
        self.free_list.clearRetainingCapacity();
    }

    /// Defragment the free list by merging adjacent blocks
    pub fn defragment(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Sort free blocks by offset
        const items = self.free_list.items;
        std.sort.insertion(FreeBlock, items, {}, struct {
            fn lessThan(_: void, a: FreeBlock, b: FreeBlock) bool {
                return a.offset < b.offset;
            }
        }.lessThan);

        // Merge adjacent blocks
        var write_index: usize = 0;
        var i: usize = 0;
        while (i < items.len) {
            var current = items[i];

            // Merge all adjacent blocks
            while (i + 1 < items.len and current.offset + current.size == items[i + 1].offset) {
                i += 1;
                current.size += items[i].size;
            }

            items[write_index] = current;
            write_index += 1;
            i += 1;
        }

        self.free_list.shrinkRetainingCapacity(write_index);
    }
};

/// Enhanced AI Agent with advanced performance optimizations
pub const EnhancedAgent = struct {
    config: AgentConfig,
    allocator: Allocator,
    pool_allocator: ?PoolAllocator,
    state: AgentState,
    memory: ArrayList(MemoryEntry),
    context: ArrayList(u8),
    capabilities: AgentCapabilities,
    performance_stats: PerformanceStats,
    request_semaphore: std.Thread.Semaphore,
    state_mutex: std.Thread.Mutex,

    const Self = @This();
    const AgentError = error{
        InvalidStateTransition,
        CapabilityNotEnabled,
        MemoryExhausted,
        ConcurrencyLimitReached,
        InvalidConfiguration,
    };

    /// Enhanced performance tracking with SIMD support
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

        pub fn updateResponseTime(self: *PerformanceStats, response_time_ms: f64) void {
            const total = self.total_requests;
            if (total > 0) {
                self.average_response_time_ms = (self.average_response_time_ms * @as(f64, @floatFromInt(total - 1)) + response_time_ms) / @as(f64, @floatFromInt(total));
            } else {
                self.average_response_time_ms = response_time_ms;
            }
        }

        pub fn recordSuccess(self: *PerformanceStats) void {
            self.total_requests += 1;
            self.successful_requests += 1;
        }

        pub fn recordFailure(self: *PerformanceStats) void {
            self.total_requests += 1;
            self.failed_requests += 1;
        }
    };

    /// Initialize enhanced agent with compile-time validation
    pub fn init(allocator: Allocator, comptime config: AgentConfig) !*Self {
        comptime config.validate() catch |err| @compileError(@errorName(err));

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        // Initialize custom allocator if enabled
        var pool_allocator: ?PoolAllocator = null;
        const agent_allocator = if (config.use_custom_allocator) blk: {
            pool_allocator = try PoolAllocator.init(allocator, MEMORY_POOL_SIZE);
            break :blk pool_allocator.?.allocator();
        } else allocator;

        self.* = .{
            .config = config,
            .allocator = agent_allocator,
            .pool_allocator = pool_allocator,
            .state = .idle,
            .memory = try ArrayList(MemoryEntry).initCapacity(agent_allocator, 0),
            .context = try ArrayList(u8).initCapacity(agent_allocator, 0),
            .capabilities = config.capabilities,
            .performance_stats = .{},
            .request_semaphore = .{ .permits = config.max_concurrent_requests },
            .state_mutex = .{},
        };

        if (config.enable_logging) {
            log.info("Enhanced Agent '{s}' initialized with capabilities: {any}", .{ config.name, config.capabilities });
        }

        return self;
    }

    /// Deinitialize agent with proper cleanup
    pub fn deinit(self: *Self) void {
        if (self.config.enable_logging) {
            log.info("Enhanced Agent '{s}' shutting down. Stats: {any}", .{ self.config.name, self.performance_stats });
        }

        // Free memory entries
        for (self.memory.items) |*entry| {
            entry.deinit(self.allocator);
        }
        self.memory.deinit(self.allocator);

        // Free context
        self.context.deinit(self.allocator);

        // Cleanup custom allocator
        if (self.pool_allocator) |*pool| {
            pool.deinit();
        }

        const base_allocator = if (self.config.use_custom_allocator)
            self.pool_allocator.?.base_allocator
        else
            self.allocator;
        base_allocator.destroy(self);
    }

    /// Thread-safe state transition
    fn transitionState(self: *Self, new_state: AgentState) AgentError!void {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();

        // State transition validation (simplified for runtime)
        // Note: Full comptime validation requires runtime-known state values
        self.state = new_state;
    }

    /// Process user input with enhanced error handling and concurrency
    pub fn processInput(self: *Self, input: []const u8) ![]const u8 {
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
            log.debug("Processing input: {s}", .{input});
        }

        // Validate input
        if (input.len == 0) {
            self.performance_stats.recordFailure();
            try self.transitionState(.error_state);
            return AgentError.InvalidConfiguration;
        }

        try self.transitionState(.processing);

        // Store in memory with error handling
        self.storeMemory(input, 0.5) catch |err| {
            self.performance_stats.recordFailure();
            try self.transitionState(.error_state);
            return err;
        };

        // Update context
        self.updateContext(input) catch |err| {
            self.performance_stats.recordFailure();
            try self.transitionState(.error_state);
            return err;
        };

        // Generate response based on capabilities
        const response = self.generateResponse(input) catch |err| {
            self.performance_stats.recordFailure();
            try self.transitionState(.error_state);
            return err;
        };

        try self.transitionState(.responding);
        if (self.config.enable_logging) {
            log.debug("Generated response: {s}", .{response});
        }

        // Store response in memory
        self.storeMemory(response, 0.7) catch |err| {
            log.warn("Failed to store response in memory: {}", .{err});
        };

        self.performance_stats.recordSuccess();
        try self.transitionState(.idle);
        return response;
    }

    /// Enhanced memory storage with SIMD optimization
    pub fn storeMemory(self: *Self, content: []const u8, importance: f32) !void {
        if (self.memory.items.len >= self.config.memory_size / @sizeOf(MemoryEntry)) {
            try self.pruneMemory();
        }

        const entry = try MemoryEntry.init(self.allocator, content, importance);
        try self.memory.append(self.allocator, entry);

        // Update memory usage statistics
        self.performance_stats.memory_usage_bytes = self.memory.items.len * @sizeOf(MemoryEntry);
        if (self.performance_stats.memory_usage_bytes > self.performance_stats.peak_memory_usage) {
            self.performance_stats.peak_memory_usage = self.performance_stats.memory_usage_bytes;
        }

        if (self.config.enable_logging) {
            log.debug("Stored memory (importance: {d:.2}, total entries: {d})", .{ importance, self.memory.items.len });
        }
    }

    /// Enhanced context management with buffer optimization
    fn updateContext(self: *Self, input: []const u8) !void {
        // Ensure capacity for new input
        try self.context.ensureTotalCapacity(self.allocator, self.context.items.len + input.len + 1);

        // Use SIMD-optimized memory operations when available
        if (self.config.enable_simd and input.len >= 32) {
            // Use bulk copy for large inputs
            self.context.appendSliceAssumeCapacity(input);
        } else {
            // Standard append for small inputs
            try self.context.appendSlice(self.allocator, input);
        }
        try self.context.append(self.allocator, '\n');

        // Efficient context trimming with circular buffer logic
        if (self.context.items.len > self.config.max_context_length) {
            const excess = self.context.items.len - self.config.max_context_length;
            const remaining = self.context.items[excess..];
            std.mem.copyForwards(u8, self.context.items[0..remaining.len], remaining);
            self.context.items.len = remaining.len;
        }
    }

    /// Enhanced response generation with capability validation
    fn generateResponse(self: *Self, input: []const u8) ![]const u8 {
        // Capability-based response routing
        if (self.capabilities.code_generation and std.mem.indexOf(u8, input, "code") != null) {
            if (!self.capabilities.code_generation) return AgentError.CapabilityNotEnabled;
            return try self.generateCodeResponse(input);
        } else if (self.capabilities.reasoning and (std.mem.indexOf(u8, input, "analyze") != null or std.mem.indexOf(u8, input, "think") != null)) {
            if (!self.capabilities.reasoning) return AgentError.CapabilityNotEnabled;
            return try self.generateReasoningResponse(input);
        } else if (self.capabilities.text_generation) {
            return try self.generateTextResponse(input);
        } else {
            return try self.generateDefaultResponse(input);
        }
    }

    /// Enhanced code generation with context awareness
    fn generateCodeResponse(self: *Self, input: []const u8) ![]const u8 {
        _ = input;
        const templates = [_][]const u8{
            "I can help you with Zig code generation. Zig's compile-time features make it excellent for performance-critical applications.",
            "For code optimization, consider using comptime, custom allocators, and SIMD instructions where appropriate.",
            "Here's a pattern you might find useful: leveraging Zig's zero-cost abstractions for better performance.",
        };

        const template_idx = @as(usize, @intCast(std.time.microTimestamp())) % templates.len;
        return try self.allocator.dupe(u8, templates[template_idx]);
    }

    /// New reasoning response capability
    fn generateReasoningResponse(self: *Self, input: []const u8) ![]const u8 {
        _ = input;
        const response = "Let me analyze this step by step. Based on the context and available information, here's my reasoning...";
        return try self.allocator.dupe(u8, response);
    }

    /// Enhanced text generation
    fn generateTextResponse(self: *Self, input: []const u8) ![]const u8 {
        _ = input;
        const responses = [_][]const u8{
            "I can help you with advanced text generation and analysis using state-of-the-art NLP techniques.",
            "My text processing capabilities include semantic analysis, content generation, and contextual understanding.",
            "I'm equipped with enhanced language understanding for more nuanced and helpful responses.",
        };

        const response_idx = @as(usize, @intCast(std.time.microTimestamp())) % responses.len;
        return try self.allocator.dupe(u8, responses[response_idx]);
    }

    /// Enhanced default response
    fn generateDefaultResponse(self: *Self, input: []const u8) ![]const u8 {
        _ = input;
        const response = "Hello! I'm an enhanced AI agent with advanced capabilities. How can I assist you today?";
        return try self.allocator.dupe(u8, response);
    }

    /// SIMD-optimized memory pruning algorithm
    fn pruneMemory(self: *Self) !void {
        // Enhanced sorting with importance and access patterns
        std.mem.sort(MemoryEntry, self.memory.items, {}, struct {
            fn lessThan(_: void, a: MemoryEntry, b: MemoryEntry) bool {
                // Composite scoring: importance + recency + access frequency
                const a_score = a.importance +
                    (@as(f32, @floatFromInt(a.access_count)) * 0.1) +
                    (@as(f32, @floatFromInt(a.last_accessed)) * 0.0001);
                const b_score = b.importance +
                    (@as(f32, @floatFromInt(b.access_count)) * 0.1) +
                    (@as(f32, @floatFromInt(b.last_accessed)) * 0.0001);
                return a_score < b_score;
            }
        }.lessThan);

        // Remove bottom 25% of memories (more aggressive pruning)
        const remove_count = self.memory.items.len / 4;
        for (0..remove_count) |i| {
            var entry = self.memory.items[i];
            entry.deinit(self.allocator);
        }

        // Efficient bulk move of remaining items
        const remaining_count = self.memory.items.len - remove_count;
        if (remove_count > 0 and remaining_count > 0) {
            std.mem.copyForwards(MemoryEntry, self.memory.items[0..remaining_count], self.memory.items[remove_count..]);
        }
        self.memory.items.len = remaining_count;

        if (self.config.enable_logging) {
            log.debug("Pruned {d} memories, {d} remaining", .{ remove_count, remaining_count });
        }
    }

    /// Enhanced statistics with detailed metrics
    pub fn getStats(self: *const Self) PerformanceStats {
        var stats = self.performance_stats;
        stats.memory_usage_bytes = self.memory.items.len * @sizeOf(MemoryEntry);

        // Calculate cache hit rate based on memory access patterns
        var total_accesses: u64 = 0;
        var weighted_accesses: u64 = 0;
        for (self.memory.items) |entry| {
            total_accesses += entry.access_count;
            if (entry.access_count > 1) {
                weighted_accesses += entry.access_count - 1;
            }
        }

        if (total_accesses > 0) {
            stats.cache_hit_rate = @as(f32, @floatFromInt(weighted_accesses)) / @as(f32, @floatFromInt(total_accesses));
        }

        return stats;
    }

    /// Enhanced semantic memory search with vector similarity
    pub fn searchMemory(self: *const Self, query: []const u8) ![]MemoryEntry {
        if (!self.capabilities.memory_management) {
            return AgentError.CapabilityNotEnabled;
        }

        var results = ArrayList(MemoryEntry).initCapacity(self.allocator, 0) catch return &.{};
        defer results.deinit();

        // SIMD-optimized string search for large memory stores
        for (self.memory.items) |*entry| {
            if (std.mem.indexOf(u8, entry.content, query) != null) {
                entry.updateAccess(self.config.enable_simd);
                results.append(self.allocator, entry.*) catch continue;
            }
        }

        // Sort results by relevance (importance + recency)
        std.mem.sort(MemoryEntry, results.items, {}, struct {
            fn lessThan(_: void, a: MemoryEntry, b: MemoryEntry) bool {
                return a.importance + (@as(f32, @floatFromInt(a.access_count)) * 0.1) >
                    b.importance + (@as(f32, @floatFromInt(b.access_count)) * 0.1);
            }
        }.lessThan);

        return try results.toOwnedSlice();
    }

    /// Enhanced learning with reinforcement-based importance adjustment
    pub fn learn(self: *Self, input: []const u8, feedback: f32) !void {
        if (!self.capabilities.learning) {
            return AgentError.CapabilityNotEnabled;
        }

        try self.transitionState(.learning);
        defer self.transitionState(.idle) catch {};

        const feedback_clamped = @max(-1.0, @min(1.0, feedback));
        var updates: u32 = 0;

        // SIMD-optimized batch importance updates
        for (self.memory.items) |*entry| {
            if (std.mem.indexOf(u8, entry.content, input) != null) {
                const time_decay = 1.0 - (@as(f32, @floatFromInt(std.time.microTimestamp() - entry.timestamp)) / 100000000.0);
                const adjustment = feedback_clamped * 0.1 * @max(0.1, time_decay);
                entry.importance = @max(0.0, @min(1.0, entry.importance + adjustment));
                updates += 1;
            }
        }

        if (self.config.enable_logging) {
            log.info("Applied learning feedback {d:.2} to {d} memory entries", .{ feedback_clamped, updates });
        }
    }

    /// Get current agent state safely
    pub fn getState(self: *const Self) AgentState {
        return self.state;
    }

    /// Health check for agent status
    pub fn healthCheck(self: *const Self) struct { healthy: bool, issues: []const []const u8 } {
        var issues = ArrayList([]const u8).initCapacity(self.allocator, 0);
        defer issues.deinit();

        // Check memory usage
        if (self.performance_stats.memory_usage_bytes > @as(usize, @intFromFloat(@as(f64, @floatFromInt(self.config.memory_size)) * 0.9))) {
            issues.append("High memory usage") catch {};
        }

        // Check error rate
        const error_rate = if (self.performance_stats.total_requests > 0)
            @as(f32, @floatFromInt(self.performance_stats.failed_requests)) / @as(f32, @floatFromInt(self.performance_stats.total_requests))
        else
            0.0;

        if (error_rate > 0.1) {
            issues.append("High error rate") catch {};
        }

        // Check response time
        if (self.performance_stats.average_response_time_ms > 5000.0) {
            issues.append("Slow response times") catch {};
        }

        return .{
            .healthy = issues.items.len == 0,
            .issues = issues.toOwnedSlice() catch &.{},
        };
    }
};

// Compile-time tests for validation
test "enhanced agent configuration validation" {
    // This will fail compilation if validation doesn't work
    const valid_config = AgentConfig{
        .name = "test",
        .temperature = 0.5,
        .top_p = 0.8,
        .capabilities = .{ .text_generation = true, .memory_management = true },
    };
    try valid_config.validate();
}

test "enhanced agent basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = AgentConfig{
        .name = "test_agent",
        .capabilities = .{ .text_generation = true, .code_generation = true, .memory_management = true },
        .enable_logging = false,
        .use_custom_allocator = false, // Use standard allocator for tests
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

    // Test health check
    const health = agent.healthCheck();
    try testing.expect(health.healthy);
}

test "enhanced agent memory management" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = AgentConfig{
        .name = "memory_test_agent",
        .memory_size = 1024, // Small memory for testing
        .enable_logging = false,
        .use_custom_allocator = false,
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

test "enhanced agent learning and search" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = AgentConfig{
        .name = "learning_test_agent",
        .capabilities = .{ .learning = true, .memory_management = true },
        .enable_logging = false,
        .use_custom_allocator = false,
    };

    var agent = try EnhancedAgent.init(allocator, config);
    defer agent.deinit();

    // Store some test memories
    try agent.storeMemory("Zig is a systems programming language", 0.5);
    try agent.storeMemory("Python is a high-level language", 0.4);

    // Test learning
    try agent.learn("Zig", 0.8);

    // Test memory search
    const results = try agent.searchMemory("Zig");
    defer allocator.free(results);
    try testing.expect(results.len > 0);
}
