//! Abbey Working Memory
//!
//! Active context for the current conversation.
//! Manages what's "in focus" with attention-based prioritization.

const std = @import("std");
const types = @import("../../core/types.zig");

// ============================================================================
// Working Memory Item
// ============================================================================

/// An item in working memory
pub const WorkingItem = struct {
    id: u64,
    content: []const u8,
    item_type: ItemType,
    importance: f32,
    attention_weight: f32 = 1.0,
    created_at: i64,
    last_accessed: i64,
    access_count: usize = 0,
    decay_rate: f32 = 0.1,
    metadata: ?ItemMetadata = null,

    pub const ItemType = enum {
        user_input,
        assistant_output,
        context,
        goal,
        constraint,
        fact,
        emotion,
        topic,
    };

    pub const ItemMetadata = struct {
        source_id: ?u64 = null,
        emotion: ?types.EmotionType = null,
        confidence: ?f32 = null,
    };

    pub fn deinit(self: *WorkingItem, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
    }

    /// Decay the attention weight
    pub fn decay(self: *WorkingItem, amount: f32) void {
        self.attention_weight = @max(0.01, self.attention_weight - amount * self.decay_rate);
    }

    /// Boost attention (when accessed)
    pub fn boost(self: *WorkingItem, amount: f32) void {
        self.attention_weight = @min(1.0, self.attention_weight + amount);
        self.access_count += 1;
        self.last_accessed = types.getTimestampSec();
    }

    /// Get effective priority (importance * attention)
    pub fn getPriority(self: *const WorkingItem) f32 {
        return self.importance * self.attention_weight;
    }
};

// ============================================================================
// Working Memory
// ============================================================================

/// Active working memory with capacity limits
pub const WorkingMemory = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayListUnmanaged(WorkingItem),
    item_counter: u64 = 0,
    capacity: usize,
    token_budget: usize,
    current_tokens: usize = 0,

    // Focus tracking
    focus_stack: std.ArrayListUnmanaged(u64), // Stack of item IDs
    current_goal: ?u64 = null,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, capacity: usize, token_budget: usize) Self {
        return Self{
            .allocator = allocator,
            .items = .{},
            .capacity = capacity,
            .token_budget = token_budget,
            .focus_stack = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.items.items) |*item| {
            item.deinit(self.allocator);
        }
        self.items.deinit(self.allocator);
        self.focus_stack.deinit(self.allocator);
    }

    /// Add an item to working memory
    pub fn add(self: *Self, content: []const u8, item_type: WorkingItem.ItemType, importance: f32) !u64 {
        // Check capacity
        while (self.items.items.len >= self.capacity) {
            try self.evictLowestPriority();
        }

        // Check token budget
        const tokens = estimateTokens(content);
        while (self.current_tokens + tokens > self.token_budget and self.items.items.len > 0) {
            try self.evictLowestPriority();
        }

        self.item_counter += 1;
        const now = types.getTimestampSec();

        const content_copy = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(content_copy);

        try self.items.append(self.allocator, .{
            .id = self.item_counter,
            .content = content_copy,
            .item_type = item_type,
            .importance = importance,
            .created_at = now,
            .last_accessed = now,
        });

        self.current_tokens += tokens;

        return self.item_counter;
    }

    /// Add with metadata
    pub fn addWithMetadata(
        self: *Self,
        content: []const u8,
        item_type: WorkingItem.ItemType,
        importance: f32,
        metadata: WorkingItem.ItemMetadata,
    ) !u64 {
        const id = try self.add(content, item_type, importance);

        // Find and update metadata
        for (self.items.items) |*item| {
            if (item.id == id) {
                item.metadata = metadata;
                break;
            }
        }

        return id;
    }

    /// Get item by ID
    pub fn get(self: *Self, id: u64) ?*WorkingItem {
        for (self.items.items) |*item| {
            if (item.id == id) {
                item.boost(0.1);
                return item;
            }
        }
        return null;
    }

    /// Get all items of a type
    pub fn getByType(self: *Self, item_type: WorkingItem.ItemType) []const *WorkingItem {
        var results: [64]*WorkingItem = undefined;
        var count: usize = 0;

        for (self.items.items) |*item| {
            if (item.item_type == item_type and count < 64) {
                results[count] = item;
                count += 1;
            }
        }

        return results[0..count];
    }

    /// Get items sorted by priority
    pub fn getByPriority(self: *Self, limit: usize) []const *WorkingItem {
        // Create sorted list
        var sorted: [128]*WorkingItem = undefined;
        const count = @min(self.items.items.len, 128);

        for (self.items.items[0..count], 0..) |*item, i| {
            sorted[i] = item;
        }

        // Sort by priority (descending)
        std.mem.sort(*WorkingItem, sorted[0..count], {}, struct {
            fn lessThan(_: void, a: *WorkingItem, b: *WorkingItem) bool {
                return a.getPriority() > b.getPriority();
            }
        }.lessThan);

        return sorted[0..@min(limit, count)];
    }

    /// Remove item by ID
    pub fn remove(self: *Self, id: u64) bool {
        for (self.items.items, 0..) |*item, i| {
            if (item.id == id) {
                const tokens = estimateTokens(item.content);
                var removed = self.items.orderedRemove(i);
                removed.deinit(self.allocator);
                self.current_tokens -= @min(self.current_tokens, tokens);
                return true;
            }
        }
        return false;
    }

    /// Evict lowest priority item
    fn evictLowestPriority(self: *Self) !void {
        if (self.items.items.len == 0) return;

        var min_priority: f32 = std.math.floatMax(f32);
        var min_idx: usize = 0;

        for (self.items.items, 0..) |item, i| {
            const priority = item.getPriority();
            if (priority < min_priority) {
                min_priority = priority;
                min_idx = i;
            }
        }

        const tokens = estimateTokens(self.items.items[min_idx].content);
        var removed = self.items.orderedRemove(min_idx);
        removed.deinit(self.allocator);
        self.current_tokens -= @min(self.current_tokens, tokens);
    }

    /// Decay all items
    pub fn decayAll(self: *Self, amount: f32) void {
        for (self.items.items) |*item| {
            item.decay(amount);
        }
    }

    /// Push focus to an item
    pub fn pushFocus(self: *Self, id: u64) !void {
        if (self.get(id)) |item| {
            item.boost(0.3);
            try self.focus_stack.append(self.allocator, id);
        }
    }

    /// Pop focus
    pub fn popFocus(self: *Self) ?u64 {
        return self.focus_stack.popOrNull();
    }

    /// Get current focus
    pub fn getCurrentFocus(self: *Self) ?*WorkingItem {
        if (self.focus_stack.items.len == 0) return null;
        const focus_id = self.focus_stack.items[self.focus_stack.items.len - 1];
        return self.get(focus_id);
    }

    /// Set current goal
    pub fn setGoal(self: *Self, goal_content: []const u8) !void {
        // Remove old goal
        if (self.current_goal) |old_id| {
            _ = self.remove(old_id);
        }

        self.current_goal = try self.add(goal_content, .goal, 1.0);
    }

    /// Get current goal
    pub fn getGoal(self: *Self) ?*WorkingItem {
        if (self.current_goal) |id| {
            return self.get(id);
        }
        return null;
    }

    /// Build context string for LLM
    pub fn buildContext(self: *Self, max_tokens: usize) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        var token_count: usize = 0;
        const priority_items = self.getByPriority(64);

        for (priority_items) |item| {
            const item_tokens = estimateTokens(item.content);
            if (token_count + item_tokens > max_tokens) break;

            // Format based on type
            const prefix = switch (item.item_type) {
                .user_input => "[User] ",
                .assistant_output => "[Assistant] ",
                .context => "[Context] ",
                .goal => "[Goal] ",
                .constraint => "[Constraint] ",
                .fact => "[Fact] ",
                .emotion => "[Emotion] ",
                .topic => "[Topic] ",
            };

            try result.appendSlice(self.allocator, prefix);
            try result.appendSlice(self.allocator, item.content);
            try result.appendSlice(self.allocator, "\n");

            token_count += item_tokens;
        }

        return result.toOwnedSlice(self.allocator);
    }

    /// Clear all items
    pub fn clear(self: *Self) void {
        for (self.items.items) |*item| {
            item.deinit(self.allocator);
        }
        self.items.clearRetainingCapacity();
        self.focus_stack.clearRetainingCapacity();
        self.current_tokens = 0;
        self.current_goal = null;
    }

    /// Get statistics
    pub fn getStats(self: *const Self) WorkingStats {
        var total_importance: f32 = 0;
        var total_attention: f32 = 0;

        for (self.items.items) |item| {
            total_importance += item.importance;
            total_attention += item.attention_weight;
        }

        const count = self.items.items.len;
        return .{
            .item_count = count,
            .token_usage = self.current_tokens,
            .token_budget = self.token_budget,
            .avg_importance = if (count > 0) total_importance / @as(f32, @floatFromInt(count)) else 0,
            .avg_attention = if (count > 0) total_attention / @as(f32, @floatFromInt(count)) else 0,
            .focus_depth = self.focus_stack.items.len,
            .has_goal = self.current_goal != null,
        };
    }

    pub const WorkingStats = struct {
        item_count: usize,
        token_usage: usize,
        token_budget: usize,
        avg_importance: f32,
        avg_attention: f32,
        focus_depth: usize,
        has_goal: bool,
    };
};

// ============================================================================
// Utility Functions
// ============================================================================

fn estimateTokens(text: []const u8) usize {
    // Rough estimate: 4 characters per token
    return (text.len + 3) / 4;
}

// ============================================================================
// Tests
// ============================================================================

test "working memory basic" {
    const allocator = std.testing.allocator;

    var memory = WorkingMemory.init(allocator, 100, 4000);
    defer memory.deinit();

    const id = try memory.add("Hello, world!", .user_input, 0.8);
    try std.testing.expect(id > 0);

    const item = memory.get(id);
    try std.testing.expect(item != null);
    try std.testing.expectEqualStrings("Hello, world!", item.?.content);
}

test "working memory priority eviction" {
    const allocator = std.testing.allocator;

    var memory = WorkingMemory.init(allocator, 3, 1000);
    defer memory.deinit();

    _ = try memory.add("Low priority", .context, 0.2);
    _ = try memory.add("Medium priority", .context, 0.5);
    _ = try memory.add("High priority", .context, 0.9);

    // This should evict the lowest priority
    _ = try memory.add("Another high", .context, 0.8);

    try std.testing.expectEqual(@as(usize, 3), memory.items.items.len);
}

test "working memory focus stack" {
    const allocator = std.testing.allocator;

    var memory = WorkingMemory.init(allocator, 100, 4000);
    defer memory.deinit();

    const id1 = try memory.add("Topic 1", .topic, 0.5);
    const id2 = try memory.add("Topic 2", .topic, 0.5);

    try memory.pushFocus(id1);
    try memory.pushFocus(id2);

    const focus = memory.getCurrentFocus();
    try std.testing.expect(focus != null);
    try std.testing.expectEqualStrings("Topic 2", focus.?.content);

    _ = memory.popFocus();
    const new_focus = memory.getCurrentFocus();
    try std.testing.expectEqualStrings("Topic 1", new_focus.?.content);
}

test "working memory context building" {
    const allocator = std.testing.allocator;

    var memory = WorkingMemory.init(allocator, 100, 4000);
    defer memory.deinit();

    _ = try memory.add("User said hello", .user_input, 0.8);
    _ = try memory.add("Assistant replied", .assistant_output, 0.7);

    const context = try memory.buildContext(1000);
    defer allocator.free(context);

    try std.testing.expect(context.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, context, "[User]") != null);
}
