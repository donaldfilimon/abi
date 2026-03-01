//! AI Memory Tests — Short-term, Sliding Window, Token Budget
//!
//! Tests conversation memory management, eviction policies, and token budgeting.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const memory = if (build_options.enable_ai) abi.features.ai.memory else struct {};
const ShortTermMemory = if (build_options.enable_ai) memory.ShortTermMemory else struct {};
const SlidingWindowMemory = if (build_options.enable_ai) memory.SlidingWindowMemory else struct {};
const Message = if (build_options.enable_ai) memory.Message else struct {};

// ============================================================================
// Short-Term Memory Tests
// ============================================================================

test "short-term: add and retrieve messages" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var mem = ShortTermMemory.init(allocator, 10);
    defer mem.deinit();

    try mem.add(Message.user("Hello"));
    try mem.add(Message.assistant("Hi there!"));

    const msgs = mem.getMessages();
    try std.testing.expectEqual(@as(usize, 2), msgs.len);
    try std.testing.expectEqualStrings("Hello", msgs[0].content);
    try std.testing.expectEqualStrings("Hi there!", msgs[1].content);
}

test "short-term: FIFO eviction at capacity" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var mem = ShortTermMemory.init(allocator, 3);
    defer mem.deinit();

    try mem.add(Message.user("first"));
    try mem.add(Message.user("second"));
    try mem.add(Message.user("third"));
    try mem.add(Message.user("fourth"));

    const msgs = mem.getMessages();
    try std.testing.expectEqual(@as(usize, 3), msgs.len);
    // Oldest ("first") should be evicted
    try std.testing.expectEqualStrings("second", msgs[0].content);
    try std.testing.expectEqualStrings("fourth", msgs[2].content);
}

test "short-term: getLastN returns tail" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var mem = ShortTermMemory.init(allocator, 10);
    defer mem.deinit();

    try mem.add(Message.user("a"));
    try mem.add(Message.user("b"));
    try mem.add(Message.user("c"));
    try mem.add(Message.user("d"));

    const last2 = mem.getLastN(2);
    try std.testing.expectEqual(@as(usize, 2), last2.len);
    try std.testing.expectEqualStrings("c", last2[0].content);
    try std.testing.expectEqualStrings("d", last2[1].content);
}

test "short-term: getLastN with n > count returns all" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var mem = ShortTermMemory.init(allocator, 10);
    defer mem.deinit();

    try mem.add(Message.user("only"));

    const all = mem.getLastN(100);
    try std.testing.expectEqual(@as(usize, 1), all.len);
}

test "short-term: getByRole filters correctly" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var mem = ShortTermMemory.init(allocator, 10);
    defer mem.deinit();

    try mem.add(Message.user("question 1"));
    try mem.add(Message.assistant("answer 1"));
    try mem.add(Message.user("question 2"));
    try mem.add(Message.assistant("answer 2"));

    const user_msgs = try mem.getByRole(.user, allocator);
    defer allocator.free(user_msgs);

    try std.testing.expectEqual(@as(usize, 2), user_msgs.len);
    try std.testing.expectEqualStrings("question 1", user_msgs[0].content);
    try std.testing.expectEqualStrings("question 2", user_msgs[1].content);
}

test "short-term: clear removes all messages" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var mem = ShortTermMemory.init(allocator, 10);
    defer mem.deinit();

    try mem.add(Message.user("test"));
    try mem.add(Message.user("test2"));
    mem.clear();

    try std.testing.expect(mem.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), mem.count());
    try std.testing.expectEqual(@as(usize, 0), mem.getMessages().len);
}

test "short-term: stats track utilization" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var mem = ShortTermMemory.init(allocator, 5);
    defer mem.deinit();

    try mem.add(Message.user("a"));
    try mem.add(Message.user("b"));
    try mem.add(Message.user("c"));

    const stats = mem.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.message_count);
    try std.testing.expectEqual(@as(usize, 5), stats.capacity);
    // 3/5 = 0.6
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), stats.utilization, 0.01);
}

test "short-term: token estimation" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var mem = ShortTermMemory.init(allocator, 10);
    defer mem.deinit();

    // "Hello world" = 11 chars → ~3 tokens at 4 chars/token
    try mem.add(Message.user("Hello world"));

    try std.testing.expect(mem.total_tokens > 0);
}

// ============================================================================
// Sliding Window Memory Tests
// ============================================================================

test "sliding window: token-based eviction" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // 20 tokens max
    var win = SlidingWindowMemory.init(allocator, 20);
    defer win.deinit();

    // Each message ~3-5 tokens. Fill until eviction occurs.
    try win.add(Message.user("Hello world")); // ~3 tokens
    try win.add(Message.user("How are you today")); // ~5 tokens
    try win.add(Message.user("I am doing great thanks")); // ~6 tokens
    try win.add(Message.user("This is a very long message that should trigger eviction")); // ~14 tokens

    // Oldest messages should have been evicted to stay within 20 tokens
    const msgs = win.getConversationMessages();
    try std.testing.expect(msgs.len > 0);
    try std.testing.expect(win.current_tokens <= 20);
}

test "sliding window: system message always retained" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var win = SlidingWindowMemory.init(allocator, 30);
    defer win.deinit();

    try win.setSystemMessage(Message.system("You are helpful."));
    try win.add(Message.user("Hello"));
    try win.add(Message.assistant("Hi"));

    const conv = win.getConversationMessages();
    // System message is separate from conversation messages
    try std.testing.expect(win.system_message != null);
    try std.testing.expectEqualStrings("You are helpful.", win.system_message.?.content);
    try std.testing.expect(conv.len >= 1);
}

test "sliding window: system reserve protects budget" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // 30 tokens total, 15 reserved for system
    var win = SlidingWindowMemory.initWithReserve(allocator, 30, 15);
    defer win.deinit();

    // Only 15 tokens available for conversation
    try win.add(Message.user("Short msg")); // ~3 tokens
    try win.add(Message.user("Another short msg")); // ~5 tokens

    // Conversation tokens should be within budget
    const sys_tokens = if (win.system_message) |s| s.estimateTokens() else 0;
    try std.testing.expect(win.current_tokens - sys_tokens <= 30 - 15);
}

test "sliding window: remainingTokens calculation" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var win = SlidingWindowMemory.init(allocator, 100);
    defer win.deinit();

    const initial = win.remainingTokens();
    try std.testing.expectEqual(@as(usize, 100), initial);

    try win.add(Message.user("Hello world")); // ~3 tokens
    try std.testing.expect(win.remainingTokens() < 100);
    try std.testing.expect(win.remainingTokens() > 90);
}

test "sliding window: clear keeps system message" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var win = SlidingWindowMemory.init(allocator, 100);
    defer win.deinit();

    try win.setSystemMessage(Message.system("System prompt"));
    try win.add(Message.user("Hello"));
    try win.add(Message.assistant("Hi"));

    win.clear();

    // System message should still be there
    try std.testing.expect(win.system_message != null);
    // Conversation messages should be gone
    try std.testing.expectEqual(@as(usize, 0), win.getConversationMessages().len);
}

// ============================================================================
// Message Type Tests
// ============================================================================

test "message: factory methods set correct roles" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    const user_msg = Message.user("test");
    try std.testing.expectEqual(memory.MessageRole.user, user_msg.role);

    const asst_msg = Message.assistant("response");
    try std.testing.expectEqual(memory.MessageRole.assistant, asst_msg.role);

    const sys_msg = Message.system("instructions");
    try std.testing.expectEqual(memory.MessageRole.system, sys_msg.role);

    const tool_msg = Message.tool("search", "results");
    try std.testing.expectEqual(memory.MessageRole.tool, tool_msg.role);
    try std.testing.expectEqualStrings("search", tool_msg.name.?);
}

test "message: estimateTokens approximation" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    // 16 chars → 4 tokens at 4 chars/token
    const msg = Message.user("1234567890123456");
    try std.testing.expectEqual(@as(usize, 4), msg.estimateTokens());

    // 3 chars → 1 token (rounds up)
    const short = Message.user("abc");
    try std.testing.expect(short.estimateTokens() >= 1);
}

test "message: clone produces independent copy" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const original = Message.user("Hello world");
    var cloned = try original.clone(allocator);
    defer cloned.deinit(allocator);

    try std.testing.expectEqualStrings("Hello world", cloned.content);
    try std.testing.expectEqual(memory.MessageRole.user, cloned.role);
    // Pointers should be different (independent copy)
    try std.testing.expect(original.content.ptr != cloned.content.ptr);
}
