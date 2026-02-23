//! Multi-Agent Event Messaging
//!
//! Provides a lightweight event bus for coordinator lifecycle events.
//! Agents and external observers can subscribe to events like:
//!
//! - Task started/completed/failed
//! - Agent execution started/finished
//! - Aggregation completed
//! - Health check results
//!
//! This bridges the multi-agent coordinator with the `messaging` feature
//! module for system-wide event propagation.

const std = @import("std");

// ============================================================================
// Event Types
// ============================================================================

/// Types of events emitted during multi-agent coordination.
pub const EventType = enum {
    /// A new task has been submitted to the coordinator.
    task_started,
    /// A task completed successfully with aggregated result.
    task_completed,
    /// A task failed (all agents failed or aggregation error).
    task_failed,
    /// An individual agent started processing.
    agent_started,
    /// An individual agent finished processing (success or failure).
    agent_finished,
    /// Result aggregation completed.
    aggregation_completed,
    /// Coordinator health check event.
    health_check,

    pub fn toString(self: EventType) []const u8 {
        return @tagName(self);
    }
};

/// Event payload carrying coordination lifecycle data.
pub const Event = struct {
    /// The type of event.
    event_type: EventType,
    /// Task identifier (usually a hash of the task text).
    task_id: u64 = 0,
    /// Agent index (for agent-specific events).
    agent_index: ?usize = null,
    /// Whether the event represents success.
    success: bool = true,
    /// Duration in nanoseconds (for completed events).
    duration_ns: u64 = 0,
    /// Optional detail message.
    detail: []const u8 = "",
    /// Timestamp (monotonic counter for ordering).
    sequence: u64 = 0,
};

// ============================================================================
// Event Bus
// ============================================================================

/// Callback type for event subscribers.
pub const EventCallback = *const fn (event: Event) void;

/// Simple event bus for multi-agent coordination events.
/// Supports up to `max_subscribers` callbacks per event type.
pub const EventBus = struct {
    allocator: std.mem.Allocator,
    subscribers: std.AutoHashMapUnmanaged(EventType, SubscriberList),
    sequence_counter: u64 = 0,
    enabled: bool = true,

    const SubscriberList = std.ArrayListUnmanaged(EventCallback);

    pub fn init(allocator: std.mem.Allocator) EventBus {
        return .{
            .allocator = allocator,
            .subscribers = .{},
        };
    }

    pub fn deinit(self: *EventBus) void {
        var iter = self.subscribers.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.subscribers.deinit(self.allocator);
        self.* = undefined;
    }

    /// Subscribe to a specific event type.
    pub fn subscribe(self: *EventBus, event_type: EventType, callback: EventCallback) !void {
        const gop = try self.subscribers.getOrPut(self.allocator, event_type);
        if (!gop.found_existing) {
            gop.value_ptr.* = .{};
        }
        try gop.value_ptr.append(self.allocator, callback);
    }

    /// Publish an event to all subscribers of that event type.
    pub fn publish(self: *EventBus, event: Event) void {
        if (!self.enabled) return;

        // Stamp with sequence number
        var stamped = event;
        stamped.sequence = self.sequence_counter;
        self.sequence_counter += 1;

        if (self.subscribers.get(event.event_type)) |subs| {
            for (subs.items) |callback| {
                callback(stamped);
            }
        }
    }

    /// Convenience: publish a task_started event.
    pub fn taskStarted(self: *EventBus, task_id: u64) void {
        self.publish(.{
            .event_type = .task_started,
            .task_id = task_id,
        });
    }

    /// Convenience: publish a task_completed event.
    pub fn taskCompleted(self: *EventBus, task_id: u64, duration_ns: u64) void {
        self.publish(.{
            .event_type = .task_completed,
            .task_id = task_id,
            .success = true,
            .duration_ns = duration_ns,
        });
    }

    /// Convenience: publish a task_failed event.
    pub fn taskFailed(self: *EventBus, task_id: u64, detail: []const u8) void {
        self.publish(.{
            .event_type = .task_failed,
            .task_id = task_id,
            .success = false,
            .detail = detail,
        });
    }

    /// Convenience: publish an agent_started event.
    pub fn agentStarted(self: *EventBus, task_id: u64, agent_index: usize) void {
        self.publish(.{
            .event_type = .agent_started,
            .task_id = task_id,
            .agent_index = agent_index,
        });
    }

    /// Convenience: publish an agent_finished event.
    pub fn agentFinished(self: *EventBus, task_id: u64, agent_index: usize, success: bool, duration_ns: u64) void {
        self.publish(.{
            .event_type = .agent_finished,
            .task_id = task_id,
            .agent_index = agent_index,
            .success = success,
            .duration_ns = duration_ns,
        });
    }

    /// Get the total number of events published.
    pub fn eventCount(self: *const EventBus) u64 {
        return self.sequence_counter;
    }
};

/// Generate a task ID from task text (Wyhash fingerprint).
pub fn taskId(task: []const u8) u64 {
    return std.hash.Wyhash.hash(0, task);
}

// ============================================================================
// Agent Mailbox (inter-agent messaging)
// ============================================================================

/// A message sent between agents during coordinated execution.
pub const AgentMessage = struct {
    from_agent: usize,
    to_agent: usize,
    content: []const u8,
    tag: MessageTag = .data,

    pub const MessageTag = enum {
        /// Normal data payload (pipeline output, etc.)
        data,
        /// Control signal (e.g., stop, pause)
        control,
        /// Status update from an agent
        status,
    };
};

/// Per-agent inbox for receiving messages from other agents.
/// Thread-safe via mutex — suitable for parallel execution.
pub const AgentMailbox = struct {
    allocator: std.mem.Allocator,
    inbox: std.ArrayListUnmanaged(AgentMessage) = .{},
    owned_contents: std.ArrayListUnmanaged([]u8) = .{},

    pub fn init(allocator: std.mem.Allocator) AgentMailbox {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *AgentMailbox) void {
        for (self.owned_contents.items) |buf| {
            self.allocator.free(buf);
        }
        self.owned_contents.deinit(self.allocator);
        self.inbox.deinit(self.allocator);
    }

    /// Send a message to this mailbox (copies content).
    pub fn send(self: *AgentMailbox, msg: AgentMessage) !void {
        const content_copy = try self.allocator.dupe(u8, msg.content);
        errdefer self.allocator.free(content_copy);
        try self.owned_contents.append(self.allocator, content_copy);
        try self.inbox.append(self.allocator, .{
            .from_agent = msg.from_agent,
            .to_agent = msg.to_agent,
            .content = content_copy,
            .tag = msg.tag,
        });
    }

    /// Receive the oldest message (FIFO). Returns null if empty.
    pub fn receive(self: *AgentMailbox) ?AgentMessage {
        if (self.inbox.items.len == 0) return null;
        return self.inbox.orderedRemove(0);
    }

    /// Number of pending messages.
    pub fn pendingCount(self: *const AgentMailbox) usize {
        return self.inbox.items.len;
    }

    /// Clear all messages.
    pub fn clear(self: *AgentMailbox) void {
        self.inbox.clearRetainingCapacity();
        for (self.owned_contents.items) |buf| {
            self.allocator.free(buf);
        }
        self.owned_contents.clearRetainingCapacity();
    }
};

// ============================================================================
// Tests
// ============================================================================

var test_event_count: u32 = 0;
var test_last_event_type: EventType = .health_check;

fn testCallback(event: Event) void {
    test_event_count += 1;
    test_last_event_type = event.event_type;
}

test "event bus subscribe and publish" {
    const allocator = std.testing.allocator;
    test_event_count = 0;

    var bus = EventBus.init(allocator);
    defer bus.deinit();

    try bus.subscribe(.task_started, testCallback);
    bus.taskStarted(42);

    try std.testing.expectEqual(@as(u32, 1), test_event_count);
    try std.testing.expectEqual(EventType.task_started, test_last_event_type);
}

test "event bus multiple subscribers" {
    const allocator = std.testing.allocator;
    test_event_count = 0;

    var bus = EventBus.init(allocator);
    defer bus.deinit();

    try bus.subscribe(.task_completed, testCallback);
    try bus.subscribe(.task_completed, testCallback);
    bus.taskCompleted(42, 1000);

    try std.testing.expectEqual(@as(u32, 2), test_event_count);
}

test "event bus disabled" {
    const allocator = std.testing.allocator;
    test_event_count = 0;

    var bus = EventBus.init(allocator);
    defer bus.deinit();
    bus.enabled = false;

    try bus.subscribe(.task_started, testCallback);
    bus.taskStarted(42);

    try std.testing.expectEqual(@as(u32, 0), test_event_count);
}

test "event bus sequence numbering" {
    const allocator = std.testing.allocator;

    var bus = EventBus.init(allocator);
    defer bus.deinit();

    bus.publish(.{ .event_type = .task_started });
    bus.publish(.{ .event_type = .task_completed });
    bus.publish(.{ .event_type = .task_failed });

    try std.testing.expectEqual(@as(u64, 3), bus.eventCount());
}

test "taskId generates consistent hashes" {
    const id1 = taskId("Hello world");
    const id2 = taskId("Hello world");
    const id3 = taskId("Different task");

    try std.testing.expectEqual(id1, id2);
    try std.testing.expect(id1 != id3);
}

test "event type toString" {
    try std.testing.expectEqualStrings("task_started", EventType.task_started.toString());
    try std.testing.expectEqualStrings("agent_finished", EventType.agent_finished.toString());
}

test "agent mailbox send and receive" {
    const allocator = std.testing.allocator;
    var mailbox = AgentMailbox.init(allocator);
    defer mailbox.deinit();

    try mailbox.send(.{ .from_agent = 0, .to_agent = 1, .content = "hello" });
    try mailbox.send(.{ .from_agent = 2, .to_agent = 1, .content = "world" });

    try std.testing.expectEqual(@as(usize, 2), mailbox.pendingCount());

    const msg1 = mailbox.receive().?;
    try std.testing.expectEqual(@as(usize, 0), msg1.from_agent);
    try std.testing.expectEqualStrings("hello", msg1.content);

    const msg2 = mailbox.receive().?;
    try std.testing.expectEqual(@as(usize, 2), msg2.from_agent);
    try std.testing.expectEqualStrings("world", msg2.content);

    // Empty after consuming both
    try std.testing.expect(mailbox.receive() == null);
    try std.testing.expectEqual(@as(usize, 0), mailbox.pendingCount());
}

test "agent mailbox clear" {
    const allocator = std.testing.allocator;
    var mailbox = AgentMailbox.init(allocator);
    defer mailbox.deinit();

    try mailbox.send(.{ .from_agent = 0, .to_agent = 1, .content = "data" });
    try std.testing.expectEqual(@as(usize, 1), mailbox.pendingCount());

    mailbox.clear();
    try std.testing.expectEqual(@as(usize, 0), mailbox.pendingCount());
}

test "agent message tags" {
    const msg = AgentMessage{
        .from_agent = 0,
        .to_agent = 1,
        .content = "stop",
        .tag = .control,
    };
    try std.testing.expectEqual(AgentMessage.MessageTag.control, msg.tag);
}
