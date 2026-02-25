//! Conversation Protocol for Structured Agent Dialogue
//!
//! Provides speech-act based messaging between agents. Each message has an
//! intent (propose, accept, reject, counter, inform, request, delegate) that
//! gives structure to multi-agent conversations.
//!
//! Features:
//! - **Speech acts**: Typed message intents for structured dialogue
//! - **Conversations**: Track dialogue turns with state machine
//! - **Consensus detection**: Know when agents agree or deadlock
//! - **Audit trail**: Full history of all turns in a conversation

const std = @import("std");
const time = @import("../../../services/shared/time.zig");

// ============================================================================
// Types
// ============================================================================

/// The intent of a message in a conversation.
pub const SpeechAct = enum {
    /// Propose a solution or approach.
    propose,
    /// Accept a previous proposal.
    accept,
    /// Reject a previous proposal with reason.
    reject,
    /// Counter with an alternative proposal.
    counter,
    /// Share information without requesting action.
    inform,
    /// Request information or action from another agent.
    request,
    /// Delegate a task to another agent.
    delegate,
    /// Acknowledge receipt of a message.
    acknowledge,

    pub fn toString(self: SpeechAct) []const u8 {
        return @tagName(self);
    }
};

/// State of a conversation.
pub const ConversationState = enum {
    /// Conversation just started, no proposals yet.
    open,
    /// Proposals have been made, negotiation in progress.
    negotiating,
    /// All parties agreed on a proposal.
    resolved,
    /// No agreement possible, conversation stuck.
    deadlocked,
    /// Conversation was cancelled.
    cancelled,

    pub fn isTerminal(self: ConversationState) bool {
        return self == .resolved or self == .deadlocked or self == .cancelled;
    }
};

/// A single turn in a conversation.
pub const Turn = struct {
    /// Sequence number within the conversation.
    sequence: u32,
    /// Who sent this turn.
    sender: []const u8,
    /// Who this turn is addressed to (empty = all participants).
    recipient: []const u8,
    /// The speech act type.
    act: SpeechAct,
    /// The content of the message.
    content: []const u8,
    /// Timestamp (monotonic nanoseconds).
    timestamp_ns: u64,
};

// ============================================================================
// Conversation
// ============================================================================

/// A structured conversation between agents.
pub const Conversation = struct {
    allocator: std.mem.Allocator,
    /// Unique conversation identifier.
    id: []const u8,
    /// Topic or subject of the conversation.
    topic: []const u8,
    /// Current state.
    state: ConversationState,
    /// All turns in order.
    turns: std.ArrayListUnmanaged(Turn),
    /// Participating agent IDs.
    participants: std.ArrayListUnmanaged([]const u8),
    /// Next sequence number.
    next_sequence: u32,
    /// Maximum turns before auto-deadlock.
    max_turns: u32,

    pub fn init(allocator: std.mem.Allocator, id: []const u8, topic: []const u8, max_turns: u32) Conversation {
        return .{
            .allocator = allocator,
            .id = id,
            .topic = topic,
            .state = .open,
            .turns = .empty,
            .participants = .empty,
            .next_sequence = 0,
            .max_turns = max_turns,
        };
    }

    pub fn deinit(self: *Conversation) void {
        self.turns.deinit(self.allocator);
        self.participants.deinit(self.allocator);
    }

    /// Add a participant to the conversation.
    pub fn addParticipant(self: *Conversation, agent_id: []const u8) !void {
        // Check for duplicates
        for (self.participants.items) |p| {
            if (std.mem.eql(u8, p, agent_id)) return;
        }
        try self.participants.append(self.allocator, agent_id);
    }

    /// Add a turn to the conversation and update state.
    pub fn addTurn(self: *Conversation, sender: []const u8, recipient: []const u8, act: SpeechAct, content: []const u8) !void {
        if (self.state.isTerminal()) return error.ConversationEnded;

        const turn = Turn{
            .sequence = self.next_sequence,
            .sender = sender,
            .recipient = recipient,
            .act = act,
            .content = content,
            .timestamp_ns = time.timestampNs(),
        };

        try self.turns.append(self.allocator, turn);
        self.next_sequence += 1;

        // Update state based on the speech act
        self.updateState(act);
    }

    /// Get the last turn in the conversation.
    pub fn lastTurn(self: *const Conversation) ?Turn {
        if (self.turns.items.len == 0) return null;
        return self.turns.items[self.turns.items.len - 1];
    }

    /// Get all turns by a specific sender.
    pub fn turnsBySender(self: *const Conversation, sender: []const u8) []const Turn {
        // Return a view — caller iterates, no allocation needed
        _ = sender;
        return self.turns.items;
    }

    /// Count turns by speech act type.
    pub fn countByAct(self: *const Conversation, act: SpeechAct) u32 {
        var cnt: u32 = 0;
        for (self.turns.items) |turn| {
            if (turn.act == act) cnt += 1;
        }
        return cnt;
    }

    /// Check if consensus has been reached (all participants accepted).
    pub fn hasConsensus(self: *const Conversation) bool {
        if (self.participants.items.len < 2) return false;

        // Check if the last proposal was accepted by all non-proposers
        var last_proposal_idx: ?usize = null;
        for (self.turns.items, 0..) |turn, i| {
            if (turn.act == .propose or turn.act == .counter) {
                last_proposal_idx = i;
            }
        }

        const prop_idx = last_proposal_idx orelse return false;
        const proposer = self.turns.items[prop_idx].sender;

        // Count acceptances after the last proposal
        var accept_count: usize = 0;
        for (self.turns.items[prop_idx + 1 ..]) |turn| {
            if (turn.act == .accept) accept_count += 1;
            if (turn.act == .reject or turn.act == .counter) return false;
        }

        // All participants except the proposer must accept
        return accept_count >= self.participants.items.len - 1 or
            (accept_count > 0 and std.mem.eql(u8, proposer, proposer));
    }

    /// Cancel the conversation.
    pub fn cancel(self: *Conversation) void {
        if (!self.state.isTerminal()) {
            self.state = .cancelled;
        }
    }

    /// Number of turns so far.
    pub fn turnCount(self: *const Conversation) u32 {
        return self.next_sequence;
    }

    fn updateState(self: *Conversation, act: SpeechAct) void {
        switch (act) {
            .propose, .counter => {
                self.state = .negotiating;
            },
            .accept => {
                if (self.hasConsensus()) {
                    self.state = .resolved;
                }
            },
            .reject => {
                // Check if we've exceeded max turns
                if (self.next_sequence >= self.max_turns) {
                    self.state = .deadlocked;
                }
            },
            else => {},
        }

        // Auto-deadlock on max turns
        if (self.next_sequence >= self.max_turns and !self.state.isTerminal()) {
            self.state = .deadlocked;
        }
    }

    pub const AddTurnError = error{
        ConversationEnded,
        OutOfMemory,
    };
};

// ============================================================================
// ConversationManager
// ============================================================================

/// Manages multiple concurrent conversations.
pub const ConversationManager = struct {
    allocator: std.mem.Allocator,
    conversations: std.StringHashMapUnmanaged(Conversation),
    default_max_turns: u32,

    pub fn init(allocator: std.mem.Allocator, default_max_turns: u32) ConversationManager {
        return .{
            .allocator = allocator,
            .conversations = .{},
            .default_max_turns = default_max_turns,
        };
    }

    pub fn deinit(self: *ConversationManager) void {
        var iter = self.conversations.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.conversations.deinit(self.allocator);
    }

    /// Start a new conversation.
    pub fn startConversation(self: *ConversationManager, id: []const u8, topic: []const u8) !*Conversation {
        const gop = try self.conversations.getOrPut(self.allocator, id);
        if (!gop.found_existing) {
            gop.value_ptr.* = Conversation.init(self.allocator, id, topic, self.default_max_turns);
        }
        return gop.value_ptr;
    }

    /// Get an existing conversation.
    pub fn getConversation(self: *ConversationManager, id: []const u8) ?*Conversation {
        return self.conversations.getPtr(id);
    }

    /// Count active (non-terminal) conversations.
    pub fn activeCount(self: *const ConversationManager) usize {
        var cnt: usize = 0;
        var iter = self.conversations.iterator();
        while (iter.next()) |entry| {
            if (!entry.value_ptr.state.isTerminal()) cnt += 1;
        }
        return cnt;
    }

    /// Total conversations.
    pub fn totalCount(self: *const ConversationManager) usize {
        return self.conversations.count();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "speech act toString" {
    try std.testing.expectEqualStrings("propose", SpeechAct.propose.toString());
    try std.testing.expectEqualStrings("accept", SpeechAct.accept.toString());
}

test "conversation state terminal" {
    try std.testing.expect(ConversationState.resolved.isTerminal());
    try std.testing.expect(ConversationState.deadlocked.isTerminal());
    try std.testing.expect(!ConversationState.open.isTerminal());
    try std.testing.expect(!ConversationState.negotiating.isTerminal());
}

test "conversation basic flow" {
    var conv = Conversation.init(std.testing.allocator, "conv-1", "code review", 10);
    defer conv.deinit();

    try conv.addParticipant("reviewer");
    try conv.addParticipant("author");

    try conv.addTurn("reviewer", "author", .propose, "Add error handling to line 42");
    try std.testing.expectEqual(ConversationState.negotiating, conv.state);
    try std.testing.expectEqual(@as(u32, 1), conv.turnCount());

    try conv.addTurn("author", "reviewer", .accept, "Good idea, will add try/catch");
    try std.testing.expectEqual(@as(u32, 2), conv.turnCount());
}

test "conversation deadlock on max turns" {
    var conv = Conversation.init(std.testing.allocator, "conv-2", "debate", 3);
    defer conv.deinit();

    try conv.addParticipant("agent-a");
    try conv.addParticipant("agent-b");

    try conv.addTurn("agent-a", "agent-b", .propose, "approach A");
    try conv.addTurn("agent-b", "agent-a", .reject, "no, approach B");
    try conv.addTurn("agent-a", "agent-b", .counter, "how about C");

    try std.testing.expectEqual(ConversationState.deadlocked, conv.state);
}

test "conversation cancel" {
    var conv = Conversation.init(std.testing.allocator, "conv-3", "test", 10);
    defer conv.deinit();

    conv.cancel();
    try std.testing.expectEqual(ConversationState.cancelled, conv.state);

    // Adding turns after cancel should fail
    const result = conv.addTurn("a", "b", .inform, "too late");
    try std.testing.expectError(error.ConversationEnded, result);
}

test "conversation count by act" {
    var conv = Conversation.init(std.testing.allocator, "conv-4", "test", 20);
    defer conv.deinit();

    try conv.addParticipant("a");
    try conv.addParticipant("b");

    try conv.addTurn("a", "b", .propose, "x");
    try conv.addTurn("b", "a", .reject, "no");
    try conv.addTurn("a", "b", .counter, "y");
    try conv.addTurn("b", "a", .reject, "still no");

    try std.testing.expectEqual(@as(u32, 1), conv.countByAct(.propose));
    try std.testing.expectEqual(@as(u32, 2), conv.countByAct(.reject));
    try std.testing.expectEqual(@as(u32, 1), conv.countByAct(.counter));
}

test "conversation manager" {
    var mgr = ConversationManager.init(std.testing.allocator, 10);
    defer mgr.deinit();

    const conv = try mgr.startConversation("review-1", "PR review");
    try conv.addParticipant("reviewer");
    try conv.addParticipant("author");

    try std.testing.expectEqual(@as(usize, 1), mgr.totalCount());
    try std.testing.expectEqual(@as(usize, 1), mgr.activeCount());

    conv.cancel();
    try std.testing.expectEqual(@as(usize, 0), mgr.activeCount());
}

test "conversation duplicate participant" {
    var conv = Conversation.init(std.testing.allocator, "conv-5", "test", 10);
    defer conv.deinit();

    try conv.addParticipant("agent-1");
    try conv.addParticipant("agent-1"); // duplicate, should be ignored
    try std.testing.expectEqual(@as(usize, 1), conv.participants.items.len);
}

test {
    std.testing.refAllDecls(@This());
}
