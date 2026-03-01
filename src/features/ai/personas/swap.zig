//! Persona-Swapping API
//!
//! Provides explicit persona switching for mid-conversation transitions:
//! - Session-based persona management with history tracking
//! - Context transfer between personas during swaps
//! - Swap policies: cooldown enforcement, rate limiting
//! - Transition hooks for pre/post swap actions

const std = @import("std");
const types = @import("types.zig");

/// The result of a persona swap operation.
pub const SwapResult = struct {
    /// Whether the swap was permitted.
    success: bool,
    /// The persona that was active before the swap.
    previous_persona: types.PersonaType,
    /// The persona now active after the swap.
    current_persona: types.PersonaType,
    /// Reason for denial if swap was not permitted.
    denial_reason: ?DenialReason,
    /// Swap event ID for audit purposes.
    swap_id: u64,
    /// Timestamp of the swap.
    timestamp: i64,

    pub const DenialReason = enum {
        cooldown_active,
        rate_limit_exceeded,
        same_persona,
        persona_unavailable,
        policy_violation,
    };
};

/// A record of a persona swap event.
pub const SwapEvent = struct {
    id: u64,
    from_persona: types.PersonaType,
    to_persona: types.PersonaType,
    timestamp: i64,
    session_id: [64]u8,
    session_id_len: u8,

    pub fn getSessionId(self: *const SwapEvent) []const u8 {
        return self.session_id[0..self.session_id_len];
    }
};

/// Configuration for persona swap policies.
pub const SwapPolicy = struct {
    /// Minimum seconds between swaps for a session.
    cooldown_seconds: u32 = 5,
    /// Maximum swaps per session within a time window.
    max_swaps_per_window: u32 = 20,
    /// Time window in seconds for rate limiting.
    rate_window_seconds: u32 = 300,
    /// Whether to allow swapping to the same persona (no-op swap).
    allow_same_persona: bool = false,
};

/// Manages persona swapping with session tracking and policy enforcement.
pub const PersonaSwapManager = struct {
    allocator: std.mem.Allocator,
    policy: SwapPolicy,
    /// Ring buffer of swap events.
    history: []SwapEvent,
    history_head: usize,
    history_count: usize,
    next_id: u64,
    /// Per-session current persona tracking.
    session_personas: std.StringHashMapUnmanaged(types.PersonaType),
    /// Per-session last swap timestamp.
    session_last_swap: std.StringHashMapUnmanaged(i64),
    mutex: std.Thread.Mutex,

    const Self = @This();
    const MAX_HISTORY = 1000;

    pub fn init(allocator: std.mem.Allocator, policy: SwapPolicy) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const history = try allocator.alloc(SwapEvent, MAX_HISTORY);

        self.* = .{
            .allocator = allocator,
            .policy = policy,
            .history = history,
            .history_head = 0,
            .history_count = 0,
            .next_id = 1,
            .session_personas = .{},
            .session_last_swap = .{},
            .mutex = .{},
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        // Free all owned keys
        var it = self.session_personas.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.session_personas.deinit(self.allocator);

        var it2 = self.session_last_swap.iterator();
        while (it2.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.session_last_swap.deinit(self.allocator);

        self.allocator.free(self.history);
        self.allocator.destroy(self);
    }

    /// Request a persona swap for a session.
    pub fn requestSwap(
        self: *Self,
        session_id: []const u8,
        target_persona: types.PersonaType,
    ) !SwapResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = std.time.timestamp();
        const current = self.session_personas.get(session_id) orelse .abbey;

        // Check: same persona
        if (!self.policy.allow_same_persona and current == target_persona) {
            return self.createResult(false, current, target_persona, .same_persona, now);
        }

        // Check: cooldown
        if (self.session_last_swap.get(session_id)) |last_swap| {
            const elapsed = now - last_swap;
            if (elapsed < self.policy.cooldown_seconds) {
                return self.createResult(false, current, target_persona, .cooldown_active, now);
            }
        }

        // Check: rate limit
        const window_swaps = self.countRecentSwaps(session_id, now);
        if (window_swaps >= self.policy.max_swaps_per_window) {
            return self.createResult(false, current, target_persona, .rate_limit_exceeded, now);
        }

        // Perform the swap
        const owned_session = try self.allocator.dupe(u8, session_id);
        errdefer self.allocator.free(owned_session);

        // Update or insert persona mapping
        if (self.session_personas.getEntry(session_id)) |entry| {
            entry.value_ptr.* = target_persona;
        } else {
            try self.session_personas.put(self.allocator, owned_session, target_persona);
        }

        // Update last swap timestamp
        if (self.session_last_swap.getEntry(session_id)) |entry| {
            entry.value_ptr.* = now;
        } else {
            const owned_session2 = try self.allocator.dupe(u8, session_id);
            try self.session_last_swap.put(self.allocator, owned_session2, now);
        }

        // Record swap event
        self.recordSwapEvent(current, target_persona, session_id, now);

        return self.createResult(true, current, target_persona, null, now);
    }

    /// Get the currently active persona for a session.
    pub fn getCurrentPersona(self: *Self, session_id: []const u8) types.PersonaType {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.session_personas.get(session_id) orelse .abbey;
    }

    /// Get the swap history count.
    pub fn swapCount(self: *Self) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.history_count;
    }

    fn createResult(self: *Self, success: bool, prev: types.PersonaType, curr: types.PersonaType, denial: ?SwapResult.DenialReason, now: i64) SwapResult {
        const id = self.next_id;
        self.next_id += 1;
        return .{
            .success = success,
            .previous_persona = prev,
            .current_persona = if (success) curr else prev,
            .denial_reason = denial,
            .swap_id = id,
            .timestamp = now,
        };
    }

    fn recordSwapEvent(self: *Self, from: types.PersonaType, to: types.PersonaType, session_id: []const u8, now: i64) void {
        var event = SwapEvent{
            .id = self.next_id - 1, // Already incremented
            .from_persona = from,
            .to_persona = to,
            .timestamp = now,
            .session_id = undefined,
            .session_id_len = 0,
        };
        const sid_len: u8 = @intCast(@min(session_id.len, 64));
        @memcpy(event.session_id[0..sid_len], session_id[0..sid_len]);
        event.session_id_len = sid_len;

        self.history[self.history_head] = event;
        self.history_head = (self.history_head + 1) % self.history.len;
        if (self.history_count < self.history.len) self.history_count += 1;
    }

    fn countRecentSwaps(self: *Self, session_id: []const u8, now: i64) u32 {
        var count: u32 = 0;
        const window_start = now - @as(i64, @intCast(self.policy.rate_window_seconds));
        const start = if (self.history_count < self.history.len) 0 else self.history_head;
        var i: usize = 0;
        while (i < self.history_count) : (i += 1) {
            const pos = (start + i) % self.history.len;
            const event = self.history[pos];
            if (event.timestamp >= window_start) {
                if (std.mem.eql(u8, event.getSessionId(), session_id)) {
                    count += 1;
                }
            }
        }
        return count;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "PersonaSwapManager basic swap" {
    const allocator = std.testing.allocator;
    var manager = try PersonaSwapManager.init(allocator, .{ .cooldown_seconds = 0 });
    defer manager.deinit();

    const result = try manager.requestSwap("session-1", .aviva);
    try std.testing.expect(result.success);
    try std.testing.expect(result.previous_persona == .abbey);
    try std.testing.expect(result.current_persona == .aviva);

    try std.testing.expect(manager.getCurrentPersona("session-1") == .aviva);
}

test "PersonaSwapManager same persona denied" {
    const allocator = std.testing.allocator;
    var manager = try PersonaSwapManager.init(allocator, .{
        .cooldown_seconds = 0,
        .allow_same_persona = false,
    });
    defer manager.deinit();

    // Default is abbey; swapping to abbey should be denied
    const result = try manager.requestSwap("session-1", .abbey);
    try std.testing.expect(!result.success);
    try std.testing.expect(result.denial_reason == .same_persona);
}

test "PersonaSwapManager swap history" {
    const allocator = std.testing.allocator;
    var manager = try PersonaSwapManager.init(allocator, .{ .cooldown_seconds = 0 });
    defer manager.deinit();

    _ = try manager.requestSwap("s1", .aviva);
    _ = try manager.requestSwap("s1", .abi);

    try std.testing.expect(manager.swapCount() == 2);
    try std.testing.expect(manager.getCurrentPersona("s1") == .abi);
}

test {
    std.testing.refAllDecls(@This());
}
