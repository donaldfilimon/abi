//! Task cancellation support.
//!
//! Provides cooperative cancellation mechanisms for async operations
//! with hierarchical cancellation (parent cancels children).

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");

/// Common cancellation errors.
pub const CancellationError = error{
    Cancelled,
    CancellationTimeout,
    OperationCancelled,
};

/// Cancellation state.
pub const CancellationState = enum {
    active,
    cancellation_requested,
    cancelled,
};

/// Cancellation reason.
pub const CancellationReason = enum {
    none,
    user_requested,
    timeout,
    parent_cancelled,
    resource_exhausted,
    shutdown,
    error_occurred,
};

/// Cancellation source - creates and controls cancellation tokens.
pub const CancellationSource = struct {
    allocator: std.mem.Allocator,
    state: std.atomic.Value(u8),
    reason: CancellationReason,
    message: ?[]const u8,
    children: std.ArrayListUnmanaged(*CancellationSource),
    parent: ?*CancellationSource,
    mutex: sync.Mutex,
    callbacks: std.ArrayListUnmanaged(CancelCallback),
    /// Monotonic deadline (seconds + nanoseconds) set by `cancelAfter`.
    /// When set, `checkExpired()` compares the current monotonic clock
    /// against this value and triggers cancellation if the deadline has passed.
    deadline_sec: ?i64 = null,
    deadline_nsec: ?i64 = null,

    const CancelCallback = struct {
        callback: *const fn (*CancellationSource, ?*anyopaque) void,
        context: ?*anyopaque,
    };

    /// Create a new cancellation source.
    pub fn init(allocator: std.mem.Allocator) CancellationSource {
        return .{
            .allocator = allocator,
            .state = std.atomic.Value(u8).init(@intFromEnum(CancellationState.active)),
            .reason = .none,
            .message = null,
            .children = .{},
            .parent = null,
            .mutex = .{},
            .callbacks = .{},
        };
    }

    /// Create a child cancellation source.
    pub fn createChild(self: *CancellationSource) !*CancellationSource {
        const child = try self.allocator.create(CancellationSource);
        child.* = CancellationSource.init(self.allocator);
        child.parent = self;

        self.mutex.lock();
        defer self.mutex.unlock();
        try self.children.append(self.allocator, child);

        // If parent is already cancelled, cancel child immediately
        if (self.isCancelled()) {
            child.cancelWithReason(.parent_cancelled, "Parent was cancelled");
        }

        return child;
    }

    /// Deinitialize the cancellation source.
    pub fn deinit(self: *CancellationSource) void {
        // Cancel children first
        for (self.children.items) |child| {
            child.deinit();
            self.allocator.destroy(child);
        }
        self.children.deinit(self.allocator);
        self.callbacks.deinit(self.allocator);

        // Remove from parent
        if (self.parent) |parent| {
            parent.mutex.lock();
            defer parent.mutex.unlock();
            for (parent.children.items, 0..) |child, i| {
                if (child == self) {
                    _ = parent.children.swapRemove(i);
                    break;
                }
            }
        }

        if (self.message) |msg| {
            self.allocator.free(msg);
        }

        self.* = undefined;
    }

    /// Request cancellation.
    pub fn cancel(self: *CancellationSource) void {
        self.cancelWithReason(.user_requested, null);
    }

    /// Request cancellation with a reason.
    pub fn cancelWithReason(self: *CancellationSource, reason: CancellationReason, message: ?[]const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const current = self.getState();
        if (current != .active) return;

        self.state.store(@intFromEnum(CancellationState.cancellation_requested), .release);
        self.reason = reason;

        if (message) |msg| {
            self.message = self.allocator.dupe(u8, msg) catch null;
        }

        // Run callbacks
        for (self.callbacks.items) |cb| {
            cb.callback(self, cb.context);
        }

        // Cancel all children
        for (self.children.items) |child| {
            child.cancelWithReason(.parent_cancelled, "Parent was cancelled");
        }

        self.state.store(@intFromEnum(CancellationState.cancelled), .release);
    }

    /// Schedule cancellation after a timeout (nanoseconds).
    ///
    /// Records a monotonic-clock deadline computed from the current time
    /// plus `timeout_ns`. The cancellation does NOT fire automatically;
    /// callers must periodically invoke `checkExpired()` (or integrate it
    /// into their event loop) to detect that the deadline has passed and
    /// trigger the actual cancellation.
    pub fn cancelAfter(self: *CancellationSource, timeout_ns: u64) void {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);

        const extra_sec: i64 = @intCast(timeout_ns / std.time.ns_per_s);
        const extra_nsec: i64 = @intCast(timeout_ns % std.time.ns_per_s);

        var deadline_nsec = ts.nsec + extra_nsec;
        var deadline_sec = ts.sec + extra_sec;
        if (deadline_nsec >= std.time.ns_per_s) {
            deadline_nsec -= std.time.ns_per_s;
            deadline_sec += 1;
        }

        self.deadline_sec = deadline_sec;
        self.deadline_nsec = deadline_nsec;
    }

    /// Check whether a deadline set by `cancelAfter` has expired.
    ///
    /// Returns `true` and triggers cancellation (with reason `.timeout`)
    /// when the monotonic clock has passed the recorded deadline. Returns
    /// `false` if no deadline is set or it has not yet elapsed.
    pub fn checkExpired(self: *CancellationSource) bool {
        const d_sec = self.deadline_sec orelse return false;
        const d_nsec = self.deadline_nsec orelse return false;

        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);

        if (ts.sec > d_sec or (ts.sec == d_sec and ts.nsec >= d_nsec)) {
            self.cancelWithReason(.timeout, "Timeout expired");
            return true;
        }
        return false;
    }

    /// Get the current state.
    pub fn getState(self: *const CancellationSource) CancellationState {
        return @enumFromInt(self.state.load(.acquire));
    }

    /// Check if cancellation has been requested.
    pub fn isCancellationRequested(self: *const CancellationSource) bool {
        const state = self.getState();
        return state == .cancellation_requested or state == .cancelled;
    }

    /// Check if fully cancelled.
    pub fn isCancelled(self: *const CancellationSource) bool {
        return self.getState() == .cancelled;
    }

    /// Get the cancellation reason.
    pub fn getReason(self: *const CancellationSource) CancellationReason {
        return self.reason;
    }

    /// Get the cancellation message.
    pub fn getMessage(self: *const CancellationSource) ?[]const u8 {
        return self.message;
    }

    /// Get a token for checking cancellation status.
    pub fn getToken(self: *CancellationSource) CancellationToken {
        return CancellationToken.init(self);
    }

    /// Register a callback to be called on cancellation.
    pub fn onCancel(
        self: *CancellationSource,
        callback: *const fn (*CancellationSource, ?*anyopaque) void,
        context: ?*anyopaque,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.isCancelled()) {
            callback(self, context);
            return;
        }

        try self.callbacks.append(self.allocator, .{
            .callback = callback,
            .context = context,
        });
    }

    /// Throw if cancellation requested.
    pub fn throwIfCancellationRequested(self: *const CancellationSource) !void {
        if (self.isCancellationRequested()) {
            return error.OperationCancelled;
        }
    }

    /// Reset the cancellation source.
    pub fn reset(self: *CancellationSource) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.state.store(@intFromEnum(CancellationState.active), .release);
        self.reason = .none;
        if (self.message) |msg| {
            self.allocator.free(msg);
            self.message = null;
        }
    }
};

/// Cancellation token - read-only view for checking cancellation status.
pub const CancellationToken = struct {
    source: ?*CancellationSource,

    /// Create a token from a source.
    pub fn init(source: *CancellationSource) CancellationToken {
        return .{ .source = source };
    }

    /// Create a token that is never cancelled.
    pub fn none() CancellationToken {
        return .{ .source = null };
    }

    /// Check if cancellation has been requested.
    pub fn isCancellationRequested(self: *const CancellationToken) bool {
        if (self.source) |src| {
            return src.isCancellationRequested();
        }
        return false;
    }

    /// Check if fully cancelled.
    pub fn isCancelled(self: *const CancellationToken) bool {
        if (self.source) |src| {
            return src.isCancelled();
        }
        return false;
    }

    /// Get the cancellation reason.
    pub fn getReason(self: *const CancellationToken) CancellationReason {
        if (self.source) |src| {
            return src.getReason();
        }
        return .none;
    }

    /// Throw if cancellation requested.
    pub fn throwIfCancellationRequested(self: *const CancellationToken) !void {
        if (self.isCancellationRequested()) {
            return error.OperationCancelled;
        }
    }

    /// Register a callback.
    pub fn register(
        self: *const CancellationToken,
        callback: *const fn (*CancellationSource, ?*anyopaque) void,
        context: ?*anyopaque,
    ) !void {
        if (self.source) |src| {
            try src.onCancel(callback, context);
        }
    }
};

/// Linked cancellation - links multiple sources together.
pub const LinkedCancellation = struct {
    allocator: std.mem.Allocator,
    source: CancellationSource,
    linked_sources: std.ArrayListUnmanaged(*CancellationSource),

    /// Create a linked cancellation from multiple sources.
    pub fn init(allocator: std.mem.Allocator, sources: []const *CancellationSource) !LinkedCancellation {
        var linked = LinkedCancellation{
            .allocator = allocator,
            .source = CancellationSource.init(allocator),
            .linked_sources = .{},
        };

        for (sources) |src| {
            try linked.linked_sources.append(allocator, src);

            // If any source is already cancelled, cancel this one
            if (src.isCancelled()) {
                linked.source.cancelWithReason(src.getReason(), src.getMessage());
                break;
            }
        }

        return linked;
    }

    pub fn deinit(self: *LinkedCancellation) void {
        self.linked_sources.deinit(self.allocator);
        self.source.deinit();
        self.* = undefined;
    }

    /// Get the combined token.
    pub fn getToken(self: *LinkedCancellation) CancellationToken {
        return self.source.getToken();
    }

    /// Check if any linked source is cancelled.
    pub fn isCancelled(self: *const LinkedCancellation) bool {
        if (self.source.isCancelled()) return true;

        for (self.linked_sources.items) |src| {
            if (src.isCancelled()) return true;
        }
        return false;
    }
};

/// Scoped cancellation - automatically cancels when scope exits.
pub fn ScopedCancellation(comptime cleanup_fn: ?fn (*CancellationSource) void) type {
    return struct {
        source: *CancellationSource,

        const Self = @This();

        pub fn init(source: *CancellationSource) Self {
            return .{ .source = source };
        }

        pub fn deinit(self: *Self) void {
            if (cleanup_fn) |cleanup| {
                cleanup(self.source);
            }
            self.source.cancel();
        }

        pub fn getToken(self: *Self) CancellationToken {
            return self.source.getToken();
        }
    };
}

test "cancellation source basic" {
    const allocator = std.testing.allocator;
    var source = CancellationSource.init(allocator);
    defer source.deinit();

    try std.testing.expect(!source.isCancelled());
    try std.testing.expectEqual(CancellationState.active, source.getState());

    source.cancel();

    try std.testing.expect(source.isCancelled());
    try std.testing.expectEqual(CancellationReason.user_requested, source.getReason());
}

test "cancellation source with reason" {
    const allocator = std.testing.allocator;
    var source = CancellationSource.init(allocator);
    defer source.deinit();

    source.cancelWithReason(.timeout, "Operation timed out");

    try std.testing.expect(source.isCancelled());
    try std.testing.expectEqual(CancellationReason.timeout, source.getReason());
    try std.testing.expectEqualStrings("Operation timed out", source.getMessage().?);
}

test "cancellation token" {
    const allocator = std.testing.allocator;
    var source = CancellationSource.init(allocator);
    defer source.deinit();

    const token = source.getToken();

    try std.testing.expect(!token.isCancelled());

    source.cancel();

    try std.testing.expect(token.isCancelled());
}

test "hierarchical cancellation" {
    const allocator = std.testing.allocator;
    var parent = CancellationSource.init(allocator);
    defer parent.deinit();

    const child = try parent.createChild();

    try std.testing.expect(!child.isCancelled());

    parent.cancel();

    try std.testing.expect(child.isCancelled());
    try std.testing.expectEqual(CancellationReason.parent_cancelled, child.getReason());
}

test "none token" {
    const token = CancellationToken.none();

    try std.testing.expect(!token.isCancelled());
    try std.testing.expect(!token.isCancellationRequested());
}

test {
    std.testing.refAllDecls(@This());
}
