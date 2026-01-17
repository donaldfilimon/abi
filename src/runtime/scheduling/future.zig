//! Future/Promise pattern for async computation.
//!
//! Provides a Future type that represents a value that will be available
//! at some point, with combinators for composition and transformation.

const std = @import("std");

/// Common future errors
pub const FutureError = std.mem.Allocator.Error || error{
    Cancelled,
    Timeout,
    AlreadyCompleted,
    InvalidState,
};

/// Future state.
pub const FutureState = enum {
    pending,
    completed,
    failed,
    cancelled,
};

/// Future result containing either a value or an error.
pub fn FutureResult(comptime T: type, comptime E: type) type {
    return union(enum) {
        value: T,
        err: E,

        pub fn isOk(self: @This()) bool {
            return self == .value;
        }

        pub fn isErr(self: @This()) bool {
            return self == .err;
        }

        pub fn unwrap(self: @This()) !T {
            return switch (self) {
                .value => |v| v,
                .err => |e| e,
            };
        }

        pub fn unwrapOr(self: @This(), default: T) T {
            return switch (self) {
                .value => |v| v,
                .err => default,
            };
        }
    };
}

/// A Future represents a value that will be available asynchronously.
pub fn Future(comptime T: type) type {
    return struct {
        const Self = @This();
        const Result = FutureResult(T, FutureError);

        allocator: std.mem.Allocator,
        state: std.atomic.Value(u8),
        result: ?Result,
        mutex: std.Thread.Mutex,
        condition: std.Thread.Condition,
        callbacks: std.ArrayListUnmanaged(CallbackEntry),
        cancel_token: ?*CancellationToken,

        const CallbackEntry = struct {
            callback: *const fn (*Self) void,
            context: ?*anyopaque,
        };

        /// Create a new pending future.
        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .state = std.atomic.Value(u8).init(@intFromEnum(FutureState.pending)),
                .result = null,
                .mutex = .{},
                .condition = .{},
                .callbacks = .{},
                .cancel_token = null,
            };
        }

        /// Create a future that is immediately resolved with a value.
        pub fn resolved(allocator: std.mem.Allocator, value: T) Self {
            var f = init(allocator);
            f.result = .{ .value = value };
            f.state.store(@intFromEnum(FutureState.completed), .release);
            return f;
        }

        /// Create a future that is immediately rejected with an error.
        pub fn rejected(allocator: std.mem.Allocator, err: FutureError) Self {
            var f = init(allocator);
            f.result = .{ .err = err };
            f.state.store(@intFromEnum(FutureState.failed), .release);
            return f;
        }

        /// Deinitialize the future.
        pub fn deinit(self: *Self) void {
            self.callbacks.deinit(self.allocator);
            self.* = undefined;
        }

        /// Get the current state.
        pub fn getState(self: *const Self) FutureState {
            return @enumFromInt(self.state.load(.acquire));
        }

        /// Check if the future is complete (resolved, rejected, or cancelled).
        pub fn isComplete(self: *const Self) bool {
            const s = self.getState();
            return s != .pending;
        }

        /// Check if the future is pending.
        pub fn isPending(self: *const Self) bool {
            return self.getState() == .pending;
        }

        /// Resolve the future with a value.
        pub fn resolve(self: *Self, value: T) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.getState() != .pending) return;

            self.result = .{ .value = value };
            self.state.store(@intFromEnum(FutureState.completed), .release);
            self.condition.broadcast();
            self.runCallbacks();
        }

        /// Reject the future with an error.
        pub fn reject(self: *Self, err: FutureError) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.getState() != .pending) return;

            self.result = .{ .err = err };
            self.state.store(@intFromEnum(FutureState.failed), .release);
            self.condition.broadcast();
            self.runCallbacks();
        }

        /// Cancel the future.
        pub fn cancel(self: *Self) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.getState() != .pending) return;

            self.result = .{ .err = error.Cancelled };
            self.state.store(@intFromEnum(FutureState.cancelled), .release);
            self.condition.broadcast();
            self.runCallbacks();
        }

        /// Wait for the future to complete and return the result.
        pub fn await(self: *Self) !T {
            self.mutex.lock();
            defer self.mutex.unlock();

            while (self.getState() == .pending) {
                self.condition.wait(&self.mutex);
            }

            if (self.result) |r| {
                return r.unwrap();
            }
            return error.NoResult;
        }

        /// Wait for the future with a timeout.
        pub fn awaitTimeout(self: *Self, timeout_ns: u64) !?T {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.getState() == .pending) {
                self.condition.timedWait(&self.mutex, timeout_ns) catch |err| {
                    if (err == error.Timeout) return null;
                    return err;
                };
            }

            if (self.getState() == .pending) return null;

            if (self.result) |r| {
                return try r.unwrap();
            }
            return error.NoResult;
        }

        /// Get the result without waiting (returns null if pending).
        pub fn poll(self: *Self) ?Result {
            if (self.getState() == .pending) return null;
            return self.result;
        }

        /// Add a callback to be called when the future completes.
        pub fn onComplete(self: *Self, callback: *const fn (*Self) void) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.isComplete()) {
                callback(self);
                return;
            }

            try self.callbacks.append(self.allocator, .{
                .callback = callback,
                .context = null,
            });
        }

        /// Set a cancellation token.
        pub fn setCancellationToken(self: *Self, token: *CancellationToken) void {
            self.cancel_token = token;
        }

        fn runCallbacks(self: *Self) void {
            for (self.callbacks.items) |entry| {
                entry.callback(self);
            }
            self.callbacks.clearRetainingCapacity();
        }

        /// Map the future's value using a transformation function.
        pub fn map(self: *Self, comptime U: type, transform: *const fn (T) U) Future(U) {
            var mapped = Future(U).init(self.allocator);

            if (self.poll()) |result| {
                switch (result) {
                    .value => |v| mapped.resolve(transform(v)),
                    .err => |e| mapped.reject(e),
                }
            }

            return mapped;
        }

        /// Flat map the future's value.
        pub fn flatMap(self: *Self, comptime U: type, transform: *const fn (T) Future(U)) Future(U) {
            var result_future = Future(U).init(self.allocator);

            if (self.poll()) |result| {
                switch (result) {
                    .value => |v| {
                        var inner = transform(v);
                        if (inner.poll()) |inner_result| {
                            switch (inner_result) {
                                .value => |iv| result_future.resolve(iv),
                                .err => |e| result_future.reject(e),
                            }
                        }
                    },
                    .err => |e| result_future.reject(e),
                }
            }

            return result_future;
        }
    };
}

/// Cancellation token for cooperative cancellation.
pub const CancellationToken = struct {
    cancelled: std.atomic.Value(bool),
    reason: ?[]const u8,

    pub fn init() CancellationToken {
        return .{
            .cancelled = std.atomic.Value(bool).init(false),
            .reason = null,
        };
    }

    pub fn cancel(self: *CancellationToken) void {
        self.cancelled.store(true, .release);
    }

    pub fn cancelWithReason(self: *CancellationToken, reason: []const u8) void {
        self.reason = reason;
        self.cancelled.store(true, .release);
    }

    pub fn isCancelled(self: *const CancellationToken) bool {
        return self.cancelled.load(.acquire);
    }

    pub fn getReason(self: *const CancellationToken) ?[]const u8 {
        return self.reason;
    }

    pub fn reset(self: *CancellationToken) void {
        self.cancelled.store(false, .release);
        self.reason = null;
    }
};

/// Combine multiple futures - wait for all to complete.
pub fn all(comptime T: type, allocator: std.mem.Allocator, futures: []Future(T)) Future([]T) {
    var result = Future([]T).init(allocator);

    // Check if any are already failed
    for (futures) |*f| {
        if (f.poll()) |r| {
            if (r.isErr()) {
                result.reject(r.err);
                return result;
            }
        }
    }

    // Check if all are complete
    var all_complete = true;
    var values = allocator.alloc(T, futures.len) catch {
        result.reject(error.OutOfMemory);
        return result;
    };

    for (futures, 0..) |*f, i| {
        if (f.poll()) |r| {
            if (r.isOk()) {
                values[i] = r.value;
            }
        } else {
            all_complete = false;
        }
    }

    if (all_complete) {
        result.resolve(values);
    }

    return result;
}

/// Combine multiple futures - wait for first to complete.
pub fn race(comptime T: type, allocator: std.mem.Allocator, futures: []Future(T)) Future(T) {
    var result = Future(T).init(allocator);

    for (futures) |*f| {
        if (f.poll()) |r| {
            switch (r) {
                .value => |v| {
                    result.resolve(v);
                    return result;
                },
                .err => |e| {
                    result.reject(e);
                    return result;
                },
            }
        }
    }

    return result;
}

/// Create a future that resolves after a delay.
pub fn delay(allocator: std.mem.Allocator, ns: u64) Future(void) {
    var result = Future(void).init(allocator);

    // In a real implementation, this would use async I/O
    // For now, we just resolve immediately for the API shape
    _ = ns;
    result.resolve({});

    return result;
}

/// Promise - the producer side of a Future.
pub fn Promise(comptime T: type) type {
    return struct {
        const Self = @This();

        future: *Future(T),

        pub fn init(future: *Future(T)) Self {
            return .{ .future = future };
        }

        pub fn resolve(self: *Self, value: T) void {
            self.future.resolve(value);
        }

        pub fn reject(self: *Self, err: FutureError) void {
            self.future.reject(err);
        }

        pub fn isCancelled(self: *const Self) bool {
            if (self.future.cancel_token) |token| {
                return token.isCancelled();
            }
            return false;
        }
    };
}

test "future resolve and await" {
    const allocator = std.testing.allocator;
    var future = Future(i32).init(allocator);
    defer future.deinit();

    future.resolve(42);

    const result = try future.await();
    try std.testing.expectEqual(@as(i32, 42), result);
}

test "future resolved constructor" {
    const allocator = std.testing.allocator;
    var future = Future(i32).resolved(allocator, 100);
    defer future.deinit();

    try std.testing.expect(future.isComplete());
    const result = try future.await();
    try std.testing.expectEqual(@as(i32, 100), result);
}

test "future rejected constructor" {
    const allocator = std.testing.allocator;
    var future = Future(i32).rejected(allocator, error.TestError);
    defer future.deinit();

    try std.testing.expect(future.isComplete());
    try std.testing.expectEqual(FutureState.failed, future.getState());
}

test "future poll" {
    const allocator = std.testing.allocator;
    var future = Future(i32).init(allocator);
    defer future.deinit();

    try std.testing.expect(future.poll() == null);

    future.resolve(42);

    const result = future.poll();
    try std.testing.expect(result != null);
}

test "cancellation token" {
    var token = CancellationToken.init();

    try std.testing.expect(!token.isCancelled());

    token.cancel();
    try std.testing.expect(token.isCancelled());

    token.reset();
    try std.testing.expect(!token.isCancelled());
}
