//! Async runtime built on std.Io.
//!
//! Provides task scheduling, concurrency, and cancellation using the Zig 0.17
//! std.Io Threaded backend.
const std = @import("std");
const builtin = @import("builtin");

// Whether the current target supports threading/OS-level IO
pub const is_threaded_target = builtin.target.os.tag != .freestanding and
    builtin.target.cpu.arch != .wasm32 and
    builtin.target.cpu.arch != .wasm64;

pub const AsyncRuntimeOptions = if (is_threaded_target) std.Io.Threaded.InitOptions else struct {
    concurrent_limit: enum { nothing } = .nothing,
    environ: std.process.Environ = .{},
    async_limit: enum { nothing } = .nothing,
};
pub const AsyncError = if (is_threaded_target) std.Io.ConcurrentError else error{ConcurrencyUnavailable};

pub fn TaskHandle(comptime Result: type) type {
    if (!is_threaded_target) {
        return struct {
            pub fn await(_: anytype) noreturn {
                @compileError("AsyncRuntime unavailable on freestanding targets");
            }
        };
    }
    return struct {
        io: std.Io,
        future: std.Io.Future(Result),

        const Self = @This();

        /// Await the task result.
        /// @return The task result.
        pub fn await(self: *Self) Result {
            return self.future.await(self.io);
        }

        /// Request cancellation and await the task result.
        /// @return The task result or error.Canceled.
        pub fn cancel(self: *Self) Result {
            return self.future.cancel(self.io);
        }
    };
}

pub const TaskGroup = if (!is_threaded_target) struct {} else struct {
    io: std.Io,
    group: std.Io.Group = .init,
    concurrent_enabled: bool,

    /// Initialize a task group bound to a runtime I/O handle.
    /// @param io Runtime I/O handle for scheduling.
    /// @return Initialized task group.
    pub fn init(io: std.Io, concurrent_enabled: bool) TaskGroup {
        return .{
            .io = io,
            .group = .init,
            .concurrent_enabled = concurrent_enabled,
        };
    }

    /// Schedule a task using async semantics (may run inline).
    /// @param function Task entrypoint.
    /// @param args Task arguments.
    pub fn spawn(
        self: *TaskGroup,
        function: anytype,
        args: std.meta.ArgsTuple(@TypeOf(function)),
    ) void {
        self.group.async(self.io, function, args);
    }

    /// Schedule a task using concurrent semantics.
    /// @param function Task entrypoint.
    /// @param args Task arguments.
    /// @return error.ConcurrencyUnavailable if no concurrency is available.
    pub fn spawnConcurrent(
        self: *TaskGroup,
        function: anytype,
        args: std.meta.ArgsTuple(@TypeOf(function)),
    ) AsyncError!void {
        if (!self.concurrent_enabled) return error.ConcurrencyUnavailable;
        return self.group.concurrent(self.io, function, args);
    }

    /// Wait for all tasks in the group to complete.
    /// @return error.Canceled if cancelation is observed.
    pub fn await(self: *TaskGroup) std.Io.Cancelable!void {
        return self.group.await(self.io);
    }

    /// Wait for all tasks, discarding any cancelation error.
    pub fn awaitUncancelable(self: *TaskGroup) void {
        self.group.await(self.io) catch |err| switch (err) {
            error.Canceled => {},
        };
    }

    /// Cancel all tasks in the group.
    pub fn cancel(self: *TaskGroup) void {
        self.group.cancel(self.io);
    }
};

pub const AsyncRuntime = if (!is_threaded_target) struct {
    pub fn init(_: std.mem.Allocator, _: anytype) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
    pub fn ioHandle(_: *@This()) noreturn {
        @compileError("AsyncRuntime unavailable on freestanding targets");
    }
    pub fn taskGroup(_: *@This()) noreturn {
        @compileError("AsyncRuntime unavailable on freestanding targets");
    }
} else struct {
    allocator: std.mem.Allocator,
    backend: std.Io.Threaded,
    io: std.Io,
    concurrent_enabled: bool,

    /// Initialize the async runtime.
    /// @param allocator Memory allocator for runtime allocations.
    /// @param options Threaded I/O options for concurrency limits and stack size.
    /// @return Initialized runtime instance.
    pub fn init(
        allocator: std.mem.Allocator,
        options: AsyncRuntimeOptions,
    ) AsyncRuntime {
        var backend = std.Io.Threaded.init(allocator, options);
        return .{
            .allocator = allocator,
            .backend = backend,
            .io = backend.io(),
            .concurrent_enabled = options.concurrent_limit != .nothing,
        };
    }

    /// Release runtime resources.
    pub fn deinit(self: *AsyncRuntime) void {
        self.backend.deinit();
        self.* = undefined;
    }

    /// Access the underlying std.Io handle.
    /// @return The std.Io handle.
    pub fn ioHandle(self: *AsyncRuntime) std.Io {
        return self.io;
    }

    /// Spawn a task using concurrent scheduling.
    /// @param function Task entrypoint.
    /// @param args Task arguments.
    /// @return A task handle that can await or cancel the task.
    pub fn spawn(
        self: *AsyncRuntime,
        function: anytype,
        args: std.meta.ArgsTuple(@TypeOf(function)),
    ) AsyncError!TaskHandle(
        @typeInfo(@TypeOf(function)).@"fn".return_type.?,
    ) {
        if (!self.concurrent_enabled) return error.ConcurrencyUnavailable;
        const Result = @typeInfo(@TypeOf(function)).@"fn".return_type.?;
        const future = try std.Io.concurrent(self.io, function, args);
        return TaskHandle(Result){ .io = self.io, .future = future };
    }

    /// Spawn a task using async scheduling (may execute inline).
    /// @param function Task entrypoint.
    /// @param args Task arguments.
    /// @return A task handle for the async task.
    pub fn spawnAsync(
        self: *AsyncRuntime,
        function: anytype,
        args: std.meta.ArgsTuple(@TypeOf(function)),
    ) TaskHandle(
        @typeInfo(@TypeOf(function)).@"fn".return_type.?,
    ) {
        const Result = @typeInfo(@TypeOf(function)).@"fn".return_type.?;
        const future = std.Io.async(self.io, function, args);
        return TaskHandle(Result){ .io = self.io, .future = future };
    }

    /// Create a task group bound to this runtime.
    /// @return Initialized task group.
    pub fn taskGroup(self: *AsyncRuntime) TaskGroup {
        return TaskGroup.init(self.io, self.concurrent_enabled);
    }
};

test {
    if (is_threaded_target) {
        std.testing.refAllDecls(@This());
    }
}
