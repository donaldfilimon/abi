//! Async event loop for real-time TUI updates.
//!
//! Provides non-blocking event handling with periodic refresh callbacks,
//! enabling dynamic updates and real-time metrics display.

const std = @import("std");
const abi = @import("abi");
const shared_time = abi.shared.time;
const terminal_mod = @import("terminal.zig");
const Terminal = terminal_mod.Terminal;
const events = @import("events.zig");
const RingBuffer = @import("ring_buffer.zig").RingBuffer;

/// Event types for the async loop
pub const AsyncEvent = union(enum) {
    /// User input event
    input: events.Event,
    /// Timer tick for periodic refresh
    tick: u64,
    /// External update request
    update: UpdateRequest,
    /// Resize event
    resize: struct { rows: u16, cols: u16 },
    /// Quit request
    quit: void,
};

/// Update request for external data changes
pub const UpdateRequest = struct {
    widget_id: u32,
    data: ?*anyopaque,
};

/// Callback function types
pub const RenderFn = *const fn (*AsyncLoop) anyerror!void;
pub const UpdateFn = *const fn (*AsyncLoop, AsyncEvent) anyerror!bool;
pub const TickFn = *const fn (*AsyncLoop) anyerror!void;

/// Configuration for the async loop
pub const AsyncLoopConfig = struct {
    /// Refresh rate in milliseconds
    refresh_rate_ms: u32 = 100,
    /// Input poll timeout in milliseconds
    input_poll_ms: u32 = 16,
    /// Enable automatic resize handling
    auto_resize: bool = true,
    /// Maximum events per frame
    max_events_per_frame: u32 = 16,
};

/// Async TUI event loop with real-time updates
pub const AsyncLoop = struct {
    allocator: std.mem.Allocator,
    terminal: *Terminal,
    config: AsyncLoopConfig,

    // State
    running: bool,
    frame_count: u64,
    last_refresh_time: i64,
    last_tick_time: i64,
    last_rows: u16,
    last_cols: u16,
    has_last_size: bool,

    // Event queue for external updates
    event_queue: std.ArrayListUnmanaged(AsyncEvent),

    // Callbacks
    render_fn: ?RenderFn,
    update_fn: ?UpdateFn,
    tick_fn: ?TickFn,

    // User data pointer
    user_data: ?*anyopaque,

    // Metrics
    fps: f32,
    frame_times: [60]i64,
    frame_time_idx: usize,

    pub fn init(allocator: std.mem.Allocator, terminal: *Terminal, config: AsyncLoopConfig) AsyncLoop {
        return .{
            .allocator = allocator,
            .terminal = terminal,
            .config = config,
            .running = false,
            .frame_count = 0,
            .last_refresh_time = 0,
            .last_tick_time = 0,
            .last_rows = 0,
            .last_cols = 0,
            .has_last_size = false,
            .event_queue = .empty,
            .render_fn = null,
            .update_fn = null,
            .tick_fn = null,
            .user_data = null,
            .fps = 0,
            .frame_times = [_]i64{0} ** 60,
            .frame_time_idx = 0,
        };
    }

    pub fn deinit(self: *AsyncLoop) void {
        self.event_queue.deinit(self.allocator);
    }

    /// Set the render callback
    pub fn setRenderCallback(self: *AsyncLoop, callback: RenderFn) void {
        self.render_fn = callback;
    }

    /// Set the update/event handler callback
    pub fn setUpdateCallback(self: *AsyncLoop, callback: UpdateFn) void {
        self.update_fn = callback;
    }

    /// Set the tick callback (called on each refresh interval)
    pub fn setTickCallback(self: *AsyncLoop, callback: TickFn) void {
        self.tick_fn = callback;
    }

    /// Set user data pointer
    pub fn setUserData(self: *AsyncLoop, data: *anyopaque) void {
        self.user_data = data;
    }

    /// Get user data pointer
    pub fn getUserData(self: *AsyncLoop, comptime T: type) ?*T {
        if (self.user_data) |ptr| {
            return @ptrCast(@alignCast(ptr));
        }
        return null;
    }

    /// Push an external update event
    pub fn pushUpdate(self: *AsyncLoop, widget_id: u32, data: ?*anyopaque) void {
        self.event_queue.append(self.allocator, .{
            .update = .{
                .widget_id = widget_id,
                .data = data,
            },
        }) catch |err| {
            std.log.warn("Failed to queue update for widget {d}: {t}", .{ widget_id, err });
        };
    }

    /// Request a quit. Sets running=false directly to guarantee
    /// the quit succeeds even under memory pressure.
    pub fn requestQuit(self: *AsyncLoop) void {
        self.running = false;
    }

    /// Get current FPS
    pub fn getFps(self: *AsyncLoop) f32 {
        return self.fps;
    }

    /// Get frame count
    pub fn getFrameCount(self: *AsyncLoop) u64 {
        return self.frame_count;
    }

    /// Run the async event loop
    pub fn run(self: *AsyncLoop) !void {
        self.running = true;
        self.last_refresh_time = currentTimeMs();
        self.last_tick_time = self.last_refresh_time;
        if (self.config.auto_resize) {
            const size = self.terminal.size();
            self.last_rows = size.rows;
            self.last_cols = size.cols;
            self.has_last_size = true;
        } else {
            self.has_last_size = false;
        }

        while (self.running) {
            const frame_start = currentTimeMs();

            // Check terminal resize changes first so layout-dependent handlers
            // can react before processing other events.
            try self.processResizeEvents();

            // Process input events (non-blocking)
            try self.processInputEvents();

            // Process queued events
            try self.processQueuedEvents();

            // Check if refresh is needed
            const elapsed = frame_start - self.last_refresh_time;
            if (elapsed >= self.config.refresh_rate_ms) {
                // Call tick callback
                if (self.tick_fn) |tick| {
                    try tick(self);
                }

                // Render frame
                if (self.render_fn) |render| {
                    try render(self);
                }

                // Flush all buffered output in a single syscall
                try self.terminal.flush();

                self.last_refresh_time = frame_start;
                self.frame_count += 1;

                // Update FPS calculation
                self.updateFps(frame_start);
            }

            // Small sleep to prevent busy-waiting
            shared_time.sleepMs(1);
        }
    }

    /// Stop the loop
    pub fn stop(self: *AsyncLoop) void {
        self.running = false;
    }

    fn processInputEvents(self: *AsyncLoop) !void {
        // Non-blocking poll for input
        var events_processed: u32 = 0;
        while (events_processed < self.config.max_events_per_frame) {
            // Try to read event with timeout — propagate real errors
            const maybe_event = self.terminal.pollEvent(
                self.config.input_poll_ms,
            ) catch |err| {
                // Terminal disconnected or fatal I/O error — stop the loop
                std.log.warn("Terminal poll error: {t}", .{err});
                self.running = false;
                return;
            };
            const event = maybe_event orelse break; // Timeout — no more events

            const async_event = AsyncEvent{ .input = event };

            // Handle quit events
            if (event == .key) {
                if (event.key.code == .ctrl_c) {
                    self.running = false;
                    return;
                }
            }

            // Call update callback
            if (self.update_fn) |update| {
                const should_quit = try update(self, async_event);
                if (should_quit) {
                    self.running = false;
                    return;
                }
            }

            events_processed += 1;
        }
    }

    fn processQueuedEvents(self: *AsyncLoop) !void {
        if (self.event_queue.items.len == 0) return;

        // Swap to local copy and drain — O(n) instead of O(n^2)
        var pending = self.event_queue;
        self.event_queue = .empty;
        defer pending.deinit(self.allocator);

        for (pending.items) |event| {
            switch (event) {
                .quit => {
                    self.running = false;
                    return;
                },
                else => {
                    if (self.update_fn) |update| {
                        const should_quit = try update(self, event);
                        if (should_quit) {
                            self.running = false;
                            return;
                        }
                    }
                },
            }
        }
    }

    fn processResizeEvents(self: *AsyncLoop) !void {
        if (!self.config.auto_resize or !self.has_last_size) return;

        // Check the SIGWINCH flag for immediate resize detection (POSIX only).
        // On non-POSIX platforms, we still fall through to the size-polling path below.
        if (terminal_mod.resize_pending) {
            terminal_mod.resize_pending = false;
            // Flag was set — force a size re-read below.
        }

        const size = self.terminal.size();
        if (size.rows == self.last_rows and size.cols == self.last_cols) {
            return;
        }

        self.last_rows = size.rows;
        self.last_cols = size.cols;

        if (self.update_fn) |update| {
            const should_quit = try update(self, .{
                .resize = .{
                    .rows = size.rows,
                    .cols = size.cols,
                },
            });
            if (should_quit) {
                self.running = false;
            }
        }
    }

    fn updateFps(self: *AsyncLoop, current_time: i64) void {
        self.frame_times[self.frame_time_idx] = current_time;
        self.frame_time_idx = (self.frame_time_idx + 1) % 60;

        // Calculate FPS from frame times
        const oldest_idx = self.frame_time_idx;
        const oldest_time = self.frame_times[oldest_idx];
        if (oldest_time > 0) {
            const elapsed_ms = current_time - oldest_time;
            if (elapsed_ms > 0) {
                self.fps = 60.0 * 1000.0 / @as(f32, @floatFromInt(elapsed_ms));
            }
        }
    }

    fn currentTimeMs() i64 {
        return shared_time.nowMs();
    }
};

/// Real-time metrics tracker for system monitoring
pub const MetricsTracker = struct {
    cpu_samples: RingBuffer(f32, 120),
    memory_samples: RingBuffer(f32, 120),
    gpu_samples: RingBuffer(f32, 120),
    network_rx_samples: RingBuffer(u64, 120),
    network_tx_samples: RingBuffer(u64, 120),

    last_update: i64,
    update_interval_ms: i64,

    pub fn init() MetricsTracker {
        return .{
            .cpu_samples = RingBuffer(f32, 120).init(),
            .memory_samples = RingBuffer(f32, 120).init(),
            .gpu_samples = RingBuffer(f32, 120).init(),
            .network_rx_samples = RingBuffer(u64, 120).init(),
            .network_tx_samples = RingBuffer(u64, 120).init(),
            .last_update = 0,
            .update_interval_ms = 1000,
        };
    }

    pub fn update(self: *MetricsTracker) void {
        const now = shared_time.nowMs();
        if (now - self.last_update < self.update_interval_ms) return;

        // Sample system metrics
        self.cpu_samples.push(getCpuUsage());
        self.memory_samples.push(getMemoryUsage());

        // GPU metrics if available
        if (getGpuUsage()) |gpu_val| {
            self.gpu_samples.push(gpu_val);
        }

        self.last_update = now;
    }

    pub fn getCpuHistory(self: *const MetricsTracker, buf: []f32) []f32 {
        return self.cpu_samples.toSlice(buf);
    }

    pub fn getMemoryHistory(self: *const MetricsTracker, buf: []f32) []f32 {
        return self.memory_samples.toSlice(buf);
    }

    pub fn getGpuHistory(self: *const MetricsTracker, buf: []f32) []f32 {
        return self.gpu_samples.toSlice(buf);
    }

    fn getCpuUsage() f32 {
        // Placeholder - would use platform-specific APIs
        return 0.0;
    }

    fn getMemoryUsage() f32 {
        // Placeholder - would use platform-specific APIs
        return 0.0;
    }

    fn getGpuUsage() ?f32 {
        // Placeholder - would query GPU backend
        return null;
    }
};

// Tests
test "AsyncLoop basic initialization" {
    const allocator = std.testing.allocator;
    var terminal = Terminal.init(allocator);
    defer terminal.deinit();

    var loop = AsyncLoop.init(allocator, &terminal, .{});
    defer loop.deinit();

    try std.testing.expect(!loop.running);
    try std.testing.expect(loop.frame_count == 0);
}

test "RingBuffer push and latest" {
    var buf = RingBuffer(f32, 4).init();

    buf.push(1.0);
    buf.push(2.0);
    buf.push(3.0);

    try std.testing.expect(buf.len() == 3);
    try std.testing.expect(buf.latest().? == 3.0);

    // Fill and overflow
    buf.push(4.0);
    buf.push(5.0);

    try std.testing.expect(buf.len() == 4);
    try std.testing.expect(buf.latest().? == 5.0);
}

test "MetricsTracker initialization" {
    const tracker = MetricsTracker.init();
    var cpu_buf: [120]f32 = undefined;
    var mem_buf: [120]f32 = undefined;
    _ = tracker.getCpuHistory(&cpu_buf);
    _ = tracker.getMemoryHistory(&mem_buf);
}

test {
    std.testing.refAllDecls(@This());
}
