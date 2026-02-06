//! GPU Stream and Event Synchronization
//!
//! Provides abstractions for GPU command streams and synchronization events.
//! Supports asynchronous kernel launches and memory transfers.

const std = @import("std");
const backend_mod = @import("backend.zig");
const device_mod = @import("device.zig");
// Import utilities from the shared directory (relative to this file)
const time_utils = @import("../../services/shared/utils.zig");

// Zig 0.16 compatibility: Simple spinlock Mutex
const Mutex = struct {
    locked: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    pub fn lock(self: *Mutex) void {
        while (self.locked.swap(true, .acquire)) std.atomic.spinLoopHint();
    }
    pub fn unlock(self: *Mutex) void {
        self.locked.store(false, .release);
    }
};

pub const Backend = backend_mod.Backend;
pub const Device = device_mod.Device;

/// Stream priority levels.
pub const StreamPriority = enum {
    low,
    normal,
    high,

    pub fn value(self: StreamPriority) i32 {
        return switch (self) {
            .low => -1,
            .normal => 0,
            .high => 1,
        };
    }
};

/// Stream flags for creation.
pub const StreamFlags = packed struct {
    /// Non-blocking stream (doesn't synchronize with default stream).
    non_blocking: bool = false,
    /// Enable timing for events on this stream.
    enable_timing: bool = true,

    pub fn default() StreamFlags {
        return .{};
    }
};

/// Stream state.
pub const StreamState = enum(u8) {
    /// Stream is idle.
    idle = 0,
    /// Stream has pending work.
    busy = 1,
    /// Stream encountered an error.
    error_state = 2,
    /// Stream is destroyed.
    destroyed = 3,
};

/// GPU command stream for asynchronous operations.
pub const Stream = struct {
    allocator: std.mem.Allocator,
    id: u64,
    device_id: u32,
    backend: Backend,
    priority: StreamPriority,
    flags: StreamFlags,
    state: std.atomic.Value(StreamState),
    /// Backend-specific handle.
    handle: ?*anyopaque,
    /// Pending event count.
    pending_count: std.atomic.Value(u32),
    /// Creation timestamp.
    created_at: i64,

    /// Create a new stream.
    pub fn init(allocator: std.mem.Allocator, device: *const Device, options: StreamOptions) !Stream {
        const id = generateStreamId();
        const now = time_utils.nowSeconds();

        return .{
            .allocator = allocator,
            .id = id,
            .device_id = device.id,
            .backend = device.backend,
            .priority = options.priority,
            .flags = options.flags,
            .state = std.atomic.Value(StreamState).init(.idle),
            .handle = null, // Would be initialized by backend
            .pending_count = std.atomic.Value(u32).init(0),
            .created_at = now,
        };
    }

    /// Destroy the stream.
    pub fn deinit(self: *Stream) void {
        self.state.store(.destroyed, .release);
        // Backend would clean up handle here
        self.handle = null;
        self.* = undefined;
    }

    /// Synchronize this stream (wait for all pending operations).
    pub fn synchronize(self: *Stream) !void {
        if (self.state.load(.acquire) == .destroyed) {
            return error.StreamDestroyed;
        }

        // In a real implementation, this would call the backend's stream sync
        // For now, just mark as idle
        self.state.store(.idle, .release);
        self.pending_count.store(0, .release);
    }

    /// Query if the stream has completed all work.
    pub fn isComplete(self: *const Stream) bool {
        return self.pending_count.load(.acquire) == 0;
    }

    /// Record an event on this stream.
    pub fn recordEvent(self: *Stream, event: *Event) !void {
        if (self.state.load(.acquire) == .destroyed) {
            return error.StreamDestroyed;
        }

        event.stream_id = self.id;
        event.state.store(.recorded, .release);
        event.recorded_at = time_utils.nowSeconds();
    }

    /// Wait for an event from another stream.
    pub fn waitEvent(self: *Stream, event: *const Event) !void {
        if (self.state.load(.acquire) == .destroyed) {
            return error.StreamDestroyed;
        }

        // Check event is valid
        if (event.state.load(.acquire) == .destroyed) {
            return error.EventDestroyed;
        }

        // In a real implementation, this would insert a wait operation
        _ = self.pending_count.fetchAdd(1, .acq_rel);
        self.state.store(.busy, .release);
    }

    /// Get the current state.
    pub fn getState(self: *const Stream) StreamState {
        return self.state.load(.acquire);
    }

    /// Get pending operation count.
    pub fn getPendingCount(self: *const Stream) u32 {
        return self.pending_count.load(.acquire);
    }

    /// Mark that an operation was submitted.
    pub fn notifySubmit(self: *Stream) void {
        _ = self.pending_count.fetchAdd(1, .acq_rel);
        self.state.store(.busy, .release);
    }

    /// Mark that an operation completed.
    pub fn notifyComplete(self: *Stream) void {
        const prev = self.pending_count.fetchSub(1, .acq_rel);
        if (prev == 1) {
            self.state.store(.idle, .release);
        }
    }
};

/// Stream creation options.
pub const StreamOptions = struct {
    priority: StreamPriority = .normal,
    flags: StreamFlags = StreamFlags.default(),
};

/// Event state.
pub const EventState = enum(u8) {
    /// Event is created but not recorded.
    initial = 0,
    /// Event is recorded on a stream.
    recorded = 1,
    /// Event has completed.
    completed = 2,
    /// Event is destroyed.
    destroyed = 3,
};

/// GPU synchronization event.
pub const Event = struct {
    allocator: std.mem.Allocator,
    id: u64,
    device_id: u32,
    backend: Backend,
    state: std.atomic.Value(EventState),
    /// Stream this event was recorded on (if any).
    stream_id: ?u64,
    /// Backend-specific handle.
    handle: ?*anyopaque,
    /// Flags.
    flags: EventFlags,
    /// Recording timestamp.
    recorded_at: ?i64,
    /// Completion timestamp.
    completed_at: ?i64,

    /// Create a new event.
    pub fn init(allocator: std.mem.Allocator, device: *const Device, options: EventOptions) !Event {
        const id = generateEventId();

        return .{
            .allocator = allocator,
            .id = id,
            .device_id = device.id,
            .backend = device.backend,
            .state = std.atomic.Value(EventState).init(.initial),
            .stream_id = null,
            .handle = null, // Would be initialized by backend
            .flags = options.flags,
            .recorded_at = null,
            .completed_at = null,
        };
    }

    /// Destroy the event.
    pub fn deinit(self: *Event) void {
        self.state.store(.destroyed, .release);
        self.handle = null;
        self.* = undefined;
    }

    /// Query if the event has completed.
    pub fn isComplete(self: *const Event) bool {
        return self.state.load(.acquire) == .completed;
    }

    /// Synchronize (wait) for this event.
    pub fn synchronize(self: *Event) !void {
        const current_state = self.state.load(.acquire);

        if (current_state == .destroyed) {
            return error.EventDestroyed;
        }

        if (current_state == .initial) {
            return error.EventNotRecorded;
        }

        // In a real implementation, this would wait on the backend event
        self.state.store(.completed, .release);
        self.completed_at = time_utils.nowSeconds();
    }

    /// Get elapsed time between this event and another (in nanoseconds).
    pub fn elapsedNs(self: *const Event, end_event: *const Event) !i64 {
        if (!self.flags.enable_timing or !end_event.flags.enable_timing) {
            return error.TimingDisabled;
        }

        const start = self.recorded_at orelse return error.EventNotRecorded;
        const end = end_event.recorded_at orelse return error.EventNotRecorded;

        return end - start;
    }

    /// Get the current state.
    pub fn getState(self: *const Event) EventState {
        return self.state.load(.acquire);
    }

    /// Mark event as completed (for testing/simulation).
    pub fn markComplete(self: *Event) void {
        self.state.store(.completed, .release);
        self.completed_at = time_utils.nowSeconds();
    }
};

/// Event flags.
pub const EventFlags = packed struct {
    /// Enable timing for this event.
    enable_timing: bool = true,
    /// Interprocess event (can be shared across processes).
    interprocess: bool = false,

    pub fn default() EventFlags {
        return .{};
    }
};

/// Event creation options.
pub const EventOptions = struct {
    flags: EventFlags = EventFlags.default(),
};

/// Stream manager for managing multiple streams.
pub const StreamManager = struct {
    allocator: std.mem.Allocator,
    streams: std.ArrayListUnmanaged(*Stream),
    events: std.ArrayListUnmanaged(*Event),
    default_stream: ?*Stream,
    mutex: Mutex,

    pub fn init(allocator: std.mem.Allocator) StreamManager {
        return .{
            .allocator = allocator,
            .streams = .empty,
            .events = .empty,
            .default_stream = null,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *StreamManager) void {
        // Destroy all events
        for (self.events.items) |event| {
            event.deinit();
            self.allocator.destroy(event);
        }
        self.events.deinit(self.allocator);

        // Destroy all streams
        for (self.streams.items) |stream| {
            stream.deinit();
            self.allocator.destroy(stream);
        }
        self.streams.deinit(self.allocator);

        self.* = undefined;
    }

    /// Create a new stream.
    pub fn createStream(self: *StreamManager, device: *const Device, options: StreamOptions) !*Stream {
        self.mutex.lock();
        defer self.mutex.unlock();

        const stream = try self.allocator.create(Stream);
        errdefer self.allocator.destroy(stream);

        stream.* = try Stream.init(self.allocator, device, options);

        try self.streams.append(self.allocator, stream);

        // Set as default if first stream
        if (self.default_stream == null) {
            self.default_stream = stream;
        }

        return stream;
    }

    /// Destroy a stream.
    pub fn destroyStream(self: *StreamManager, stream: *Stream) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Remove from list
        for (self.streams.items, 0..) |s, i| {
            if (s == stream) {
                _ = self.streams.swapRemove(i);
                break;
            }
        }

        // Clear default if this was it
        if (self.default_stream == stream) {
            self.default_stream = if (self.streams.items.len > 0) self.streams.items[0] else null;
        }

        stream.deinit();
        self.allocator.destroy(stream);
    }

    /// Create a new event.
    pub fn createEvent(self: *StreamManager, device: *const Device, options: EventOptions) !*Event {
        self.mutex.lock();
        defer self.mutex.unlock();

        const event = try self.allocator.create(Event);
        errdefer self.allocator.destroy(event);

        event.* = try Event.init(self.allocator, device, options);

        try self.events.append(self.allocator, event);

        return event;
    }

    /// Destroy an event.
    pub fn destroyEvent(self: *StreamManager, event: *Event) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Remove from list
        for (self.events.items, 0..) |e, i| {
            if (e == event) {
                _ = self.events.swapRemove(i);
                break;
            }
        }

        event.deinit();
        self.allocator.destroy(event);
    }

    /// Get the default stream.
    pub fn getDefaultStream(self: *StreamManager) ?*Stream {
        return self.default_stream;
    }

    /// Synchronize all streams.
    pub fn synchronizeAll(self: *StreamManager) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.streams.items) |stream| {
            try stream.synchronize();
        }
    }

    /// Get stream count.
    pub fn streamCount(self: *const StreamManager) usize {
        return self.streams.items.len;
    }

    /// Get event count.
    pub fn eventCount(self: *const StreamManager) usize {
        return self.events.items.len;
    }
};

// ID generation
var stream_id_counter = std.atomic.Value(u64).init(0);
var event_id_counter = std.atomic.Value(u64).init(0);

fn generateStreamId() u64 {
    return stream_id_counter.fetchAdd(1, .monotonic);
}

fn generateEventId() u64 {
    return event_id_counter.fetchAdd(1, .monotonic);
}

// ============================================================================
// Tests
// ============================================================================

test "Stream basic operations" {
    const device = Device{
        .id = 0,
        .backend = .vulkan,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var stream = try Stream.init(std.testing.allocator, &device, .{});
    defer stream.deinit();

    try std.testing.expect(stream.getState() == .idle);
    try std.testing.expect(stream.isComplete());

    stream.notifySubmit();
    try std.testing.expect(stream.getState() == .busy);
    try std.testing.expect(!stream.isComplete());

    stream.notifyComplete();
    try std.testing.expect(stream.getState() == .idle);
    try std.testing.expect(stream.isComplete());
}

test "Event basic operations" {
    const device = Device{
        .id = 0,
        .backend = .cuda,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var event = try Event.init(std.testing.allocator, &device, .{});
    defer event.deinit();

    try std.testing.expect(event.getState() == .initial);
    try std.testing.expect(!event.isComplete());
}

test "StreamManager create and destroy" {
    const device = Device{
        .id = 0,
        .backend = .vulkan,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var manager = StreamManager.init(std.testing.allocator);
    defer manager.deinit();

    const stream1 = try manager.createStream(&device, .{});
    const stream2 = try manager.createStream(&device, .{ .priority = .high });

    try std.testing.expect(manager.streamCount() == 2);
    try std.testing.expect(manager.getDefaultStream() == stream1);

    manager.destroyStream(stream1);
    try std.testing.expect(manager.streamCount() == 1);
    try std.testing.expect(manager.getDefaultStream() == stream2);
}

test "Stream record and wait event" {
    const device = Device{
        .id = 0,
        .backend = .cuda,
        .name = "Test Device",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var stream = try Stream.init(std.testing.allocator, &device, .{});
    defer stream.deinit();

    var event = try Event.init(std.testing.allocator, &device, .{});
    defer event.deinit();

    try stream.recordEvent(&event);
    try std.testing.expect(event.getState() == .recorded);
    try std.testing.expect(event.stream_id == stream.id);
}
