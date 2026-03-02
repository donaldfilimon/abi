//! Training Metrics Parser
//!
//! Parses JSONL metrics files written by TrainingLogger.MetricsStream
//! for the training dashboard TUI.

const std = @import("std");

// ===============================================================================
// Types
// ===============================================================================

/// Type of metric event
pub const EventType = enum {
    scalar,
    checkpoint,
    progress,
    unknown,

    pub fn fromString(s: []const u8) EventType {
        if (std.mem.eql(u8, s, "scalar")) return .scalar;
        if (std.mem.eql(u8, s, "checkpoint")) return .checkpoint;
        if (std.mem.eql(u8, s, "progress")) return .progress;
        return .unknown;
    }
};

/// A parsed metric event from JSONL
pub const MetricEvent = struct {
    event_type: EventType = .unknown,
    tag: []const u8 = "",
    value: f32 = 0,
    step: u64 = 0,
    timestamp: u64 = 0,
    // Checkpoint-specific
    path: []const u8 = "",
    size: u64 = 0,
    // Progress-specific
    epoch: u32 = 0,
    total_epochs: u32 = 0,
    total_steps: u64 = 0,
};

/// Aggregated training metrics for display
pub const TrainingMetrics = struct {
    // Loss values (ring buffer for sparkline)
    train_loss: MetricHistory = MetricHistory.init(),
    val_loss: MetricHistory = MetricHistory.init(),

    // Learning rate
    learning_rate: MetricHistory = MetricHistory.init(),

    // Progress
    current_epoch: u32 = 0,
    total_epochs: u32 = 0,
    current_step: u64 = 0,
    total_steps: u64 = 0,

    // Checkpoints (most recent)
    last_checkpoint_path: [256]u8 = [_]u8{0} ** 256,
    last_checkpoint_size: u64 = 0,
    checkpoint_count: u32 = 0,

    // Timestamps
    start_time: u64 = 0,
    last_update: u64 = 0,

    pub fn update(self: *TrainingMetrics, event: MetricEvent) void {
        self.last_update = event.timestamp;
        if (self.start_time == 0) self.start_time = event.timestamp;

        switch (event.event_type) {
            .scalar => {
                if (std.mem.eql(u8, event.tag, "loss/train")) {
                    self.train_loss.push(event.value);
                } else if (std.mem.eql(u8, event.tag, "loss/val")) {
                    self.val_loss.push(event.value);
                } else if (std.mem.eql(u8, event.tag, "lr") or
                    std.mem.eql(u8, event.tag, "learning_rate"))
                {
                    self.learning_rate.push(event.value);
                }
            },
            .checkpoint => {
                const len = @min(event.path.len, self.last_checkpoint_path.len);
                @memcpy(self.last_checkpoint_path[0..len], event.path[0..len]);
                self.last_checkpoint_size = event.size;
                self.checkpoint_count += 1;
            },
            .progress => {
                self.current_epoch = event.epoch;
                self.total_epochs = event.total_epochs;
                self.current_step = event.step;
                self.total_steps = event.total_steps;
            },
            .unknown => {},
        }
    }

    /// Get elapsed time in seconds
    pub fn elapsedSeconds(self: TrainingMetrics) u64 {
        if (self.start_time == 0 or self.last_update == 0) return 0;
        return self.last_update - self.start_time;
    }
};

/// Ring buffer for metric history (for sparklines)
pub const MetricHistory = struct {
    pub const SIZE = 30;

    values: [SIZE]f32 = [_]f32{0} ** SIZE,
    normalized: [SIZE]u8 = [_]u8{50} ** SIZE,
    pos: usize = 0,
    count: usize = 0,
    min_val: f32 = std.math.floatMax(f32),
    max_val: f32 = std.math.floatMin(f32),

    pub fn init() MetricHistory {
        return .{};
    }

    pub fn push(self: *MetricHistory, value: f32) void {
        self.values[self.pos] = value;

        // Update min/max
        if (value < self.min_val) self.min_val = value;
        if (value > self.max_val) self.max_val = value;

        // Normalize to 0-100 for sparkline
        const range = self.max_val - self.min_val;
        if (range > 0.0001) {
            self.normalized[self.pos] = @intFromFloat(@min(100, @max(0, ((value - self.min_val) / range) * 100)));
        } else {
            self.normalized[self.pos] = 50;
        }

        self.pos = (self.pos + 1) % SIZE;
        if (self.count < SIZE) self.count += 1;
    }

    /// Get normalized values in chronological order
    pub fn getNormalized(self: *const MetricHistory) [SIZE]u8 {
        var result: [SIZE]u8 = [_]u8{50} ** SIZE;
        if (self.count == 0) return result;

        const start = if (self.count < SIZE) 0 else self.pos;
        for (0..self.count) |i| {
            const src_idx = (start + i) % SIZE;
            result[i] = self.normalized[src_idx];
        }
        return result;
    }

    /// Get latest value
    pub fn latest(self: *const MetricHistory) f32 {
        if (self.count == 0) return 0;
        const idx = if (self.pos == 0) SIZE - 1 else self.pos - 1;
        return self.values[idx];
    }
};

// ===============================================================================
// Parser
// ===============================================================================

pub const MetricsParser = struct {
    /// Parse a single JSONL line into a MetricEvent.
    /// Uses simple string searching for known JSON format.
    pub fn parseLine(line: []const u8) MetricEvent {
        var event = MetricEvent{};

        // Extract type
        if (extractString(line, "\"type\":\"")) |type_str| {
            event.event_type = EventType.fromString(type_str);
        }

        // Extract tag
        if (extractString(line, "\"tag\":\"")) |tag| {
            event.tag = tag;
        }

        // Extract path (for checkpoints)
        if (extractString(line, "\"path\":\"")) |path| {
            event.path = path;
        }

        // Extract numeric values
        if (extractNumber(line, "\"value\":")) |val| {
            event.value = @floatCast(val);
        }
        if (extractInt(line, "\"step\":")) |step| {
            event.step = step;
        }
        if (extractInt(line, "\"ts\":")) |ts| {
            event.timestamp = ts;
        }
        if (extractInt(line, "\"size\":")) |size| {
            event.size = size;
        }
        if (extractInt(line, "\"epoch\":")) |epoch| {
            event.epoch = @intCast(epoch);
        }
        if (extractInt(line, "\"total_epochs\":")) |te| {
            event.total_epochs = @intCast(te);
        }
        if (extractInt(line, "\"total_steps\":")) |ts| {
            event.total_steps = ts;
        }

        return event;
    }

    fn extractString(json: []const u8, key: []const u8) ?[]const u8 {
        const start = std.mem.indexOf(u8, json, key) orelse return null;
        const val_start = start + key.len;
        const end = std.mem.indexOfPos(u8, json, val_start, "\"") orelse return null;
        return json[val_start..end];
    }

    fn extractNumber(json: []const u8, key: []const u8) ?f64 {
        const start = std.mem.indexOf(u8, json, key) orelse return null;
        const val_start = start + key.len;
        var end = val_start;
        while (end < json.len and (json[end] == '.' or json[end] == '-' or
            json[end] == 'e' or json[end] == 'E' or json[end] == '+' or
            (json[end] >= '0' and json[end] <= '9'))) : (end += 1)
        {}
        return std.fmt.parseFloat(f64, json[val_start..end]) catch null;
    }

    fn extractInt(json: []const u8, key: []const u8) ?u64 {
        const start = std.mem.indexOf(u8, json, key) orelse return null;
        const val_start = start + key.len;
        var end = val_start;
        while (end < json.len and json[end] >= '0' and json[end] <= '9') : (end += 1) {}
        return std.fmt.parseInt(u64, json[val_start..end], 10) catch null;
    }
};

// ===============================================================================
// Tests
// ===============================================================================

test "MetricsParser parses scalar event" {
    const line = "{\"type\":\"scalar\",\"tag\":\"loss/train\",\"value\":0.5,\"step\":100,\"ts\":1706123456}";
    const event = MetricsParser.parseLine(line);

    try std.testing.expectEqual(EventType.scalar, event.event_type);
    try std.testing.expectEqualStrings("loss/train", event.tag);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), event.value, 0.001);
    try std.testing.expectEqual(@as(u64, 100), event.step);
    try std.testing.expectEqual(@as(u64, 1706123456), event.timestamp);
}

test "MetricsParser parses checkpoint event" {
    const line = "{\"type\":\"checkpoint\",\"path\":\"epoch-5.ckpt\",\"size\":13002400,\"step\":500,\"ts\":1706123500}";
    const event = MetricsParser.parseLine(line);

    try std.testing.expectEqual(EventType.checkpoint, event.event_type);
    try std.testing.expectEqualStrings("epoch-5.ckpt", event.path);
    try std.testing.expectEqual(@as(u64, 13002400), event.size);
}

test "MetricsParser parses progress event" {
    const line = "{\"type\":\"progress\",\"epoch\":3,\"total_epochs\":10,\"step\":150,\"total_steps\":500,\"ts\":1706123600}";
    const event = MetricsParser.parseLine(line);

    try std.testing.expectEqual(EventType.progress, event.event_type);
    try std.testing.expectEqual(@as(u32, 3), event.epoch);
    try std.testing.expectEqual(@as(u32, 10), event.total_epochs);
}

test "MetricHistory tracks values" {
    var history = MetricHistory.init();

    history.push(0.5);
    history.push(0.4);
    history.push(0.3);

    try std.testing.expectEqual(@as(usize, 3), history.count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), history.latest(), 0.001);
}

test "TrainingMetrics updates from events" {
    var metrics = TrainingMetrics{};

    const scalar_event = MetricEvent{
        .event_type = .scalar,
        .tag = "loss/train",
        .value = 0.5,
        .step = 100,
        .timestamp = 1000,
    };
    metrics.update(scalar_event);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), metrics.train_loss.latest(), 0.001);
    try std.testing.expectEqual(@as(u64, 1000), metrics.start_time);
}

test {
    std.testing.refAllDecls(@This());
}
