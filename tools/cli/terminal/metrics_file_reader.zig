//! Metrics File Reader — Polling JSONL file tailer.
//!
//! Reads training metrics from a JSONL file written by TrainingLogger.
//! Tracks file offset to only read new data on each poll() call.
//! Uses C stdio for file access (no Io backend needed in TUI tools).

const std = @import("std");
const training_metrics = @import("training_metrics.zig");

pub const MetricsFileReader = struct {
    /// Path stored as null-terminated fixed buffer.
    path: [512]u8,
    path_len: usize,
    /// File offset to resume reading from.
    last_offset: u64,
    /// Aggregated metrics from parsed events.
    metrics: training_metrics.TrainingMetrics,
    /// Buffer for reading file data.
    read_buf: [8192]u8,

    pub fn init(path: []const u8) MetricsFileReader {
        var self: MetricsFileReader = undefined;
        self.path = [_]u8{0} ** 512;
        self.path_len = @min(path.len, 511);
        @memcpy(self.path[0..self.path_len], path[0..self.path_len]);
        self.last_offset = 0;
        self.metrics = .{};
        self.read_buf = undefined;
        return self;
    }

    /// Poll for new metrics data. Non-blocking.
    /// Returns true if new metrics were found and parsed.
    pub fn poll(self: *MetricsFileReader) bool {
        const data = self.readNewData() orelse return false;
        if (data.len == 0) return false;

        var found_new = false;
        var line_start: usize = 0;

        for (data, 0..) |byte, i| {
            if (byte == '\n') {
                if (i > line_start) {
                    const line = data[line_start..i];
                    const event = training_metrics.MetricsParser.parseLine(line);
                    self.metrics.update(event);
                    found_new = true;
                }
                line_start = i + 1;
            }
        }

        // Advance offset only past complete lines
        self.last_offset += line_start;
        return found_new;
    }

    /// Get current aggregated metrics.
    pub fn getMetrics(self: *const MetricsFileReader) *const training_metrics.TrainingMetrics {
        return &self.metrics;
    }

    /// Reset reader to re-read file from the beginning.
    pub fn reset(self: *MetricsFileReader) void {
        self.last_offset = 0;
        self.metrics = .{};
    }

    fn readNewData(self: *MetricsFileReader) ?[]const u8 {
        const path_z = self.path[0..self.path_len :0];
        const file = std.c.fopen(path_z.ptr, "r") orelse return null;
        defer _ = std.c.fclose(file);

        // Skip past already-read data by discarding bytes
        var remaining = self.last_offset;
        var discard_buf: [4096]u8 = undefined;
        while (remaining > 0) {
            const to_read = @min(remaining, discard_buf.len);
            const n = std.c.fread(&discard_buf, 1, to_read, file);
            if (n == 0) return null;
            remaining -= n;
        }

        const n = std.c.fread(&self.read_buf, 1, self.read_buf.len, file);
        if (n == 0) return null;

        return self.read_buf[0..n];
    }
};

// =============================================================================
// Tests
// =============================================================================

test "MetricsFileReader init" {
    var reader = MetricsFileReader.init("/tmp/test_metrics.jsonl");
    try std.testing.expectEqual(@as(u64, 0), reader.last_offset);
    try std.testing.expectEqual(@as(u64, 0), reader.metrics.current_step);
}

test "MetricsFileReader poll on missing file" {
    var reader = MetricsFileReader.init("/nonexistent/path/metrics.jsonl");
    try std.testing.expect(!reader.poll());
}

test {
    std.testing.refAllDecls(@This());
}
