//! Safe learning telemetry runtime for chat and agents.
//!
//! This facade connects self-improvement metrics, feedback capture, and optional
//! manual training hooks without mutating models or source code automatically.

const std = @import("std");
const build_options = @import("build_options");
const time = @import("../../foundation/mod.zig").time;
const json_utils = @import("../../foundation/mod.zig").utils.json;
const feedback_mod = @import("feedback/mod.zig");
const self_improve_mod = @import("self_improve.zig");
const training_mod = if (build_options.feat_training) @import("training/mod.zig") else @import("training/stub.zig");

const default_events_path = ".abi/learning/events.jsonl";

pub const FeedbackKind = enum {
    positive,
    negative,
    neutral,

    pub fn fromString(value: []const u8) ?FeedbackKind {
        if (std.ascii.eqlIgnoreCase(value, "positive") or
            std.ascii.eqlIgnoreCase(value, "up") or
            std.ascii.eqlIgnoreCase(value, "good"))
        {
            return .positive;
        }
        if (std.ascii.eqlIgnoreCase(value, "negative") or
            std.ascii.eqlIgnoreCase(value, "down") or
            std.ascii.eqlIgnoreCase(value, "bad"))
        {
            return .negative;
        }
        if (std.ascii.eqlIgnoreCase(value, "neutral")) return .neutral;
        return null;
    }

    pub fn label(self: FeedbackKind) []const u8 {
        return switch (self) {
            .positive => "positive",
            .negative => "negative",
            .neutral => "neutral",
        };
    }
};

pub const Interaction = struct {
    prompt: []const u8,
    response: []const u8,
    profile: []const u8 = "unknown",
    backend: []const u8 = "unknown",
    latency_ms: f32 = 0,
    selected_model: []const u8 = "",
    quality_score: ?f32 = null,
    wdbx_block_id: ?u64 = null,
    route_reason: []const u8 = "",
    retrieval_hits: usize = 0,
    constitution_passed: bool = true,
    used_fallback_provider: bool = false,
};

pub const LearningReport = struct {
    total_interactions: usize = 0,
    avg_quality: f32 = 0,
    positive_feedback_count: usize = 0,
    negative_feedback_count: usize = 0,
    stored_events: usize = 0,
    auto_retrain_enabled: bool = false,
};

pub const LearningRuntime = struct {
    allocator: std.mem.Allocator,
    improver: self_improve_mod.SelfImprover,
    feedback: *feedback_mod.FeedbackSystem,
    learning_system: ?training_mod.SelfLearningSystem = null,
    events_path: []const u8,
    auto_retrain: bool = false,

    pub fn init(allocator: std.mem.Allocator) !LearningRuntime {
        const feedback = try feedback_mod.FeedbackSystem.init(allocator, .{ .min_analysis_threshold = 1 });
        errdefer feedback.deinit();

        return .{
            .allocator = allocator,
            .improver = self_improve_mod.SelfImprover.init(allocator),
            .feedback = feedback,
            .events_path = default_events_path,
        };
    }

    pub fn deinit(self: *LearningRuntime) void {
        if (self.learning_system) |*learning| learning.deinit();
        self.feedback.deinit();
        self.improver.deinit();
    }

    pub fn recordInteraction(self: *LearningRuntime, interaction: Interaction) !void {
        const metrics = self.improver.evaluateResponse(interaction.response, interaction.prompt);
        try self.improver.recordMetrics(metrics);
        try self.appendInteraction(interaction, interaction.quality_score orelse metrics.overall);
    }

    pub fn recordFeedback(self: *LearningRuntime, kind: FeedbackKind, note: ?[]const u8) !void {
        switch (kind) {
            .positive => self.improver.recordFeedback(true),
            .negative => self.improver.recordFeedback(false),
            .neutral => {},
        }

        const thumbs_up = kind != .negative;
        _ = self.feedback.submitThumbs(thumbs_up, .abi, .quality, "cli", note);
        try self.appendFeedback(kind, note);
    }

    pub fn report(self: *LearningRuntime) LearningReport {
        const perf = self.improver.getReport();
        var result = LearningReport{
            .total_interactions = perf.total_interactions,
            .avg_quality = perf.avg_quality,
            .positive_feedback_count = perf.positive_feedback_count,
            .negative_feedback_count = perf.negative_feedback_count,
            .stored_events = countStoredEvents(self.allocator, self.events_path),
            .auto_retrain_enabled = self.auto_retrain,
        };

        const persisted = readPersistedSummary(self.allocator, self.events_path);
        result.total_interactions += persisted.interactions;
        result.positive_feedback_count += persisted.positive_feedback;
        result.negative_feedback_count += persisted.negative_feedback;
        result.stored_events = persisted.events;
        if (result.avg_quality == 0 and persisted.quality_samples > 0) {
            result.avg_quality = persisted.quality_total / @as(f32, @floatFromInt(persisted.quality_samples));
        }
        return result;
    }

    pub fn maybeTriggerRetrain(self: *LearningRuntime) !bool {
        if (!self.auto_retrain) return false;
        if (self.learning_system) |*learning| {
            try learning.update();
            return true;
        }
        return false;
    }

    pub fn forceRetrain(self: *LearningRuntime) !bool {
        if (self.learning_system == null) {
            self.learning_system = training_mod.SelfLearningSystem.init(self.allocator, .{
                .enable_vision = false,
                .enable_documents = false,
                .enable_video = false,
                .enable_audio = false,
                .enable_all_modalities = false,
                .min_buffer_size = 1,
                .batch_size = 1,
            }) catch return false;
        }
        if (self.learning_system) |*learning| {
            try learning.update();
            return true;
        }
        return false;
    }

    pub fn exportArtifacts(self: *LearningRuntime, out_path: []const u8) !usize {
        var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = .empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        const data = std.Io.Dir.cwd().readFileAlloc(io, self.events_path, self.allocator, .limited(8 * 1024 * 1024)) catch return 0;
        defer self.allocator.free(data);

        const dir_path = std.fs.path.dirname(out_path) orelse ".";
        std.Io.Dir.cwd().createDirPath(io, dir_path) catch {};

        var file = try std.Io.Dir.cwd().createFile(io, out_path, .{ .truncate = true });
        defer file.close(io);
        var buf: [1024]u8 = undefined;
        var writer = file.writer(io, &buf);
        try writer.interface.writeAll(data);
        try writer.flush();
        return std.mem.count(u8, data, "\n");
    }

    fn appendInteraction(self: *LearningRuntime, interaction: Interaction, quality: f32) !void {
        var line = std.ArrayListUnmanaged(u8).empty;
        defer line.deinit(self.allocator);
        try line.appendSlice(self.allocator, "{\"type\":\"interaction\",\"ts\":");
        try line.print(self.allocator, "{d}", .{time.unixSeconds()});
        try line.appendSlice(self.allocator, ",\"profile\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &line, interaction.profile);
        try line.appendSlice(self.allocator, "\",\"backend\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &line, interaction.backend);
        try line.appendSlice(self.allocator, "\",\"model\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &line, interaction.selected_model);
        try line.print(self.allocator, "\",\"prompt_len\":{d},\"response_len\":{d},\"latency_ms\":{d:.2},\"quality\":{d:.4},\"retrieval_hits\":{d},\"constitution_passed\":{s},\"used_fallback_provider\":{s}", .{
            interaction.prompt.len,
            interaction.response.len,
            interaction.latency_ms,
            quality,
            interaction.retrieval_hits,
            if (interaction.constitution_passed) "true" else "false",
            if (interaction.used_fallback_provider) "true" else "false",
        });
        if (interaction.route_reason.len > 0) {
            try line.appendSlice(self.allocator, ",\"route_reason\":\"");
            try json_utils.appendJsonEscaped(self.allocator, &line, interaction.route_reason);
            try line.appendSlice(self.allocator, "\"");
        }
        if (interaction.wdbx_block_id) |block_id| {
            try line.print(self.allocator, ",\"wdbx_block_id\":{d}", .{block_id});
        }
        try line.appendSlice(self.allocator, "}\n");
        try appendLine(self.allocator, self.events_path, line.items);
    }

    fn appendFeedback(self: *LearningRuntime, kind: FeedbackKind, note: ?[]const u8) !void {
        var line = std.ArrayListUnmanaged(u8).empty;
        defer line.deinit(self.allocator);
        try line.appendSlice(self.allocator, "{\"type\":\"feedback\",\"ts\":");
        try line.print(self.allocator, "{d}", .{time.unixSeconds()});
        try line.appendSlice(self.allocator, ",\"kind\":\"");
        try json_utils.appendJsonEscaped(self.allocator, &line, kind.label());
        try line.appendSlice(self.allocator, "\"");
        if (note) |n| {
            try line.appendSlice(self.allocator, ",\"note\":\"");
            try json_utils.appendJsonEscaped(self.allocator, &line, n);
            try line.appendSlice(self.allocator, "\"");
        }
        try line.appendSlice(self.allocator, "}\n");
        try appendLine(self.allocator, self.events_path, line.items);
    }
};

fn appendLine(allocator: std.mem.Allocator, path: []const u8, line: []const u8) !void {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = .empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const dir_path = std.fs.path.dirname(path) orelse ".";
    std.Io.Dir.cwd().createDirPath(io, dir_path) catch {};

    var file = std.Io.Dir.cwd().openFile(io, path, .{ .mode = .write_only }) catch |err| switch (err) {
        error.FileNotFound => try std.Io.Dir.cwd().createFile(io, path, .{}),
        else => return err,
    };
    defer file.close(io);
    const end_offset = try file.length(io);
    try file.writePositionalAll(io, line, end_offset);
}

const PersistedSummary = struct {
    events: usize = 0,
    interactions: usize = 0,
    positive_feedback: usize = 0,
    negative_feedback: usize = 0,
    quality_total: f32 = 0,
    quality_samples: usize = 0,
};

fn countStoredEvents(allocator: std.mem.Allocator, path: []const u8) usize {
    return readPersistedSummary(allocator, path).events;
}

fn readPersistedSummary(allocator: std.mem.Allocator, path: []const u8) PersistedSummary {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = .empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(4 * 1024 * 1024)) catch return .{};
    defer allocator.free(data);

    var summary = PersistedSummary{};
    var it = std.mem.splitScalar(u8, data, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        summary.events += 1;
        if (std.mem.indexOf(u8, line, "\"type\":\"interaction\"") != null) summary.interactions += 1;
        if (std.mem.indexOf(u8, line, "\"kind\":\"positive\"") != null) summary.positive_feedback += 1;
        if (std.mem.indexOf(u8, line, "\"kind\":\"negative\"") != null) summary.negative_feedback += 1;
        if (std.mem.indexOf(u8, line, "\"quality\":")) |idx| {
            const start = idx + "\"quality\":".len;
            var end = start;
            while (end < line.len and (std.ascii.isDigit(line[end]) or line[end] == '.')) : (end += 1) {}
            if (std.fmt.parseFloat(f32, line[start..end])) |q| {
                summary.quality_total += q;
                summary.quality_samples += 1;
            } else |_| {}
        }
    }
    return summary;
}

test "learning runtime records metrics and feedback" {
    var runtime = try LearningRuntime.init(std.testing.allocator);
    defer runtime.deinit();
    try runtime.recordInteraction(.{
        .prompt = "hello",
        .response = "hello back with enough detail",
        .profile = "abi",
        .backend = "test",
    });
    try runtime.recordFeedback(.positive, "useful");
    const r = runtime.report();
    try std.testing.expect(r.total_interactions >= 1);
    try std.testing.expect(r.positive_feedback_count >= 1);
}
