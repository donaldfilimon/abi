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

        const data = std.Io.Dir.cwd().readFileAlloc(io, self.events_path, self.allocator, .limited(8 * 1024 * 1024)) catch |err| switch (err) {
            error.FileNotFound => return 0,
            else => return err,
        };
        defer self.allocator.free(data);

        const dir_path = std.fs.path.dirname(out_path) orelse ".";
        try std.Io.Dir.cwd().createDirPath(io, dir_path);

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
    try std.Io.Dir.cwd().createDirPath(io, dir_path);

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

fn readPersistedSummary(allocator: std.mem.Allocator, path: []const u8) PersistedSummary {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = .empty });
    defer io_backend.deinit();
    const io = io_backend.io();
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(4 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => return .{},
        else => {
            std.log.warn("learning telemetry: failed to read {s}: {s}", .{ path, @errorName(err) });
            return .{};
        },
    };
    defer allocator.free(data);

    var summary = PersistedSummary{};
    var it = std.mem.splitScalar(u8, data, '\n');
    var line_no: usize = 0;
    while (it.next()) |line| {
        line_no += 1;
        if (line.len == 0) continue;

        const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch |err| {
            std.log.warn("learning telemetry: ignoring malformed JSONL record in {s}:{d}: {s}", .{ path, line_no, @errorName(err) });
            continue;
        };
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |object| object,
            else => {
                std.log.warn("learning telemetry: ignoring non-object JSONL record in {s}:{d}", .{ path, line_no });
                continue;
            },
        };

        const event_type = jsonString(obj.get("type")) orelse {
            std.log.warn("learning telemetry: ignoring JSONL record without string type in {s}:{d}", .{ path, line_no });
            continue;
        };

        summary.events += 1;

        if (std.mem.eql(u8, event_type, "interaction")) {
            summary.interactions += 1;
            if (obj.get("quality")) |quality_value| {
                if (jsonFloat(quality_value)) |q| {
                    summary.quality_total += q;
                    summary.quality_samples += 1;
                } else {
                    std.log.warn("learning telemetry: ignoring interaction with non-numeric quality in {s}:{d}", .{ path, line_no });
                }
            }
        } else if (std.mem.eql(u8, event_type, "feedback")) {
            const kind = jsonString(obj.get("kind")) orelse {
                std.log.warn("learning telemetry: ignoring feedback without string kind in {s}:{d}", .{ path, line_no });
                continue;
            };
            if (std.mem.eql(u8, kind, "positive")) {
                summary.positive_feedback += 1;
            } else if (std.mem.eql(u8, kind, "negative")) {
                summary.negative_feedback += 1;
            }
        }
    }
    return summary;
}

fn jsonString(value: ?std.json.Value) ?[]const u8 {
    const v = value orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn jsonFloat(value: std.json.Value) ?f32 {
    return switch (value) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f32, s) catch null,
        else => null,
    };
}

test "learning runtime records metrics and feedback" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/events.jsonl", .{tmp.sub_path});
    defer allocator.free(path);

    var runtime = try LearningRuntime.init(allocator);
    defer runtime.deinit();
    runtime.events_path = path;

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
    try std.testing.expectEqual(@as(usize, 2), r.stored_events);
}

test "learning report tolerates missing and malformed persisted events" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const missing_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/missing.jsonl", .{tmp.sub_path});
    defer allocator.free(missing_path);

    var runtime = try LearningRuntime.init(allocator);
    defer runtime.deinit();
    runtime.events_path = missing_path;

    const missing_report = runtime.report();
    try std.testing.expectEqual(@as(usize, 0), missing_report.stored_events);

    const events_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/events.jsonl", .{tmp.sub_path});
    defer allocator.free(events_path);
    try appendLine(allocator, events_path,
        \\{"type":"interaction","quality":0.75}
        \\not json
        \\{"type":"feedback","kind":"negative"}
        \\
    );

    runtime.events_path = events_path;
    const malformed_report = runtime.report();
    try std.testing.expectEqual(@as(usize, 2), malformed_report.stored_events);
    try std.testing.expectEqual(@as(usize, 1), malformed_report.total_interactions);
    try std.testing.expectEqual(@as(usize, 1), malformed_report.negative_feedback_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), malformed_report.avg_quality, 0.001);
}

test "learning export copies persisted events" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const events_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/events.jsonl", .{tmp.sub_path});
    defer allocator.free(events_path);
    const export_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/export.jsonl", .{tmp.sub_path});
    defer allocator.free(export_path);

    var runtime = try LearningRuntime.init(allocator);
    defer runtime.deinit();
    runtime.events_path = events_path;

    try std.testing.expectEqual(@as(usize, 0), try runtime.exportArtifacts(export_path));

    try runtime.recordFeedback(.positive, null);
    try runtime.recordFeedback(.negative, null);
    try std.testing.expectEqual(@as(usize, 2), try runtime.exportArtifacts(export_path));

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = .empty });
    defer io_backend.deinit();
    const data = try std.Io.Dir.cwd().readFileAlloc(io_backend.io(), export_path, allocator, .limited(4096));
    defer allocator.free(data);
    try std.testing.expect(std.mem.indexOf(u8, data, "\"kind\":\"positive\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, data, "\"kind\":\"negative\"") != null);
}
