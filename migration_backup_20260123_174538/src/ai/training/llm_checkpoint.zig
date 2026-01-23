//! LLM trainer checkpoint persistence (weights + optimizer state).
const std = @import("std");
const binary = @import("../../shared/utils.zig").binary;
const time = @import("../../shared/utils.zig");

const checkpoint_magic = "ABLC";
const checkpoint_version: u16 = 1;

pub const CheckpointError = error{
    InvalidFormat,
    UnsupportedVersion,
    PayloadTooLarge,
    StateMismatch,
};

pub const SaveError =
    std.Io.File.OpenError ||
    std.Io.File.Writer.Error ||
    std.mem.Allocator.Error ||
    CheckpointError;
pub const LoadError =
    std.Io.Dir.ReadFileAllocError ||
    std.mem.Allocator.Error ||
    CheckpointError ||
    error{OutOfBounds};

pub const LlmCheckpoint = struct {
    step: u64,
    epoch: u32,
    tokens_processed: u64,
    timestamp: u64,
    weights: []f32,
    m: []f32,
    v: []f32,

    pub fn deinit(self: *LlmCheckpoint, allocator: std.mem.Allocator) void {
        allocator.free(self.v);
        allocator.free(self.m);
        allocator.free(self.weights);
        self.* = undefined;
    }
};

pub const LlmCheckpointView = struct {
    step: u64,
    epoch: u32,
    tokens_processed: u64,
    weights: []const f32,
    m: []const f32,
    v: []const f32,
};

/// Persist a trainer checkpoint to disk.
pub fn saveLlmCheckpoint(
    allocator: std.mem.Allocator,
    path: []const u8,
    checkpoint: LlmCheckpointView,
) SaveError!void {
    var writer = binary.SerializationWriter.init(allocator);
    defer writer.deinit();

    try writer.appendBytes(checkpoint_magic);
    try writer.appendInt(u16, checkpoint_version);
    try writer.appendInt(u64, checkpoint.step);
    try writer.appendInt(u32, checkpoint.epoch);
    try writer.appendInt(u64, checkpoint.tokens_processed);
    try writer.appendInt(u64, unixTimestamp());

    if (checkpoint.weights.len > std.math.maxInt(u32)) return CheckpointError.PayloadTooLarge;
    if (checkpoint.m.len > std.math.maxInt(u32)) return CheckpointError.PayloadTooLarge;
    if (checkpoint.v.len > std.math.maxInt(u32)) return CheckpointError.PayloadTooLarge;

    try writer.appendInt(u32, @intCast(checkpoint.weights.len));
    try writer.appendInt(u32, @intCast(checkpoint.m.len));
    try writer.appendInt(u32, @intCast(checkpoint.v.len));

    for (checkpoint.weights) |value| {
        try writer.appendInt(u32, @bitCast(value));
    }
    for (checkpoint.m) |value| {
        try writer.appendInt(u32, @bitCast(value));
    }
    for (checkpoint.v) |value| {
        try writer.appendInt(u32, @bitCast(value));
    }

    const bytes = try writer.toOwnedSlice();
    defer allocator.free(bytes);

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, bytes);
}

/// Load a trainer checkpoint from disk.
pub fn loadLlmCheckpoint(allocator: std.mem.Allocator, path: []const u8) LoadError!LlmCheckpoint {
    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const data = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(256 * 1024 * 1024),
    );
    defer allocator.free(data);

    var cursor = binary.SerializationCursor.init(data);
    const magic = try cursor.readBytes(checkpoint_magic.len);
    if (!std.mem.eql(u8, magic, checkpoint_magic)) return CheckpointError.InvalidFormat;

    const version = try cursor.readInt(u16);
    if (version != checkpoint_version) return CheckpointError.UnsupportedVersion;

    const step = try cursor.readInt(u64);
    const epoch = try cursor.readInt(u32);
    const tokens_processed = try cursor.readInt(u64);
    const timestamp = try cursor.readInt(u64);
    const weight_count = try cursor.readInt(u32);
    const m_count = try cursor.readInt(u32);
    const v_count = try cursor.readInt(u32);

    const weights = try allocator.alloc(f32, weight_count);
    errdefer allocator.free(weights);
    const m = try allocator.alloc(f32, m_count);
    errdefer allocator.free(m);
    const v = try allocator.alloc(f32, v_count);
    errdefer allocator.free(v);

    var i: usize = 0;
    while (i < weight_count) : (i += 1) {
        const bits = try cursor.readInt(u32);
        weights[i] = @bitCast(bits);
    }
    i = 0;
    while (i < m_count) : (i += 1) {
        const bits = try cursor.readInt(u32);
        m[i] = @bitCast(bits);
    }
    i = 0;
    while (i < v_count) : (i += 1) {
        const bits = try cursor.readInt(u32);
        v[i] = @bitCast(bits);
    }

    return .{
        .step = step,
        .epoch = epoch,
        .tokens_processed = tokens_processed,
        .timestamp = timestamp,
        .weights = weights,
        .m = m,
        .v = v,
    };
}

fn unixTimestamp() u64 {
    const ts = time.unixSeconds();
    if (ts <= 0) return 0;
    return @intCast(ts);
}

test "llm checkpoint roundtrip" {
    const allocator = std.testing.allocator;
    const test_path = "test_llm_checkpoint_roundtrip.bin";

    const weights = [_]f32{ 1.0, 2.0, 3.0 };
    const m = [_]f32{ 0.1, 0.2, 0.3 };
    const v = [_]f32{ 0.01, 0.02, 0.03 };

    const view = LlmCheckpointView{
        .step = 7,
        .epoch = 2,
        .tokens_processed = 42,
        .weights = &weights,
        .m = &m,
        .v = &v,
    };

    saveLlmCheckpoint(allocator, test_path, view) catch |err| {
        std.debug.print("saveLlmCheckpoint failed: {t}\n", .{err});
        return err;
    };

    defer {
        var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();
        std.Io.Dir.cwd().deleteFile(io, test_path) catch {};
    }

    var loaded = loadLlmCheckpoint(allocator, test_path) catch |err| {
        std.debug.print("loadLlmCheckpoint failed: {t}\n", .{err});
        return err;
    };
    defer loaded.deinit(allocator);

    try std.testing.expectEqual(@as(u64, 7), loaded.step);
    try std.testing.expectEqual(@as(u32, 2), loaded.epoch);
    try std.testing.expectEqual(@as(u64, 42), loaded.tokens_processed);
    try std.testing.expectEqualSlices(f32, &weights, loaded.weights);
    try std.testing.expectEqualSlices(f32, &m, loaded.m);
    try std.testing.expectEqualSlices(f32, &v, loaded.v);
}
