//! Training checkpoint storage and persistence helpers.
const std = @import("std");
const binary = @import("../../../shared/utils/binary.zig");
const time = @import("../../../shared/utils/time.zig");

const checkpoint_magic = "ABIC";
const checkpoint_version: u16 = 1;

pub const CheckpointError = error{
    InvalidFormat,
    UnsupportedVersion,
    PayloadTooLarge,
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

pub const StoreError = error{
    NoCheckpoint,
};

pub const Checkpoint = struct {
    step: u64,
    timestamp: u64,
    weights: []f32,

    /// Release resources owned by the checkpoint.
    pub fn deinit(self: *Checkpoint, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        self.* = undefined;
    }
};

pub const CheckpointView = struct {
    step: u64,
    timestamp: u64,
    weights: []const f32,
};

pub const CheckpointStore = struct {
    allocator: std.mem.Allocator,
    checkpoints: std.ArrayListUnmanaged(Checkpoint),
    max_checkpoints: usize,

    /// Initialize an in-memory checkpoint store.
    /// @param allocator Memory allocator for storage
    /// @param max_checkpoints Max checkpoints retained (0 = unlimited)
    /// @return Initialized CheckpointStore
    pub fn init(allocator: std.mem.Allocator, max_checkpoints: usize) CheckpointStore {
        return .{
            .allocator = allocator,
            .checkpoints = std.ArrayListUnmanaged(Checkpoint).empty,
            .max_checkpoints = max_checkpoints,
        };
    }

    /// Release checkpoints held in memory.
    pub fn deinit(self: *CheckpointStore) void {
        for (self.checkpoints.items) |*checkpoint| {
            checkpoint.deinit(self.allocator);
        }
        self.checkpoints.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a new checkpoint by copying weights.
    /// @param step Training step identifier
    /// @param weights Model weights snapshot
    pub fn add(self: *CheckpointStore, step: u64, weights: []const f32) !void {
        const copy = try self.allocator.alloc(f32, weights.len);
        std.mem.copyForwards(f32, copy, weights);

        const timestamp = unixTimestamp();
        try self.checkpoints.append(self.allocator, .{
            .step = step,
            .timestamp = timestamp,
            .weights = copy,
        });
        self.prune();
    }

    /// Return the most recent checkpoint.
    pub fn latest(self: *const CheckpointStore) ?CheckpointView {
        if (self.checkpoints.items.len == 0) return null;
        const last = self.checkpoints.items[self.checkpoints.items.len - 1];
        return .{
            .step = last.step,
            .timestamp = last.timestamp,
            .weights = last.weights,
        };
    }

    /// Count the checkpoints stored.
    pub fn count(self: *const CheckpointStore) usize {
        return self.checkpoints.items.len;
    }

    /// Persist the latest checkpoint to disk.
    pub fn saveLatestToFile(self: *const CheckpointStore, path: []const u8) SaveLatestError!void {
        const latest_view = self.latest() orelse return StoreError.NoCheckpoint;
        return saveCheckpoint(self.allocator, path, latest_view);
    }

    fn prune(self: *CheckpointStore) void {
        if (self.max_checkpoints == 0) return;
        while (self.checkpoints.items.len > self.max_checkpoints) {
            var removed = self.checkpoints.orderedRemove(0);
            removed.deinit(self.allocator);
        }
    }
};

pub const SaveLatestError = SaveError || StoreError;

/// Persist a checkpoint view to disk using a binary format.
/// @param allocator Memory allocator for the serialization buffer
/// @param path Destination file path
/// @param checkpoint Checkpoint view to serialize
/// @return Error on failure
pub fn saveCheckpoint(
    allocator: std.mem.Allocator,
    path: []const u8,
    checkpoint: CheckpointView,
) SaveError!void {
    var writer = binary.SerializationWriter.init(allocator);
    defer writer.deinit();

    try writer.appendBytes(checkpoint_magic);
    try writer.appendInt(u16, checkpoint_version);
    try writer.appendInt(u64, checkpoint.step);
    try writer.appendInt(u64, checkpoint.timestamp);
    if (checkpoint.weights.len > std.math.maxInt(u32)) return CheckpointError.PayloadTooLarge;
    try writer.appendInt(u32, @intCast(checkpoint.weights.len));
    for (checkpoint.weights) |value| {
        try writer.appendInt(u32, @bitCast(value));
    }

    const bytes = try writer.toOwnedSlice();
    defer allocator.free(bytes);

    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, bytes);
}

/// Load a checkpoint from disk into owned memory.
/// @param allocator Memory allocator for allocations
/// @param path Source file path
/// @return Checkpoint with owned weights
pub fn loadCheckpoint(allocator: std.mem.Allocator, path: []const u8) LoadError!Checkpoint {
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    const data = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(64 * 1024 * 1024),
    );
    defer allocator.free(data);

    var cursor = binary.SerializationCursor.init(data);
    const magic = try cursor.readBytes(checkpoint_magic.len);
    if (!std.mem.eql(u8, magic, checkpoint_magic)) return CheckpointError.InvalidFormat;

    const version = try cursor.readInt(u16);
    if (version != checkpoint_version) return CheckpointError.UnsupportedVersion;

    const step = try cursor.readInt(u64);
    const timestamp = try cursor.readInt(u64);
    const weight_count = try cursor.readInt(u32);

    const weights = try allocator.alloc(f32, weight_count);
    errdefer allocator.free(weights);
    var i: usize = 0;
    while (i < weight_count) : (i += 1) {
        const bits = try cursor.readInt(u32);
        weights[i] = @bitCast(bits);
    }

    return .{
        .step = step,
        .timestamp = timestamp,
        .weights = weights,
    };
}

fn unixTimestamp() u64 {
    const ts = time.unixSeconds();
    if (ts <= 0) return 0;
    return @intCast(ts);
}

test "checkpoint store retains latest entries" {
    var store = CheckpointStore.init(std.testing.allocator, 2);
    defer store.deinit();

    try store.add(1, &.{ 1.0, 2.0 });
    try store.add(2, &.{ 3.0, 4.0 });
    try store.add(3, &.{ 5.0, 6.0 });

    try std.testing.expectEqual(@as(usize, 2), store.count());
    const latest = store.latest().?;
    try std.testing.expectEqual(@as(u64, 3), latest.step);
}

test "checkpoint save/load roundtrip" {
    const allocator = std.testing.allocator;

    // Use a temp file path that we can clean up
    const test_path = "test_checkpoint_roundtrip.bin";

    const weights = [_]f32{ 0.5, 1.5, 2.5 };
    const ckpt = CheckpointView{
        .step = 42,
        .timestamp = 1234,
        .weights = &weights,
    };

    // Save checkpoint
    saveCheckpoint(allocator, test_path, ckpt) catch |err| {
        std.debug.print("saveCheckpoint failed: {t}\n", .{err});
        return err;
    };

    // Cleanup: delete the test file after the test (best effort)
    defer {
        var io_backend = std.Io.Threaded.init(allocator, .{
            .environ = std.process.Environ.empty,
        });
        defer io_backend.deinit();
        const io = io_backend.io();
        std.Io.Dir.cwd().deleteFile(io, test_path) catch {};
    }

    // Load checkpoint
    var loaded = loadCheckpoint(allocator, test_path) catch |err| {
        std.debug.print("loadCheckpoint failed: {t}\n", .{err});
        return err;
    };
    defer loaded.deinit(allocator);

    try std.testing.expectEqual(@as(u64, 42), loaded.step);
    try std.testing.expectEqual(@as(u64, 1234), loaded.timestamp);
    try std.testing.expectEqualSlices(f32, &weights, loaded.weights);
}
