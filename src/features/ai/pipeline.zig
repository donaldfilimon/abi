//! AI training pipeline orchestration.
//!
//! Thin, honest orchestration over the existing `training.trainWithStore` path:
//! it stands up a durable WDBX store, routes the run through a `MemoryTracker`
//! so the pipeline's memory cost is observable, executes the real (local,
//! deterministic) training-record pass, and reports metrics. It introduces no
//! new ML — it wires the established training + store primitives together.

const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const training = @import("training.zig");
const types = @import("types.zig");
const core_memory = @import("../../core/memory.zig");

/// Options for a local training-pipeline run.
pub const PipelineOptions = struct {
    /// Dataset to inspect for the run (a missing dataset degrades gracefully).
    dataset: types.DatasetSpec = .{ .path = "datasets/local-training.jsonl" },
    /// Where training artifacts would be written.
    artifact_dir: []const u8 = "zig-cache/agent-artifacts",
    /// Optional tracker. When set, the run's durable store routes its
    /// allocations through it so the pipeline's memory cost is observable.
    tracker: ?*core_memory.MemoryTracker = null,
};

/// Outcome of a pipeline run. Owns `profile` and `message`.
pub const PipelineResult = struct {
    profile: []u8,
    records_stored: usize,
    query_vector_id: ?u32,
    response_vector_id: ?u32,
    peak_memory_bytes: usize,
    message: []u8,

    pub fn deinit(self: *PipelineResult, allocator: std.mem.Allocator) void {
        if (self.profile.len > 0) allocator.free(self.profile);
        if (self.message.len > 0) allocator.free(self.message);
        self.profile = &.{};
        self.message = &.{};
    }
};

/// Run a real local training pipeline for `profile`:
///   1. validate the profile,
///   2. stand up a durable WDBX store (tracker-instrumented when provided),
///   3. execute the real `training.trainWithStore` path (profile embedding →
///      query/response vectors + block persistence), and
///   4. return observable metrics (records, vector ids, peak memory).
pub fn run(
    allocator: std.mem.Allocator,
    profile: []const u8,
    options: PipelineOptions,
) !PipelineResult {
    if (profile.len == 0) return error.InvalidTrainingProfile;

    var store = wdbx.Store.init(allocator);
    defer store.deinit();
    if (options.tracker) |t| store.setTracker(t);

    var result = try training.trainWithStore(allocator, &store, .{
        .profile = profile,
        .dataset = options.dataset,
        .artifact_dir = options.artifact_dir,
        .tracker = options.tracker,
    });
    defer result.deinit(allocator);

    // Read peak before the store is torn down by the deferred deinit; peak usage
    // is monotonic, so it reflects the run's high-water mark.
    const peak: usize = if (options.tracker) |t| t.getPeakUsage() else 0;

    const message = try std.fmt.allocPrint(
        allocator,
        "pipeline complete; profile={s}; records_stored={d}; peak_memory_bytes={d}",
        .{ profile, result.records_stored, peak },
    );
    errdefer allocator.free(message);

    return .{
        .profile = try allocator.dupe(u8, profile),
        .records_stored = result.records_stored,
        .query_vector_id = result.query_vector_id,
        .response_vector_id = result.response_vector_id,
        .peak_memory_bytes = peak,
        .message = message,
    };
}

/// Backwards-compatible convenience: run the pipeline for `profile` with default
/// options under a transient tracker and log the outcome. Preserves the
/// empty-profile contract (`error.InvalidTrainingProfile`).
pub fn train(profile: []const u8) !void {
    if (profile.len == 0) return error.InvalidTrainingProfile;
    const allocator = std.heap.page_allocator;
    var tracker = core_memory.MemoryTracker.init(allocator);
    defer tracker.deinit();
    var result = run(allocator, profile, .{ .tracker = &tracker }) catch |err| {
        std.log.err("training pipeline failed for profile {s}: {s}", .{ profile, @errorName(err) });
        return err;
    };
    defer result.deinit(allocator);
    std.log.info("{s}", .{result.message});
}

test {
    std.testing.refAllDecls(@This());
}

test "pipeline rejects empty profile" {
    try std.testing.expectError(error.InvalidTrainingProfile, train(""));
}

test "pipeline run trains a profile and reports observable memory" {
    if (!build_options.feat_wdbx or !build_options.feat_ai) return;
    const allocator = std.testing.allocator;

    var tracker = core_memory.MemoryTracker.init(allocator);
    defer tracker.deinit();

    var result = try run(allocator, "abbey", .{ .tracker = &tracker });
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("abbey", result.profile);
    try std.testing.expect(result.records_stored >= 1);
    // The training path routed store allocations through the tracker, so the
    // pipeline's memory high-water mark is observable.
    try std.testing.expect(result.peak_memory_bytes > 0);
    try std.testing.expect(std.mem.indexOf(u8, result.message, "pipeline complete") != null);
}

test "pipeline run rejects an unknown profile" {
    if (!build_options.feat_wdbx or !build_options.feat_ai) return;
    try std.testing.expectError(error.InvalidTrainingProfile, run(std.testing.allocator, "not-a-profile", .{}));
}
