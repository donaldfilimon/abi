//! Per-turn pipeline telemetry snapshot (WDBX Rust `PipelineTelemetry`).
//!
//! The Rust SDK makes telemetry emission *mandatory* after every turn: a single
//! snapshot folds the constitutional ethics scores, neural telemetry, provider /
//! retrieval summaries, cluster status, governance/prompt versions, guardrail
//! summary, latency, and error counts together (see
//! `docs/spec/wdbx-rust-capability-extract.mdx` §3). ABI already has the bounded
//! six-principle E-score pass (`constitution.zig`) and fire-and-forget telemetry
//! counters (`telemetry/`); this module is the missing seam — a single
//! `PipelineTelemetry` value emitted by `ObservabilityHub.finishTurn`.
//!
//! Honesty note: this is the aggregation/discipline layer only. The individual
//! subscores come from the already-shipped `constitution` audit; the neural
//! telemetry comes from `point_neural_net`. No new ML is introduced here.

const std = @import("std");
const constitution = @import("constitution.zig");
const point_neural_net = @import("point_neural_net.zig");

pub const Principle = constitution.Principle;
pub const AuditResult = constitution.AuditResult;

/// Read-only view of the neural hub's telemetry, copied into a snapshot so the
/// snapshot owns its `l1_norms` slice. Mirrors `PointNeuralNetwork.telemetry()`.
pub const NeuralTelemetryView = struct {
    layer_count: usize,
    total_weights: usize,
    nonzero_weights: usize,
    l1_norms: []const f32,
};

/// Mandatory per-turn telemetry snapshot (Rust `PipelineTelemetry`). Every
/// field is readable; the `[]u8` members are owned by the snapshot and freed by
/// `deinit`. `ethical_scores` / `neural.l1_norms` are fixed-width for
/// determinism.
pub const PipelineTelemetry = struct {
    // Ethics: six ABI principles + weighted E-score + hard safety veto.
    ethical_scores: [6]f32,
    escore: f32,
    vetoed: bool,

    // Neural hub telemetry (optional; zeroed when no network was trained).
    neural: NeuralTelemetryView,

    // Provider / retrieval summaries.
    provider: []u8,
    retrieval_summary: []u8,

    // Cluster status (free-form, e.g. "single-node").
    cluster_status: []u8,

    // Governance / prompt versions surfaced in the snapshot.
    governance_version: []u8,
    prompt_version: []u8,

    // Guardrail summary.
    guardrail_summary: []u8,

    // Latency + counts.
    p99_latency_ms: f64,
    total_errors: u64,
    responses_total: u64,

    pub fn deinit(self: *PipelineTelemetry, allocator: std.mem.Allocator) void {
        allocator.free(self.provider);
        allocator.free(self.retrieval_summary);
        allocator.free(self.cluster_status);
        allocator.free(self.governance_version);
        allocator.free(self.prompt_version);
        allocator.free(self.guardrail_summary);
        if (self.neural.l1_norms.len > 0) allocator.free(self.neural.l1_norms);
    }
};

/// Options for `ObservabilityHub.snapshot`. All `[]const u8` members become
/// owned copies inside the resulting `PipelineTelemetry`.
pub const SnapshotOptions = struct {
    provider: []const u8 = "local",
    retrieval_summary: []const u8 = "",
    cluster_status: []const u8 = "single-node",
    governance_version: []const u8 = "constitution:6.0",
    prompt_version: []const u8 = "prompt:1.0",
    guardrail_summary: []const u8 = "ok",
    p99_latency_ms: f64 = 0,
};

/// Aggregator mirroring the Rust `ObservabilityHub`: holds response / error
/// counters and emits a `PipelineTelemetry` snapshot per turn. Counter updates
/// are fire-and-forget; the snapshot read is the mandatory `finishTurn` tail.
pub const ObservabilityHub = struct {
    allocator: std.mem.Allocator,
    responses_total: u64 = 0,
    total_errors: u64 = 0,
    constitution_blocks: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) ObservabilityHub {
        return .{ .allocator = allocator };
    }

    /// A completed turn (regardless of pass/fail) increments the response count.
    pub fn recordResponse(self: *ObservabilityHub) void {
        self.responses_total += 1;
    }

    /// A generic error (network/parse/runtime) on a turn.
    pub fn recordError(self: *ObservabilityHub) void {
        self.total_errors += 1;
    }

    /// A constitutional block. Per the Rust design, `total_errors` folds in
    /// `constitution_blocks` so a blocked response is observable in the snapshot.
    pub fn recordConstitutionBlock(self: *ObservabilityHub) void {
        self.constitution_blocks += 1;
        self.total_errors += 1;
    }

    /// The mandatory per-turn tail: build a `PipelineTelemetry` from the audit
    /// result, an optional neural telemetry reading, and the current counters.
    /// Owned string fields are duplicated; caller frees via `PipelineTelemetry.deinit`.
    pub fn snapshot(
        self: *ObservabilityHub,
        allocator: std.mem.Allocator,
        audit: AuditResult,
        maybe_neural: ?point_neural_net.NeuralTelemetry,
        opts: SnapshotOptions,
    ) !PipelineTelemetry {
        var neural: NeuralTelemetryView = .{
            .layer_count = 0,
            .total_weights = 0,
            .nonzero_weights = 0,
            .l1_norms = &.{},
        };
        var owned_l1: ?[]f32 = null;
        if (maybe_neural) |n| {
            owned_l1 = try allocator.dupe(f32, n.l1_norms);
            neural = .{
                .layer_count = n.layer_count,
                .total_weights = n.total_weights,
                .nonzero_weights = n.nonzero_weights,
                .l1_norms = owned_l1.?,
            };
        }
        errdefer if (owned_l1) |l| allocator.free(l);

        return .{
            .ethical_scores = audit.scores,
            .escore = audit.escore,
            .vetoed = audit.vetoed,
            .neural = neural,
            .provider = try allocator.dupe(u8, opts.provider),
            .retrieval_summary = try allocator.dupe(u8, opts.retrieval_summary),
            .cluster_status = try allocator.dupe(u8, opts.cluster_status),
            .governance_version = try allocator.dupe(u8, opts.governance_version),
            .prompt_version = try allocator.dupe(u8, opts.prompt_version),
            .guardrail_summary = try allocator.dupe(u8, opts.guardrail_summary),
            .p99_latency_ms = opts.p99_latency_ms,
            .total_errors = self.total_errors,
            .responses_total = self.responses_total,
        };
    }

    /// Convenience wrapper that records the response and (when the audit vetoed
    /// or failed) the constitution block, then emits the snapshot. This is the
    /// single "finish turn" call the rest of the pipeline should make.
    pub fn finishTurn(
        self: *ObservabilityHub,
        allocator: std.mem.Allocator,
        audit: AuditResult,
        maybe_neural: ?point_neural_net.NeuralTelemetry,
        opts: SnapshotOptions,
    ) !PipelineTelemetry {
        self.recordResponse();
        if (!audit.passed) self.recordConstitutionBlock();
        return try self.snapshot(allocator, audit, maybe_neural, opts);
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "observability hub snapshots ethics and counters" {
    const allocator = std.testing.allocator;
    var hub = ObservabilityHub.init(allocator);

    const audit = constitution.Constitution.validate("this is a safe and helpful response for everyone");
    try std.testing.expect(audit.passed);
    try std.testing.expect(!audit.vetoed);

    var snap = try hub.finishTurn(allocator, audit, null, .{});
    defer snap.deinit(allocator);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), snap.escore, 1e-5);
    try std.testing.expect(!snap.vetoed);
    try std.testing.expectEqual(@as(u64, 1), snap.responses_total);
    try std.testing.expectEqual(@as(u64, 0), snap.total_errors);
    // No network trained -> neural telemetry is zeroed.
    try std.testing.expectEqual(@as(usize, 0), snap.neural.layer_count);
    // String members were owned-copied and free without leaking.
    try std.testing.expectEqualStrings("local", snap.provider);
    try std.testing.expectEqualStrings("single-node", snap.cluster_status);
}

test "observability hub folds a constitution block into total_errors" {
    const allocator = std.testing.allocator;
    var hub = ObservabilityHub.init(allocator);

    // "harm" trips the hard safety veto and fails the audit.
    const audit = constitution.Constitution.validate("this could cause harm to users");
    try std.testing.expect(audit.vetoed);
    try std.testing.expect(!audit.passed);

    var snap = try hub.finishTurn(allocator, audit, null, .{});
    defer snap.deinit(allocator);

    try std.testing.expectEqual(@as(u64, 1), snap.responses_total);
    try std.testing.expectEqual(@as(u64, 1), snap.total_errors);
}

test "observability hub includes neural telemetry when provided" {
    const allocator = std.testing.allocator;
    var hub = ObservabilityHub.init(allocator);

    var net = try point_neural_net.PointNeuralNetwork.init(allocator, &[_]usize{ 3, 4, 1 }, 0.1);
    defer net.deinit();
    const neural = point_neural_net.PointNeuralNetwork.telemetry(&net);

    const audit = constitution.Constitution.validate("hello world");
    var snap = try hub.snapshot(allocator, audit, neural, .{ .provider = "abi-test" });
    defer snap.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), snap.neural.layer_count);
    try std.testing.expect(snap.neural.total_weights > 0);
    try std.testing.expectEqual(@as(usize, 2), snap.neural.l1_norms.len);
    try std.testing.expectEqualStrings("abi-test", snap.provider);

    neural.deinit(allocator);
}
