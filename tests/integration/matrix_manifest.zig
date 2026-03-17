//! Integration Test Matrix Manifest
//!
//! Comptime data structure defining exhaustive command × flag-combination
//! test vectors for CLI integration gates. The build system iterates this
//! manifest to generate deterministic, isolated integration test runs.
//!
//! Consumed by the build system (via relative import) and by standalone
//! test harnesses that `@import("abi")`.

const std = @import("std");
const flags = @import("../../build/flags.zig");
const FlagCombo = flags.FlagCombo;

// ── Timeout Policy ──────────────────────────────────────────────────────

/// Timeout tiers for integration test vectors.
pub const TimeoutTier = enum {
    /// Fast commands: help, version, list, status (5 s).
    quick,
    /// Standard CLI commands: config, model, db ops (30 s).
    cli,
    /// TUI / interactive commands: ui, editor, chat (60 s).
    tui,

    /// Return the deadline in nanoseconds for this tier.
    pub fn toNs(self: TimeoutTier) u64 {
        return switch (self) {
            .quick => quick_timeout_ns,
            .cli => cli_timeout_ns,
            .tui => tui_timeout_ns,
        };
    }

    /// Return the deadline in whole seconds for display.
    pub fn toSeconds(self: TimeoutTier) u64 {
        return self.toNs() / std.time.ns_per_s;
    }
};

/// Default timeout for fast / read-only commands (5 seconds).
pub const quick_timeout_ns: u64 = 5 * std.time.ns_per_s;

/// Default timeout for standard CLI commands (30 seconds).
pub const cli_timeout_ns: u64 = 30 * std.time.ns_per_s;

/// Default timeout for TUI / interactive commands (60 seconds).
pub const tui_timeout_ns: u64 = 60 * std.time.ns_per_s;

// ── Test Expectation ────────────────────────────────────────────────────

/// Expected outcome of an integration test vector.
pub const TestExpectation = struct {
    /// Expected process exit code (0 = success).
    exit_code: u8 = 0,
    /// When true, the vector is expected to fail (non-zero exit).
    expect_failure: bool = false,
};

// ── Feature Requirements ────────────────────────────────────────────────

/// Bitfield of feature-flag requirements for a test vector.
/// A vector runs only when *all* required flags are enabled in the active
/// `FlagCombo`.
pub const FeatureReqs = struct {
    feat_ai: bool = false,
    feat_gpu: bool = false,
    feat_web: bool = false,
    feat_database: bool = false,
    feat_network: bool = false,
    feat_profiling: bool = false,
    feat_analytics: bool = false,
    feat_cloud: bool = false,
    feat_training: bool = false,
    feat_reasoning: bool = false,
    feat_auth: bool = false,
    feat_messaging: bool = false,
    feat_cache: bool = false,
    feat_storage: bool = false,
    feat_search: bool = false,
    feat_mobile: bool = false,
    feat_gateway: bool = false,
    feat_pages: bool = false,
    feat_benchmarks: bool = false,
    feat_compute: bool = false,
    feat_documents: bool = false,
    feat_desktop: bool = false,
    feat_lsp: bool = false,
    feat_mcp: bool = false,
    feat_llm: bool = false,
    feat_explore: bool = false,
    feat_vision: bool = false,

    /// No feature requirements — vector runs under any combo.
    pub const none: FeatureReqs = .{};

    /// Returns true when `combo` satisfies every required flag.
    pub fn satisfiedBy(self: FeatureReqs, combo: FlagCombo) bool {
        inline for (std.meta.fields(FeatureReqs)) |field| {
            if (@field(self, field.name)) {
                if (!@field(combo, field.name)) return false;
            }
        }
        return true;
    }
};

// ── Integration Vector ──────────────────────────────────────────────────

/// A single integration test vector: a CLI invocation with its expected
/// outcome, timeout tier, and feature-flag requirements.
pub const IntegrationVector = struct {
    /// Human-readable identifier (e.g. "help.gpu", "db.stats").
    id: []const u8,
    /// Command-line arguments (excluding the binary name).
    args: []const []const u8,
    /// Expected outcome.
    expectation: TestExpectation = .{},
    /// Timeout tier governing how long this vector may run.
    timeout: TimeoutTier = .quick,
    /// Feature flags that must be enabled for this vector to apply.
    required_features: FeatureReqs = FeatureReqs.none,
    /// Category tag for reporting and filtering.
    category: CommandCategory = .core,
};

/// Broad category for grouping test vectors in reports.
pub const CommandCategory = enum {
    core,
    ai,
    gpu,
    database,
    network,
    training,
    ui,
    config,
    model,
    bench,
    search,
    cloud,
    auth,
    messaging,
    storage,
    lsp,
    mcp,
    documents,
    desktop,
    profiling,
};

// ── Vector Catalogue ────────────────────────────────────────────────────

/// Exhaustive list of integration test vectors covering the CLI command
/// tree. Vectors are grouped by category for readability.
pub const all_vectors = core_vectors //
    ++ ai_vectors //
    ++ gpu_vectors //
    ++ db_vectors //
    ++ config_vectors //
    ++ model_vectors //
    ++ bench_vectors //
    ++ train_vectors //
    ++ ui_vectors //
    ++ network_vectors //
    ++ lsp_vectors //
    ++ mcp_vectors //
    ++ profile_vectors //
    ++ task_vectors //
    ++ search_vectors //
    ++ cloud_vectors //
    ++ auth_vectors //
    ++ messaging_vectors //
    ++ storage_vectors //
    ++ documents_vectors //
    ++ desktop_vectors //
    ++ profiling_vectors;

// ── Core commands (no feature requirements) ─────────────────────────────

const core_vectors = [_]IntegrationVector{
    .{
        .id = "top.help",
        .args = &.{"help"},
        .category = .core,
    },
    .{
        .id = "top.version",
        .args = &.{"version"},
        .category = .core,
    },
    .{
        .id = "top.flag-help",
        .args = &.{"--help"},
        .category = .core,
    },
    .{
        .id = "status.show",
        .args = &.{"status"},
        .category = .core,
    },
    .{
        .id = "completions.help",
        .args = &.{ "help", "completions" },
        .category = .core,
    },
    .{
        .id = "doctor.help",
        .args = &.{ "help", "doctor" },
        .category = .core,
    },
    .{
        .id = "clean.help",
        .args = &.{ "help", "clean" },
        .category = .core,
    },
    .{
        .id = "env.help",
        .args = &.{ "help", "env" },
        .category = .core,
    },
    .{
        .id = "init.help",
        .args = &.{ "help", "init" },
        .category = .core,
    },
    .{
        .id = "unknown.command",
        .args = &.{"__nonexistent_command__"},
        .expectation = .{ .expect_failure = true },
        .category = .core,
    },
};

// ── AI commands ─────────────────────────────────────────────────────────

const ai_vectors = [_]IntegrationVector{
    .{
        .id = "llm.providers",
        .args = &.{ "llm", "providers" },
        .category = .ai,
        .required_features = .{ .feat_ai = true, .feat_llm = true },
    },
    .{
        .id = "llm.plugins.list",
        .args = &.{ "llm", "plugins", "list" },
        .category = .ai,
        .required_features = .{ .feat_ai = true, .feat_llm = true },
    },
    .{
        .id = "llm.help",
        .args = &.{ "help", "llm" },
        .category = .ai,
        .required_features = .{ .feat_ai = true, .feat_llm = true },
    },
    .{
        .id = "embed.help",
        .args = &.{ "help", "embed" },
        .category = .ai,
        .required_features = .{ .feat_ai = true },
    },
    .{
        .id = "explore.help",
        .args = &.{ "help", "explore" },
        .category = .ai,
        .required_features = .{ .feat_ai = true, .feat_explore = true },
    },
    .{
        .id = "agent.help",
        .args = &.{ "help", "agent" },
        .category = .ai,
        .required_features = .{ .feat_ai = true },
    },
    .{
        .id = "multi-agent.help",
        .args = &.{ "help", "multi-agent" },
        .category = .ai,
        .required_features = .{ .feat_ai = true },
    },
    .{
        .id = "os-agent.help",
        .args = &.{ "help", "os-agent" },
        .category = .ai,
        .required_features = .{ .feat_ai = true },
    },
    .{
        .id = "reasoning.help",
        .args = &.{ "help", "ralph" },
        .category = .ai,
        .required_features = .{ .feat_ai = true, .feat_reasoning = true },
    },
    .{
        .id = "brain.help",
        .args = &.{ "help", "brain" },
        .category = .ai,
        .required_features = .{ .feat_ai = true },
    },
    .{
        .id = "convert.help",
        .args = &.{ "help", "convert" },
        .category = .ai,
        .required_features = .{ .feat_ai = true },
    },
};

// ── GPU commands ────────────────────────────────────────────────────────

const gpu_vectors = [_]IntegrationVector{
    .{
        .id = "gpu.summary",
        .args = &.{ "gpu", "summary" },
        .category = .gpu,
        .required_features = .{ .feat_gpu = true },
    },
    .{
        .id = "gpu.devices",
        .args = &.{ "gpu", "devices" },
        .category = .gpu,
        .required_features = .{ .feat_gpu = true },
    },
    .{
        .id = "gpu.backends",
        .args = &.{ "gpu", "backends" },
        .category = .gpu,
        .required_features = .{ .feat_gpu = true },
    },
    .{
        .id = "gpu.help",
        .args = &.{ "help", "gpu" },
        .category = .gpu,
        .required_features = .{ .feat_gpu = true },
    },
    .{
        .id = "simd.help",
        .args = &.{ "help", "simd" },
        .category = .gpu,
    },
};

// ── Database commands ───────────────────────────────────────────────────

const db_vectors = [_]IntegrationVector{
    .{
        .id = "db.stats",
        .args = &.{ "db", "stats" },
        .category = .database,
        .required_features = .{ .feat_database = true },
    },
    .{
        .id = "db.help",
        .args = &.{ "help", "db" },
        .category = .database,
        .required_features = .{ .feat_database = true },
    },
};

// ── Config commands ─────────────────────────────────────────────────────

const config_vectors = [_]IntegrationVector{
    .{
        .id = "config.show",
        .args = &.{ "config", "show" },
        .category = .config,
    },
    .{
        .id = "config.path",
        .args = &.{ "config", "path" },
        .category = .config,
    },
    .{
        .id = "config.help",
        .args = &.{ "help", "config" },
        .category = .config,
    },
};

// ── Model commands ──────────────────────────────────────────────────────

const model_vectors = [_]IntegrationVector{
    .{
        .id = "model.list",
        .args = &.{ "model", "list" },
        .category = .model,
        .required_features = .{ .feat_ai = true },
    },
    .{
        .id = "model.path",
        .args = &.{ "model", "path" },
        .category = .model,
        .required_features = .{ .feat_ai = true },
    },
    .{
        .id = "model.help",
        .args = &.{ "help", "model" },
        .category = .model,
        .required_features = .{ .feat_ai = true },
    },
};

// ── Bench commands ──────────────────────────────────────────────────────

const bench_vectors = [_]IntegrationVector{
    .{
        .id = "bench.list",
        .args = &.{ "bench", "list" },
        .category = .bench,
        .required_features = .{ .feat_benchmarks = true },
    },
    .{
        .id = "bench.micro.hash",
        .args = &.{ "bench", "micro", "hash" },
        .timeout = .cli,
        .category = .bench,
        .required_features = .{ .feat_benchmarks = true },
    },
    .{
        .id = "bench.micro.noop",
        .args = &.{ "bench", "micro", "noop" },
        .timeout = .cli,
        .category = .bench,
        .required_features = .{ .feat_benchmarks = true },
    },
    .{
        .id = "bench.help",
        .args = &.{ "help", "bench" },
        .category = .bench,
        .required_features = .{ .feat_benchmarks = true },
    },
};

// ── Train commands ──────────────────────────────────────────────────────

const train_vectors = [_]IntegrationVector{
    .{
        .id = "train.info",
        .args = &.{ "train", "info" },
        .category = .training,
        .required_features = .{ .feat_ai = true, .feat_training = true },
    },
    .{
        .id = "train.help",
        .args = &.{ "help", "train" },
        .category = .training,
        .required_features = .{ .feat_ai = true, .feat_training = true },
    },
};

// ── UI / TUI commands ───────────────────────────────────────────────────

const ui_vectors = [_]IntegrationVector{
    .{
        .id = "ui.help",
        .args = &.{ "ui", "--help" },
        .category = .ui,
    },
    .{
        .id = "ui.list-themes",
        .args = &.{ "ui", "--list-themes" },
        .category = .ui,
    },
    .{
        .id = "ui.gpu.help",
        .args = &.{ "ui", "gpu", "--help" },
        .category = .ui,
        .required_features = .{ .feat_gpu = true },
    },
    .{
        .id = "ui.brain.help",
        .args = &.{ "ui", "brain", "--help" },
        .category = .ui,
        .required_features = .{ .feat_ai = true },
    },
    .{
        .id = "ui.model.help",
        .args = &.{ "ui", "model", "--help" },
        .category = .ui,
        .required_features = .{ .feat_ai = true },
    },
    .{
        .id = "ui.editor.help",
        .args = &.{ "ui", "editor", "--help" },
        .category = .ui,
    },
    .{
        .id = "ui.dashboard.alias",
        .args = &.{ "ui", "dashboard" },
        .timeout = .tui,
        .category = .ui,
    },
    .{
        .id = "ui.launch.removed",
        .args = &.{ "ui", "launch" },
        .expectation = .{ .expect_failure = true },
        .category = .ui,
    },
    .{
        .id = "editor.help",
        .args = &.{ "editor", "--help" },
        .category = .ui,
    },
};

// ── Network commands ────────────────────────────────────────────────────

const network_vectors = [_]IntegrationVector{
    .{
        .id = "network.help",
        .args = &.{ "help", "network" },
        .category = .network,
        .required_features = .{ .feat_network = true },
    },
};

// ── LSP commands ────────────────────────────────────────────────────────

const lsp_vectors = [_]IntegrationVector{
    .{
        .id = "lsp.help",
        .args = &.{ "help", "lsp" },
        .category = .lsp,
        .required_features = .{ .feat_lsp = true },
    },
};

// ── MCP commands ────────────────────────────────────────────────────────

const mcp_vectors = [_]IntegrationVector{
    .{
        .id = "mcp.help",
        .args = &.{ "help", "mcp" },
        .category = .mcp,
        .required_features = .{ .feat_mcp = true },
    },
    .{
        .id = "acp.help",
        .args = &.{ "help", "acp" },
        .category = .mcp,
    },
};

// ── Profile commands ────────────────────────────────────────────────────

const profile_vectors = [_]IntegrationVector{
    .{
        .id = "profile.show",
        .args = &.{ "profile", "show" },
        .category = .core,
    },
    .{
        .id = "profile.help",
        .args = &.{ "help", "profile" },
        .category = .core,
    },
};

// ── Task commands ───────────────────────────────────────────────────────

const task_vectors = [_]IntegrationVector{
    .{
        .id = "task.list",
        .args = &.{ "task", "list" },
        .category = .core,
    },
    .{
        .id = "task.stats",
        .args = &.{ "task", "stats" },
        .category = .core,
    },
    .{
        .id = "task.help",
        .args = &.{ "help", "task" },
        .category = .core,
    },
};

// ── Search commands ─────────────────────────────────────────────────────

const search_vectors = [_]IntegrationVector{
    .{
        .id = "search.help",
        .args = &.{ "help", "search" },
        .category = .search,
        .required_features = .{ .feat_search = true },
    },
};

// ── Cloud commands ──────────────────────────────────────────────────────

const cloud_vectors = [_]IntegrationVector{
    .{
        .id = "cloud.help",
        .args = &.{ "help", "cloud" },
        .category = .cloud,
        .required_features = .{ .feat_cloud = true },
    },
};

// ── Auth commands ───────────────────────────────────────────────────────

const auth_vectors = [_]IntegrationVector{
    .{
        .id = "auth.help",
        .args = &.{ "help", "auth" },
        .category = .auth,
        .required_features = .{ .feat_auth = true },
    },
};

// ── Messaging commands ──────────────────────────────────────────────────

const messaging_vectors = [_]IntegrationVector{
    .{
        .id = "discord.help",
        .args = &.{ "help", "discord" },
        .category = .messaging,
        .required_features = .{ .feat_messaging = true },
    },
};

// ── Storage commands ────────────────────────────────────────────────────

const storage_vectors = [_]IntegrationVector{
    .{
        .id = "storage.help",
        .args = &.{ "help", "storage" },
        .category = .storage,
        .required_features = .{ .feat_storage = true },
    },
};

// ── Documents commands ──────────────────────────────────────────────────

const documents_vectors = [_]IntegrationVector{
    .{
        .id = "documents.help",
        .args = &.{ "help", "documents" },
        .category = .documents,
        .required_features = .{ .feat_documents = true },
    },
};

// ── Desktop commands ────────────────────────────────────────────────────

const desktop_vectors = [_]IntegrationVector{
    .{
        .id = "desktop.help",
        .args = &.{ "help", "desktop" },
        .category = .desktop,
        .required_features = .{ .feat_desktop = true },
    },
};

// ── Profiling commands ──────────────────────────────────────────────────

const profiling_vectors = [_]IntegrationVector{
    .{
        .id = "profiling.help",
        .args = &.{ "help", "profiling" },
        .category = .profiling,
        .required_features = .{ .feat_profiling = true },
    },
};

// ── Matrix Query Helpers ────────────────────────────────────────────────

/// Count how many vectors are applicable for a given flag combo.
pub fn countVectorsForCombo(combo: FlagCombo) usize {
    var count: usize = 0;
    for (all_vectors) |vec| {
        if (vec.required_features.satisfiedBy(combo)) {
            count += 1;
        }
    }
    return count;
}

/// Return the subset of `all_vectors` indices whose feature requirements
/// are satisfied by `combo`. Caller provides a fixed-size buffer.
pub fn vectorIndicesForCombo(
    combo: FlagCombo,
    buf: *[all_vectors.len]usize,
) []const usize {
    var n: usize = 0;
    for (all_vectors, 0..) |vec, i| {
        if (vec.required_features.satisfiedBy(combo)) {
            buf[i] = i;
            // Pack indices at the front.
            buf[n] = i;
            n += 1;
        }
    }
    return buf[0..n];
}

/// Reference to the canonical flag validation matrix from `build/flags.zig`.
pub const flag_combos = flags.validation_matrix;

/// Total number of matrix cells (combos × applicable vectors).
pub fn totalMatrixCells() usize {
    var total: usize = 0;
    for (flag_combos) |combo| {
        total += countVectorsForCombo(combo);
    }
    return total;
}

// ── Comptime Validation ─────────────────────────────────────────────────

comptime {
    // Ensure every vector has a non-empty id and args.
    for (all_vectors) |vec| {
        if (vec.id.len == 0)
            @compileError("IntegrationVector with empty id");
        if (vec.args.len == 0)
            @compileError("IntegrationVector '" ++ vec.id ++ "' has empty args");
    }

    // Ensure no duplicate vector ids.
    for (all_vectors, 0..) |a, i| {
        for (all_vectors[0..i]) |b| {
            if (std.mem.eql(u8, a.id, b.id))
                @compileError("Duplicate IntegrationVector id: " ++ a.id);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

test "all_vectors comptime validation passes" {
    // The comptime block above already validates; this test ensures the
    // module is reachable from the test runner.
    try std.testing.expect(all_vectors.len > 0);
}

test "all-enabled combo covers every vector" {
    const all_enabled = flag_combos[0]; // "all-enabled"
    try std.testing.expectEqualStrings("all-enabled", all_enabled.name);
    try std.testing.expectEqual(all_vectors.len, countVectorsForCombo(all_enabled));
}

test "all-disabled combo covers only coreless vectors" {
    const all_disabled = flag_combos[1]; // "all-disabled"
    try std.testing.expectEqualStrings("all-disabled", all_disabled.name);
    const count = countVectorsForCombo(all_disabled);
    // All-disabled should only match vectors with no feature requirements.
    try std.testing.expect(count > 0);
    try std.testing.expect(count < all_vectors.len);
}

test "vectorIndicesForCombo returns valid indices" {
    var buf: [all_vectors.len]usize = undefined;
    const indices = vectorIndicesForCombo(flag_combos[0], &buf);
    try std.testing.expect(indices.len > 0);
    for (indices) |idx| {
        try std.testing.expect(idx < all_vectors.len);
    }
}

test "timeout tiers have correct values" {
    try std.testing.expectEqual(@as(u64, 5 * std.time.ns_per_s), TimeoutTier.quick.toNs());
    try std.testing.expectEqual(@as(u64, 30 * std.time.ns_per_s), TimeoutTier.cli.toNs());
    try std.testing.expectEqual(@as(u64, 60 * std.time.ns_per_s), TimeoutTier.tui.toNs());
}

test "totalMatrixCells is positive" {
    const cells = totalMatrixCells();
    try std.testing.expect(cells > 0);
}
