//! Canonical roadmap and plan catalog.
//!
//! This file is the single source of truth for:
//! - task roadmap imports
//! - generated roadmap docs
//! - generated active plan docs

const std = @import("std");

pub const Horizon = enum {
    now,
    next,
    later,

    pub fn label(self: Horizon) []const u8 {
        return switch (self) {
            .now => "Now",
            .next => "Next",
            .later => "Later",
        };
    }

    pub fn order(self: Horizon) u8 {
        return switch (self) {
            .now => 0,
            .next => 1,
            .later => 2,
        };
    }
};

pub const RoadmapStatus = enum {
    planned,
    in_progress,
    blocked,
    done,

    pub fn label(self: RoadmapStatus) []const u8 {
        return switch (self) {
            .planned => "Planned",
            .in_progress => "In Progress",
            .blocked => "Blocked",
            .done => "Done",
        };
    }

    pub fn order(self: RoadmapStatus) u8 {
        return switch (self) {
            .in_progress => 0,
            .planned => 1,
            .blocked => 2,
            .done => 3,
        };
    }
};

pub const RoadmapTrack = enum {
    gpu,
    cli_tui,
    ai,
    docs,
    platform,
    infra,

    pub fn label(self: RoadmapTrack) []const u8 {
        return switch (self) {
            .gpu => "GPU",
            .cli_tui => "CLI/TUI",
            .ai => "AI",
            .docs => "Docs",
            .platform => "Platform",
            .infra => "Infrastructure",
        };
    }

    pub fn order(self: RoadmapTrack) u8 {
        return switch (self) {
            .docs => 0,
            .gpu => 1,
            .cli_tui => 2,
            .ai => 3,
            .platform => 4,
            .infra => 5,
        };
    }
};

pub const PlanSpec = struct {
    slug: []const u8,
    title: []const u8,
    status: RoadmapStatus,
    owner: []const u8,
    scope: []const u8,
    success_criteria: []const []const u8,
    gate_commands: []const []const u8,
    milestones: []const []const u8,
};

pub const RoadmapEntry = struct {
    id: []const u8,
    title: []const u8,
    summary: []const u8,
    track: RoadmapTrack,
    horizon: Horizon,
    status: RoadmapStatus,
    owner: []const u8,
    validation_gate: []const []const u8,
    plan_slug: []const u8,
};

const owner_abbey = "Abbey";

// Shared validation gate groups. Keep these canonical so plan specs and
// roadmap entries stay synchronized when command contracts evolve.
const gate_docs_generation = [_][]const u8{};

const gate_docs_plan = [_][]const u8{};

const gate_docs_only = [_][]const u8{};

const gate_cli_core = [_][]const u8{
    "zig build cli-tests",
    "zig build verify-all",
};

const gate_cli_feature = [_][]const u8{
    "zig build feature-tests",
    "zig build cli-tests",
};

const gate_cli_plan = [_][]const u8{
    "zig build cli-tests",
    "zig build feature-tests",
    "zig build verify-all",
};

const gate_tui_core = [_][]const u8{
    "zig build cli-tests",
    "zig build tui-tests",
    "zig build dashboard-smoke",
};

const gate_gpu_core = [_][]const u8{
    "zig build typecheck",
    "zig build verify-all",
};

const gate_gpu_plan = [_][]const u8{
    "zig build typecheck",
    "zig build -Dtarget=x86_64-linux-gnu -Dgpu-backend=auto typecheck",
    "zig build -Dtarget=x86_64-windows-gnu -Dgpu-backend=auto typecheck",
    "zig build verify-all",
};

const gate_gpu_cross_target = [_][]const u8{
    "zig build -Dtarget=x86_64-linux-gnu -Dgpu-backend=auto typecheck",
    "zig build -Dtarget=x86_64-windows-gnu -Dgpu-backend=auto typecheck",
    "zig build -Dtarget=aarch64-macos -Dgpu-backend=auto typecheck",
};

const gate_feature_core = [_][]const u8{
    "zig build validate-flags",
    "zig build full-check",
};

const gate_integration_interim = [_][]const u8{
    "zig build cli-tests",
    "zig build tui-tests",
    "zig build dashboard-smoke",
};

const gate_integration_plan = [_][]const u8{
    "zig build cli-tests",
    "zig build tui-tests",
    "zig build dashboard-smoke",
    "zig build verify-all",
};

const gate_full_check_verify = [_][]const u8{
    "zig build full-check",
    "zig build verify-all",
};

const gate_verify_only = [_][]const u8{
    "zig build verify-all",
};

pub const plan_specs = [_]PlanSpec{
    .{
        .slug = "docs-roadmap-sync-v2",
        .title = "Docs + Assistant Canonical Sync",
        .status = .blocked,
        .owner = owner_abbey,
        .scope = "Deferred docs wave: keep handwritten docs aligned with the executable validation contract while generated-docs and workflow-check tooling remain unavailable.",
        .success_criteria = &.{
            "Active validation contracts name only implemented build steps and no longer claim unavailable docs tooling.",
            "Handwritten docs remain aligned around AGENTS.md, tasks/todo.md, and the executable Zig build contract.",
            "The lane stays blocked until real docs generation and workflow validation commands exist.",
        },
        .gate_commands = &gate_docs_plan,
        .milestones = &.{
            "Keep roadmap metadata and handwritten docs truthful about the currently implemented validation commands.",
            "Avoid reintroducing generated-docs or workflow-orchestration gates into the active contract until tooling lands.",
            "Resume the lane only after executable docs generation and docs verification steps exist in build.zig.",
        },
    },
    .{
        .slug = "gpu-redesign-v3",
        .title = "GPU Redesign v3",
        .status = .done,
        .owner = owner_abbey,
        .scope = "Wave 3 active lane: enforce strict backend policy, pool lifecycle safety, and cross-target policy verification.",
        .success_criteria = &.{
            "Explicit backend requests fail fast instead of silently falling back.",
            "Pool lifecycle transitions remain safe under mixed-backend execution.",
            "Cross-target policy checks stay deterministic for Linux, Windows, and macOS targets.",
        },
        .gate_commands = &gate_gpu_plan,
        .milestones = &.{
            "Wave 3A: finalize strict backend request handling across creation paths.",
            "Wave 3B: harden pool deinit/ownership rules for mixed backend graphs.",
            "Wave 3C: close remaining cross-target policy parity gaps and lock tests.",
        },
    },
    .{
        .slug = "cli-framework-local-agents",
        .title = "CLI Framework + Local-Agent Fallback",
        .status = .done,
        .owner = owner_abbey,
        .scope = "Wave 1 active lane: descriptor/runtime parity, local-first provider/plugin hardening, and command help/assertion drift cleanup.",
        .success_criteria = &.{
            "Descriptor metadata and runtime dispatch remain parity-locked for command families.",
            "Provider/plugin selection and fallback chains remain deterministic in strict and fallback modes.",
            "CLI help and assertions stay in sync with descriptor definitions.",
        },
        .gate_commands = &gate_cli_plan,
        .milestones = &.{
            "Wave 1A: close remaining descriptor/runtime parity gaps.",
            "Wave 1B: harden providers/plugins health checks and strict-mode routing.",
            "Wave 1C: remove stale help/completion/assertion drift across command families.",
            "Wave 1D: stabilize regression tests for llm run/session/providers/plugins flows.",
        },
    },
    .{
        .slug = "tui-modular-v2",
        .title = "TUI Modular Extraction v2",
        .status = .done,
        .owner = owner_abbey,
        .scope = "Wave 2 active lane: complete modular extraction, enforce layout/input correctness, and expand regression tests.",
        .success_criteria = &.{
            "Launcher and dashboard flows use shared module boundaries without behavior drift.",
            "Resize, navigation, and input handling stay correct across small and full terminal layouts.",
            "TUI layout and hit-testing regressions are covered by deterministic tests.",
        },
        .gate_commands = &gate_tui_core,
        .milestones = &.{
            "Wave 2A: complete launcher/dashboard extraction onto shared render/layout primitives.",
            "Wave 2B: close input routing and focus-state correctness gaps.",
            "Wave 2C: expand unit and integration-style TUI tests for layout and hit-testing.",
        },
    },
    .{
        .slug = "feature-modules-restructure-v1",
        .title = "Feature Modules Restructure v1",
        .status = .done,
        .owner = owner_abbey,
        .scope = "Wave 5 active lane: remove legacy facades, finalize module boundaries, and consolidate shared primitives.",
        .success_criteria = &.{
            "No stale imports remain against removed facade modules.",
            "Module boundaries are explicit and stable for feature enable/disable permutations.",
            "Shared primitives are centralized and reused without duplicate local forks.",
        },
        .gate_commands = &gate_feature_core,
        .milestones = &.{
            "Wave 5A: finish AI/service boundary cleanup and remove obsolete facade surfaces.",
            "Wave 5B: consolidate shared primitives into canonical modules.",
            "Wave 5C: update integration roots and tests to the final module topology.",
        },
    },
    .{
        .slug = "integration-gates-v1",
        .title = "Integration Gates v1",
        .status = .blocked,
        .owner = owner_abbey,
        .scope = "Wave 4 deferred lane: keep the truthful interim integration contract green while exhaustive PTY/probe validation remains unimplemented.",
        .success_criteria = &.{
            "The supported integration contract is executable end to end via cli-tests, tui-tests, dashboard-smoke, and verify-all.",
            "Dashboard smoke coverage exercises the current non-interactive fallback path instead of stale launcher/GPU help commands.",
            "Future exhaustive PTY/probe validation stays out of the canonical contract until real tooling exists.",
        },
        .gate_commands = &gate_integration_plan,
        .milestones = &.{
            "Policy guard: keep cli-tests, tui-tests, dashboard-smoke, and verify-all passing while the exhaustive lane is blocked.",
            "Unblock criterion A: implement a real PTY/probe matrix runner rather than catalog-only command strings.",
            "Unblock criterion B: deliver actionable preflight diagnostics for environment and tool blockers.",
            "Unblock criterion C: document and validate the real integration environment contract before restoring any wider gate.",
        },
    },
};

pub const roadmap_entries = [_]RoadmapEntry{
    .{
        .id = "RM-001",
        .title = "Complete canonical docs and assistant contract sync",
        .summary = "Keep handwritten docs aligned with the executable validation contract while generated-docs tooling remains blocked.",
        .track = .docs,
        .horizon = .now,
        .status = .blocked,
        .owner = owner_abbey,
        .validation_gate = &gate_docs_generation,
        .plan_slug = "docs-roadmap-sync-v2",
    },
    .{
        .id = "RM-002",
        .title = "Close GPU strictness and pool lifecycle gaps",
        .summary = "Wave 3 active: enforce strict backend requests, safe pool lifecycle, and deterministic cross-target policy checks.",
        .track = .gpu,
        .horizon = .now,
        .status = .done,
        .owner = owner_abbey,
        .validation_gate = &gate_gpu_core,
        .plan_slug = "gpu-redesign-v3",
    },
    .{
        .id = "RM-003",
        .title = "Finalize CLI descriptor framework cutover",
        .summary = "Wave 1: move remaining command routing and help/completion behavior to descriptor-first framework paths.",
        .track = .cli_tui,
        .horizon = .now,
        .status = .done,
        .owner = owner_abbey,
        .validation_gate = &gate_cli_core,
        .plan_slug = "cli-framework-local-agents",
    },
    .{
        .id = "RM-004",
        .title = "Finish TUI modular extraction",
        .summary = "Wave 2 active: complete TUI modular extraction with layout/input correctness and expanded regression tests.",
        .track = .cli_tui,
        .horizon = .now,
        .status = .done,
        .owner = owner_abbey,
        .validation_gate = &gate_tui_core,
        .plan_slug = "tui-modular-v2",
    },
    .{
        .id = "RM-005",
        .title = "Docs pipeline contract deferred pending tooling",
        .summary = "Generated docs and workflow validation remain blocked until real gendocs/check-docs tooling exists.",
        .track = .docs,
        .horizon = .now,
        .status = .blocked,
        .owner = owner_abbey,
        .validation_gate = &gate_docs_only,
        .plan_slug = "docs-roadmap-sync-v2",
    },
    .{
        .id = "RM-006",
        .title = "Automate cross-target GPU policy verification",
        .summary = "Codify policy and capability assertions for Linux, Windows, and macOS target checks.",
        .track = .platform,
        .horizon = .next,
        .status = .planned,
        .owner = owner_abbey,
        .validation_gate = &gate_gpu_cross_target,
        .plan_slug = "gpu-redesign-v3",
    },
    .{
        .id = "RM-007",
        .title = "Keep the truthful interim CLI integration gate green",
        .summary = "Wave 4 remains blocked on exhaustive PTY/probe tooling, so the supported contract is cli-tests, tui-tests, dashboard-smoke, and verify-all.",
        .track = .infra,
        .horizon = .next,
        .status = .blocked,
        .owner = owner_abbey,
        .validation_gate = &gate_integration_interim,
        .plan_slug = "integration-gates-v1",
    },
    .{
        .id = "RM-008",
        .title = "Harden local-agent provider plugins",
        .summary = "Wave 1: stabilize native ABI v1 and HTTP manifest plugin behavior in strict and fallback modes.",
        .track = .ai,
        .horizon = .now,
        .status = .done,
        .owner = owner_abbey,
        .validation_gate = &gate_cli_feature,
        .plan_slug = "cli-framework-local-agents",
    },
    .{
        .id = "RM-009",
        .title = "Complete feature module hierarchy cleanup",
        .summary = "Wave 5 active: remove legacy facades, finalize module boundaries, and consolidate shared primitives.",
        .track = .platform,
        .horizon = .now,
        .status = .done,
        .owner = owner_abbey,
        .validation_gate = &gate_feature_core,
        .plan_slug = "feature-modules-restructure-v1",
    },
    .{
        .id = "RM-010",
        .title = "Hardware acceleration research track",
        .summary = "Investigate FPGA/ASIC acceleration strategy and integration constraints for ABI workloads.",
        .track = .infra,
        .horizon = .later,
        .status = .planned,
        .owner = owner_abbey,
        .validation_gate = &gate_verify_only,
        .plan_slug = "gpu-redesign-v3",
    },
    .{
        .id = "RM-011",
        .title = "Launch developer education track",
        .summary = "Developer education content stays deferred until the docs tooling lane is unblocked and executable again.",
        .track = .docs,
        .horizon = .later,
        .status = .blocked,
        .owner = owner_abbey,
        .validation_gate = &gate_docs_only,
        .plan_slug = "docs-roadmap-sync-v2",
    },
    .{
        .id = "RM-012",
        .title = "Expand cloud function adapters",
        .summary = "Finalize adapters for Lambda, GCF, and Azure Functions with consistent deployment docs.",
        .track = .platform,
        .horizon = .later,
        .status = .planned,
        .owner = owner_abbey,
        .validation_gate = &gate_full_check_verify,
        .plan_slug = "feature-modules-restructure-v1",
    },
};

pub fn findPlanBySlug(slug: []const u8) ?*const PlanSpec {
    for (&plan_specs) |*plan| {
        if (std.mem.eql(u8, plan.slug, slug)) return plan;
    }
    return null;
}

pub fn nonDoneEntryCount() usize {
    var count: usize = 0;
    for (roadmap_entries) |entry| {
        if (entry.status != .done) count += 1;
    }
    return count;
}

pub fn formatValidationGate(
    allocator: std.mem.Allocator,
    commands: []const []const u8,
) ![]u8 {
    var joined = std.ArrayListUnmanaged(u8).empty;
    errdefer joined.deinit(allocator);

    for (commands, 0..) |command, idx| {
        if (idx > 0) try joined.appendSlice(allocator, " ; ");
        try joined.appendSlice(allocator, command);
    }

    return joined.toOwnedSlice(allocator);
}

test "catalog has plan coverage for all roadmap entries" {
    for (roadmap_entries) |entry| {
        try std.testing.expect(findPlanBySlug(entry.plan_slug) != null);
    }
}

test "nonDoneEntryCount excludes done roadmap entries" {
    var expected: usize = 0;
    for (roadmap_entries) |entry| {
        if (entry.status != .done) expected += 1;
    }
    try std.testing.expectEqual(expected, nonDoneEntryCount());
}

test {
    std.testing.refAllDecls(@This());
}
