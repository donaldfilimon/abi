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

pub const plan_specs = [_]PlanSpec{
    .{
        .slug = "docs-roadmap-sync-v2",
        .title = "Docs + Roadmap Canonical Sync",
        .status = .in_progress,
        .owner = "Abbey",
        .scope = "Canonical roadmap catalog, generated roadmap docs, generated plan docs, and task import synchronization.",
        .success_criteria = &.{
            "Roadmap and plans are generated from one catalog source.",
            "Task roadmap import reads canonical entries and skips done items.",
            "check-docs fails on roadmap/plans drift.",
        },
        .gate_commands = &.{
            "zig build gendocs",
            "zig build check-docs",
            "zig build verify-all",
        },
        .milestones = &.{
            "Introduce roadmap_catalog.zig with typed entries.",
            "Wire gendocs roadmap + plans renderers to canonical source.",
            "Enable plans drift checks and archive handling.",
        },
    },
    .{
        .slug = "gpu-redesign-v3",
        .title = "GPU Redesign v3",
        .status = .in_progress,
        .owner = "Abbey",
        .scope = "Metal/Vulkan policy hardening, GL family consolidation, strict backend creation, and mixed-backend stability.",
        .success_criteria = &.{
            "Explicit backend requests stay strict.",
            "Auto policy is deterministic per target.",
            "Mixed-backend execution remains stable under pool lifecycle checks.",
        },
        .gate_commands = &.{
            "zig build typecheck",
            "zig build -Dtarget=x86_64-linux-gnu -Dgpu-backend=auto typecheck",
            "zig build -Dtarget=x86_64-windows-gnu -Dgpu-backend=auto typecheck",
            "zig build verify-all",
        },
        .milestones = &.{
            "Complete backend registry/pool strictness enforcement.",
            "Finalize GL profile wrappers over shared runtime.",
            "Close cross-target compile and policy consistency gaps.",
        },
    },
    .{
        .slug = "cli-framework-local-agents",
        .title = "CLI Framework + Local-Agent Fallback",
        .status = .in_progress,
        .owner = "Abbey",
        .scope = "Descriptor-driven CLI framework and local-first LLM provider routing with plugin support.",
        .success_criteria = &.{
            "LLM command family runs through provider router.",
            "Fallback chain is deterministic and configurable.",
            "CLI command metadata and runtime dispatch share one source.",
        },
        .gate_commands = &.{
            "zig build cli-tests",
            "zig build feature-tests",
            "zig build verify-all",
        },
        .milestones = &.{
            "Use canonical command catalog as metadata source for descriptors/spec/matrix.",
            "Finalize llm run/session/providers/plugins command tree.",
            "Harden provider health checks and strict backend mode.",
            "Align TUI command preview with descriptor graph.",
        },
    },
    .{
        .slug = "tui-modular-v2",
        .title = "TUI Modular Extraction v2",
        .status = .in_progress,
        .owner = "Abbey",
        .scope = "Split launcher/dashboard rendering into reusable modules with responsive layout and shared async loop behavior.",
        .success_criteria = &.{
            "Launcher execution path is unified across enter/search/mouse.",
            "Resize behavior is immediate and stable across panels.",
            "Small terminal fallback rendering remains readable.",
        },
        .gate_commands = &.{
            "zig build cli-tests",
            "zig build run -- ui launch --help",
            "zig build run -- ui gpu --help",
        },
        .milestones = &.{
            "Finalize launcher split modules and helpers.",
            "Migrate dashboards to shared layout/render primitives.",
            "Expand TUI unit tests for layout and hit-testing.",
        },
    },
    .{
        .slug = "feature-modules-restructure-v1",
        .title = "Feature Modules Restructure v1",
        .status = .planned,
        .owner = "Abbey",
        .scope = "Consolidate feature layout, remove obsolete facades, and align mod/stub parity under new module boundaries.",
        .success_criteria = &.{
            "No stale imports to removed facade modules.",
            "Feature enable/disable builds pass with parity intact.",
            "Shared primitives are centralized under services/shared.",
        },
        .gate_commands = &.{
            "zig build validate-flags",
            "zig build full-check",
        },
        .milestones = &.{
            "Finish AI hierarchy consolidation.",
            "Complete shared resilience extraction.",
            "Update integration imports and feature test roots.",
        },
    },
    .{
        .slug = "integration-gates-v1",
        .title = "Integration Gates v1",
        .status = .blocked,
        .owner = "Abbey",
        .scope = "Expand exhaustive integration and long-running command probes while keeping default gates fast.",
        .success_criteria = &.{
            "cli-tests-full has deterministic isolated runner behavior.",
            "Preflight clearly reports missing credentials/endpoints.",
            "Gate artifacts include per-command diagnostics and summaries.",
        },
        .gate_commands = &.{
            "zig build cli-tests-full",
            "zig build verify-all",
        },
        .milestones = &.{
            "Complete full matrix manifest coverage.",
            "Finalize PTY probe scripts and timeout policies.",
            "Improve preflight blocked-report diagnostics (env/tool/network granularity).",
            "Document required integration environment contract.",
        },
    },
};

pub const roadmap_entries = [_]RoadmapEntry{
    .{
        .id = "RM-001",
        .title = "Complete canonical roadmap/plans sync",
        .summary = "Unify task roadmap import and docs generation behind a single typed catalog.",
        .track = .docs,
        .horizon = .now,
        .status = .in_progress,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build gendocs",
            "zig build check-docs",
        },
        .plan_slug = "docs-roadmap-sync-v2",
    },
    .{
        .id = "RM-002",
        .title = "Close GPU strictness and pool lifecycle gaps",
        .summary = "Ensure explicit backend requests never silently fall back and mixed-backend pools deinit safely.",
        .track = .gpu,
        .horizon = .now,
        .status = .in_progress,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build typecheck",
            "zig build verify-all",
        },
        .plan_slug = "gpu-redesign-v3",
    },
    .{
        .id = "RM-003",
        .title = "Finalize CLI descriptor framework cutover",
        .summary = "Move remaining command routing and help/completion behavior to descriptor-first framework paths.",
        .track = .cli_tui,
        .horizon = .now,
        .status = .in_progress,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build cli-tests",
            "zig build verify-all",
        },
        .plan_slug = "cli-framework-local-agents",
    },
    .{
        .id = "RM-004",
        .title = "Finish TUI modular extraction",
        .summary = "Complete module split and shared layout behavior across launcher and live dashboards.",
        .track = .cli_tui,
        .horizon = .now,
        .status = .in_progress,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build cli-tests",
            "zig build run -- ui launch --help",
        },
        .plan_slug = "tui-modular-v2",
    },
    .{
        .id = "RM-005",
        .title = "Docs v3 pipeline baseline established",
        .summary = "Initial docs pipeline with gendocs/check-docs/api-app/wasm fallback is complete and in use.",
        .track = .docs,
        .horizon = .now,
        .status = .done,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build check-docs",
        },
        .plan_slug = "docs-roadmap-sync-v2",
    },
    .{
        .id = "RM-006",
        .title = "Automate cross-target GPU policy verification",
        .summary = "Codify policy and capability assertions for Linux, Windows, and macOS target checks.",
        .track = .platform,
        .horizon = .next,
        .status = .planned,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build -Dtarget=x86_64-linux-gnu -Dgpu-backend=auto typecheck",
            "zig build -Dtarget=x86_64-windows-gnu -Dgpu-backend=auto typecheck",
            "zig build -Dtarget=aarch64-macos -Dgpu-backend=auto typecheck",
        },
        .plan_slug = "gpu-redesign-v3",
    },
    .{
        .id = "RM-007",
        .title = "Complete exhaustive CLI integration gate",
        .summary = "Finish full command-tree PTY/probe validation with integration preflight and artifact reporting.",
        .track = .infra,
        .horizon = .next,
        .status = .blocked,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build cli-tests-full",
        },
        .plan_slug = "integration-gates-v1",
    },
    .{
        .id = "RM-008",
        .title = "Harden local-agent provider plugins",
        .summary = "Stabilize native ABI v1 and HTTP manifest plugin behavior in strict and fallback modes.",
        .track = .ai,
        .horizon = .next,
        .status = .planned,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build feature-tests",
            "zig build cli-tests",
        },
        .plan_slug = "cli-framework-local-agents",
    },
    .{
        .id = "RM-009",
        .title = "Complete feature module hierarchy cleanup",
        .summary = "Finish moving modules into stable domains and remove legacy facade drift.",
        .track = .platform,
        .horizon = .next,
        .status = .planned,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build validate-flags",
            "zig build full-check",
        },
        .plan_slug = "feature-modules-restructure-v1",
    },
    .{
        .id = "RM-010",
        .title = "Hardware acceleration research track",
        .summary = "Investigate FPGA/ASIC acceleration strategy and integration constraints for ABI workloads.",
        .track = .infra,
        .horizon = .later,
        .status = .planned,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build verify-all",
        },
        .plan_slug = "gpu-redesign-v3",
    },
    .{
        .id = "RM-011",
        .title = "Launch developer education track",
        .summary = "Define tutorials, certification pathway, and onboarding materials around ABI workflows.",
        .track = .docs,
        .horizon = .later,
        .status = .planned,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build check-docs",
        },
        .plan_slug = "docs-roadmap-sync-v2",
    },
    .{
        .id = "RM-012",
        .title = "Expand cloud function adapters",
        .summary = "Finalize adapters for Lambda, GCF, and Azure Functions with consistent deployment docs.",
        .track = .platform,
        .horizon = .later,
        .status = .planned,
        .owner = "Abbey",
        .validation_gate = &.{
            "zig build full-check",
            "zig build verify-all",
        },
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
