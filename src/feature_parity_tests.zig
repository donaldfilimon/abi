//! Comptime mod/stub parity checks.
//!
//! Verifies that every feature's stub.zig exports the same set of public
//! declarations as its mod.zig. This catches parity drift at compile time
//! rather than at runtime when a feature flag is disabled.
//!
//! Tier 1: Catalog features — maps Feature enum → mod/stub types via
//!         exhaustive switch. Adding a Feature enum variant without a
//!         switch case is a compile error.
//! Tier 2: AI sub-features not yet in the catalog are listed separately.
//!
//! Run via: `zig build check-parity`

const std = @import("std");
const catalog = @import("core/feature_catalog.zig");

/// Compare public declaration names between two module types.
/// Returns a list of names present in `expected` but missing from `actual`.
fn missingDecls(comptime Expected: type, comptime Actual: type) []const []const u8 {
    @setEvalBranchQuota(1_000_000);
    const expected_decls = @typeInfo(Expected).@"struct".decls;
    const actual_decls = @typeInfo(Actual).@"struct".decls;

    comptime var missing: []const []const u8 = &.{};
    inline for (expected_decls) |decl| {
        comptime var found = false;
        inline for (actual_decls) |actual_decl| {
            if (comptime std.mem.eql(u8, decl.name, actual_decl.name)) {
                found = true;
            }
        }
        if (!found) {
            missing = missing ++ .{decl.name};
        }
    }
    return missing;
}

/// Assert that two modules have matching public declarations.
/// Produces a compile error listing missing declarations if they diverge.
fn assertParity(comptime name: []const u8, comptime Mod: type, comptime Stub: type) void {
    const mod_missing = comptime missingDecls(Mod, Stub);
    const stub_missing = comptime missingDecls(Stub, Mod);

    if (mod_missing.len > 0) {
        var msg: []const u8 = name ++ "/stub.zig is missing declarations from mod.zig:";
        for (mod_missing) |m| {
            msg = msg ++ " " ++ m;
        }
        @compileError(msg);
    }
    if (stub_missing.len > 0) {
        var msg: []const u8 = name ++ "/mod.zig is missing declarations from stub.zig:";
        for (stub_missing) |m| {
            msg = msg ++ " " ++ m;
        }
        @compileError(msg);
    }
}

// ── Catalog feature imports ──────────────────────────────────────────────
// Exhaustive switch on Feature enum: adding a new variant without a case
// here is a compile error. @import requires string literals, so we map
// each enum variant to its mod/stub types explicitly.

fn catalogMod(comptime feature: catalog.Feature) type {
    return switch (feature) {
        .gpu => @import("features/gpu/mod.zig"),
        .ai => @import("features/ai/mod.zig"),
        .llm => @import("features/ai/llm/mod.zig"),
        .embeddings => @import("features/ai/embeddings/mod.zig"),
        .agents => @import("features/ai/agents/mod.zig"),
        .training => @import("features/ai/training/mod.zig"),
        .database => @import("features/database/mod.zig"),
        .network => @import("features/network/mod.zig"),
        .observability => @import("features/observability/mod.zig"),
        .web => @import("features/web/mod.zig"),
        .profiles => @import("features/ai/profile/mod.zig"),
        .cloud => @import("features/cloud/mod.zig"),
        .analytics => @import("features/analytics/mod.zig"),
        .auth => @import("features/auth/mod.zig"),
        .messaging => @import("features/messaging/mod.zig"),
        .cache => @import("features/cache/mod.zig"),
        .storage => @import("features/storage/mod.zig"),
        .search => @import("features/search/mod.zig"),
        .mobile => @import("features/mobile/mod.zig"),
        .gateway => @import("features/gateway/mod.zig"),
        .pages => @import("features/observability/pages/mod.zig"),
        .benchmarks => @import("features/benchmarks/mod.zig"),
        .reasoning => @import("features/ai/reasoning/mod.zig"),
        .constitution => @import("features/ai/constitution/mod.zig"),
        .compute => @import("features/compute/mod.zig"),
        .documents => @import("features/documents/mod.zig"),
        .desktop => @import("features/desktop/mod.zig"),
        .tui => @import("features/tui/mod.zig"),
        .lsp => @import("protocols/lsp/mod.zig"),
        .mcp => @import("protocols/mcp/mod.zig"),
        .acp => @import("protocols/acp/mod.zig"),
        .ha => @import("protocols/ha/mod.zig"),
        .connectors => @import("connectors/mod.zig"),
        .tasks => @import("tasks/mod.zig"),
        .inference => @import("inference/mod.zig"),
    };
}

fn catalogStub(comptime feature: catalog.Feature) type {
    return switch (feature) {
        .gpu => @import("features/gpu/stub.zig"),
        .ai => @import("features/ai/stub.zig"),
        .llm => @import("features/ai/llm/stub.zig"),
        .embeddings => @import("features/ai/embeddings/stub.zig"),
        .agents => @import("features/ai/agents/stub.zig"),
        .training => @import("features/ai/training/stub.zig"),
        .database => @import("features/database/stub.zig"),
        .network => @import("features/network/stub.zig"),
        .observability => @import("features/observability/stub.zig"),
        .web => @import("features/web/stub.zig"),
        .profiles => @import("features/ai/profile/stub.zig"),
        .cloud => @import("features/cloud/stub.zig"),
        .analytics => @import("features/analytics/stub.zig"),
        .auth => @import("features/auth/stub.zig"),
        .messaging => @import("features/messaging/stub.zig"),
        .cache => @import("features/cache/stub.zig"),
        .storage => @import("features/storage/stub.zig"),
        .search => @import("features/search/stub.zig"),
        .mobile => @import("features/mobile/stub.zig"),
        .gateway => @import("features/gateway/stub.zig"),
        .pages => @import("features/observability/pages/stub.zig"),
        .benchmarks => @import("features/benchmarks/stub.zig"),
        .reasoning => @import("features/ai/reasoning/stub.zig"),
        .constitution => @import("features/ai/constitution/stub.zig"),
        .compute => @import("features/compute/stub.zig"),
        .documents => @import("features/documents/stub.zig"),
        .desktop => @import("features/desktop/stub.zig"),
        .tui => @import("features/tui/stub.zig"),
        .lsp => @import("protocols/lsp/stub.zig"),
        .mcp => @import("protocols/mcp/stub.zig"),
        .acp => @import("protocols/acp/stub.zig"),
        .ha => @import("protocols/ha/stub.zig"),
        .connectors => @import("connectors/stub.zig"),
        .tasks => @import("tasks/stub.zig"),
        .inference => @import("inference/stub.zig"),
    };
}

// ── Parity assertions ────────────────────────────────────────────────────

comptime {
    // Tier 1: All catalog features (35 entries).
    // The exhaustive switch in catalogMod/catalogStub ensures new Feature
    // enum variants get a compile error until their import is added.
    for (catalog.all) |entry| {
        assertParity(@tagName(entry.feature), catalogMod(entry.feature), catalogStub(entry.feature));
    }

    // Tier 2: AI sub-features not yet in the catalog (24 entries).
    // Note: ai/aviva lacks stub.zig — skip until stub is created.
    assertParity("ai/abi", @import("features/ai/abi/mod.zig"), @import("features/ai/abi/stub.zig"));
    assertParity("ai/abbey", @import("features/ai/abbey/mod.zig"), @import("features/ai/abbey/stub.zig"));
    assertParity("ai/compliance", @import("features/ai/compliance/mod.zig"), @import("features/ai/compliance/stub.zig"));
    assertParity("ai/context_engine", @import("features/ai/context_engine/mod.zig"), @import("features/ai/context_engine/stub.zig"));
    assertParity("ai/coordination", @import("features/ai/coordination/mod.zig"), @import("features/ai/coordination/stub.zig"));
    assertParity("ai/core", @import("features/ai/core/mod.zig"), @import("features/ai/core/stub.zig"));
    assertParity("ai/database", @import("features/ai/database/mod.zig"), @import("features/ai/database/stub.zig"));
    assertParity("ai/documents", @import("features/ai/documents/mod.zig"), @import("features/ai/documents/stub.zig"));
    assertParity("ai/eval", @import("features/ai/eval/mod.zig"), @import("features/ai/eval/stub.zig"));
    assertParity("ai/explore", @import("features/ai/explore/mod.zig"), @import("features/ai/explore/stub.zig"));
    assertParity("ai/feedback", @import("features/ai/feedback/mod.zig"), @import("features/ai/feedback/stub.zig"));
    assertParity("ai/federated", @import("features/ai/federated/mod.zig"), @import("features/ai/federated/stub.zig"));
    assertParity("ai/memory", @import("features/ai/memory/mod.zig"), @import("features/ai/memory/stub.zig"));
    assertParity("ai/models", @import("features/ai/models/mod.zig"), @import("features/ai/models/stub.zig"));
    assertParity("ai/multi_agent", @import("features/ai/multi_agent/mod.zig"), @import("features/ai/multi_agent/stub.zig"));
    assertParity("ai/orchestration", @import("features/ai/orchestration/mod.zig"), @import("features/ai/orchestration/stub.zig"));
    assertParity("ai/profiles", @import("features/ai/profiles/mod.zig"), @import("features/ai/profiles/stub.zig"));
    assertParity("ai/prompts", @import("features/ai/prompts/mod.zig"), @import("features/ai/prompts/stub.zig"));
    assertParity("ai/rag", @import("features/ai/rag/mod.zig"), @import("features/ai/rag/stub.zig"));
    assertParity("ai/streaming", @import("features/ai/streaming/mod.zig"), @import("features/ai/streaming/stub.zig"));
    assertParity("ai/templates", @import("features/ai/templates/mod.zig"), @import("features/ai/templates/stub.zig"));
    assertParity("ai/tools", @import("features/ai/tools/mod.zig"), @import("features/ai/tools/stub.zig"));
    assertParity("ai/transformer", @import("features/ai/transformer/mod.zig"), @import("features/ai/transformer/stub.zig"));
    assertParity("ai/vision", @import("features/ai/vision/mod.zig"), @import("features/ai/vision/stub.zig"));
}

test "mod/stub parity check compiled successfully" {
    // If we reach this test, all comptime assertions above passed.
    // The assertions run at compile time; this test exists so the
    // build system has a test entry point to depend on.
}
