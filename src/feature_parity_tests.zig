//! Comptime mod/stub parity checks.
//!
//! Verifies that every feature's stub.zig exports the same set of public
//! declarations as its mod.zig. This catches parity drift at compile time
//! rather than at runtime when a feature flag is disabled.
//!
//! Run via: `zig build check-parity`

const std = @import("std");

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

// ── Feature parity assertions ────────────────────────────────────────────
// Each comptime call verifies that mod.zig and stub.zig export matching
// public API surfaces. A compile error here means the stub has drifted.

comptime {
    assertParity("ai", @import("features/ai/mod.zig"), @import("features/ai/stub.zig"));
    assertParity("analytics", @import("features/analytics/mod.zig"), @import("features/analytics/stub.zig"));
    assertParity("auth", @import("features/auth/mod.zig"), @import("features/auth/stub.zig"));
    assertParity("benchmarks", @import("features/benchmarks/mod.zig"), @import("features/benchmarks/stub.zig"));
    assertParity("cache", @import("features/cache/mod.zig"), @import("features/cache/stub.zig"));
    assertParity("cloud", @import("features/cloud/mod.zig"), @import("features/cloud/stub.zig"));
    assertParity("compute", @import("features/compute/mod.zig"), @import("features/compute/stub.zig"));
    assertParity("database", @import("features/database/mod.zig"), @import("features/database/stub.zig"));
    assertParity("desktop", @import("features/desktop/mod.zig"), @import("features/desktop/stub.zig"));
    assertParity("documents", @import("features/documents/mod.zig"), @import("features/documents/stub.zig"));
    assertParity("gateway", @import("features/gateway/mod.zig"), @import("features/gateway/stub.zig"));
    assertParity("gpu", @import("features/gpu/mod.zig"), @import("features/gpu/stub.zig"));
    assertParity("messaging", @import("features/messaging/mod.zig"), @import("features/messaging/stub.zig"));
    assertParity("mobile", @import("features/mobile/mod.zig"), @import("features/mobile/stub.zig"));
    assertParity("network", @import("features/network/mod.zig"), @import("features/network/stub.zig"));
    assertParity("observability", @import("features/observability/mod.zig"), @import("features/observability/stub.zig"));
    assertParity("pages", @import("features/observability/pages/mod.zig"), @import("features/observability/pages/stub.zig"));
    assertParity("search", @import("features/search/mod.zig"), @import("features/search/stub.zig"));
    assertParity("storage", @import("features/storage/mod.zig"), @import("features/storage/stub.zig"));
    assertParity("tui", @import("features/tui/mod.zig"), @import("features/tui/stub.zig"));
    assertParity("web", @import("features/web/mod.zig"), @import("features/web/stub.zig"));
    assertParity("lsp", @import("protocols/lsp/mod.zig"), @import("protocols/lsp/stub.zig"));
    assertParity("mcp", @import("protocols/mcp/mod.zig"), @import("protocols/mcp/stub.zig"));
    assertParity("acp", @import("protocols/acp/mod.zig"), @import("protocols/acp/stub.zig"));
    assertParity("ha", @import("protocols/ha/mod.zig"), @import("protocols/ha/stub.zig"));
    assertParity("connectors", @import("connectors/mod.zig"), @import("connectors/stub.zig"));
    assertParity("tasks", @import("tasks/mod.zig"), @import("tasks/stub.zig"));
    assertParity("inference", @import("inference/mod.zig"), @import("inference/stub.zig"));

    // ── AI sub-feature parity assertions ────────────────────────────────
    // Note: ai/aviva lacks stub.zig — skip until stub is created.
    assertParity("ai/abi", @import("features/ai/abi/mod.zig"), @import("features/ai/abi/stub.zig"));
    assertParity("ai/abbey", @import("features/ai/abbey/mod.zig"), @import("features/ai/abbey/stub.zig"));
    assertParity("ai/compliance", @import("features/ai/compliance/mod.zig"), @import("features/ai/compliance/stub.zig"));
    assertParity("ai/agents", @import("features/ai/agents/mod.zig"), @import("features/ai/agents/stub.zig"));
    assertParity("ai/constitution", @import("features/ai/constitution/mod.zig"), @import("features/ai/constitution/stub.zig"));
    assertParity("ai/context_engine", @import("features/ai/context_engine/mod.zig"), @import("features/ai/context_engine/stub.zig"));
    assertParity("ai/coordination", @import("features/ai/coordination/mod.zig"), @import("features/ai/coordination/stub.zig"));
    assertParity("ai/core", @import("features/ai/core/mod.zig"), @import("features/ai/core/stub.zig"));
    assertParity("ai/database", @import("features/ai/database/mod.zig"), @import("features/ai/database/stub.zig"));
    assertParity("ai/documents", @import("features/ai/documents/mod.zig"), @import("features/ai/documents/stub.zig"));
    assertParity("ai/embeddings", @import("features/ai/embeddings/mod.zig"), @import("features/ai/embeddings/stub.zig"));
    assertParity("ai/eval", @import("features/ai/eval/mod.zig"), @import("features/ai/eval/stub.zig"));
    assertParity("ai/feedback", @import("features/ai/feedback/mod.zig"), @import("features/ai/feedback/stub.zig"));
    assertParity("ai/explore", @import("features/ai/explore/mod.zig"), @import("features/ai/explore/stub.zig"));
    assertParity("ai/federated", @import("features/ai/federated/mod.zig"), @import("features/ai/federated/stub.zig"));
    assertParity("ai/llm", @import("features/ai/llm/mod.zig"), @import("features/ai/llm/stub.zig"));
    assertParity("ai/memory", @import("features/ai/memory/mod.zig"), @import("features/ai/memory/stub.zig"));
    assertParity("ai/models", @import("features/ai/models/mod.zig"), @import("features/ai/models/stub.zig"));
    assertParity("ai/multi_agent", @import("features/ai/multi_agent/mod.zig"), @import("features/ai/multi_agent/stub.zig"));
    assertParity("ai/orchestration", @import("features/ai/orchestration/mod.zig"), @import("features/ai/orchestration/stub.zig"));
    assertParity("ai/profile", @import("features/ai/profile/mod.zig"), @import("features/ai/profile/stub.zig"));
    assertParity("ai/profiles", @import("features/ai/profiles/mod.zig"), @import("features/ai/profiles/stub.zig"));
    assertParity("ai/prompts", @import("features/ai/prompts/mod.zig"), @import("features/ai/prompts/stub.zig"));
    assertParity("ai/rag", @import("features/ai/rag/mod.zig"), @import("features/ai/rag/stub.zig"));
    assertParity("ai/reasoning", @import("features/ai/reasoning/mod.zig"), @import("features/ai/reasoning/stub.zig"));
    assertParity("ai/streaming", @import("features/ai/streaming/mod.zig"), @import("features/ai/streaming/stub.zig"));
    assertParity("ai/templates", @import("features/ai/templates/mod.zig"), @import("features/ai/templates/stub.zig"));
    assertParity("ai/tools", @import("features/ai/tools/mod.zig"), @import("features/ai/tools/stub.zig"));
    assertParity("ai/training", @import("features/ai/training/mod.zig"), @import("features/ai/training/stub.zig"));
    assertParity("ai/transformer", @import("features/ai/transformer/mod.zig"), @import("features/ai/transformer/stub.zig"));
    assertParity("ai/vision", @import("features/ai/vision/mod.zig"), @import("features/ai/vision/stub.zig"));
}

test "mod/stub parity check compiled successfully" {
    // If we reach this test, all comptime assertions above passed.
    // The assertions run at compile time; this test exists so the
    // build system has a test entry point to depend on.
}
