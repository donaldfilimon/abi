//! Comptime mod/stub parity checks.
//!
//! Verifies that every feature's stub.zig exports the same set of public
//! declarations as its mod.zig. This catches parity drift at compile time
//! rather than at runtime when a feature flag is disabled.
//!
//! Run via: `zig build check-stub-parity`

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
    assertParity("search", @import("features/search/mod.zig"), @import("features/search/stub.zig"));
    assertParity("storage", @import("features/storage/mod.zig"), @import("features/storage/stub.zig"));
    assertParity("web", @import("features/web/mod.zig"), @import("features/web/stub.zig"));
}

test "mod/stub parity check compiled successfully" {
    // If we reach this test, all comptime assertions above passed.
    // The assertions run at compile time; this test exists so the
    // build system has a test entry point to depend on.
}
