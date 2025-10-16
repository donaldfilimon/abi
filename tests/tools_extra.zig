const std = @import("std");
const testing = std.testing;

/// Helper to obtain a general purpose allocator for tests.
fn getTestAllocator() std.mem.Allocator {
    return testing.allocator;
}

// -----------------------------------------------------------------------------
// Advanced code analyzer test (basic compile check)
// -----------------------------------------------------------------------------
test "AdvancedCodeAnalyzer: import and init" {
    const adv = @import("../tools/advanced_code_analyzer.zig");
    const allocator = getTestAllocator();

    if (@hasDecl(adv, "init")) {
        const instance = try adv.init(allocator);
        defer if (@hasDecl(instance, "deinit")) instance.deinit();

        if (@hasDecl(instance, "analyze")) {
            const rpt = try instance.analyze("");
            try testing.expect(rpt.issues.len == 0);
        }
    }
}

// -----------------------------------------------------------------------------
// Static analysis tool test (basic compile check)
// -----------------------------------------------------------------------------
test "StaticAnalysis: import and run" {
    const sa = @import("../tools/static_analysis.zig");
    const allocator = getTestAllocator();

    if (@hasDecl(sa, "Analyzer")) {
        const analyzer = try sa.Analyzer.init(allocator);
        defer if (@hasDecl(analyzer, "deinit")) analyzer.deinit();

        if (@hasDecl(analyzer, "run")) {
            const result = try analyzer.run("");
            try testing.expect(result.errors.len == 0);
        }
    } else if (@hasDecl(sa, "run")) {
        const result = try sa.run("");
        try testing.expect(result.errors.len == 0);
    }
}

// -----------------------------------------------------------------------------
// Docs generator basic test (ensure generation does not panic)
// -----------------------------------------------------------------------------
test "DocsGenerator: generate without panic" {
    const dg = @import("../tools/docs_generator.zig");
    const allocator = getTestAllocator();

    if (@hasDecl(dg, "generate")) {
        try dg.generate(allocator);
    }
}

// -----------------------------------------------------------------------------
// Generate API docs test (basic compile check)
// -----------------------------------------------------------------------------
test "GenerateAPIDocs: import and call" {
    const api = @import("../tools/generate_api_docs.zig");
    const allocator = getTestAllocator();

    if (@hasDecl(api, "generate")) {
        try api.generate(allocator);
    }
}

// -----------------------------------------------------------------------------
// Generate index test (basic compile check)
// -----------------------------------------------------------------------------
test "GenerateIndex: import and execute" {
    const idx = @import("../tools/generate_index.zig");
    const allocator = getTestAllocator();

    if (@hasDecl(idx, "run")) {
        try idx.run(allocator);
    }
}

// -----------------------------------------------------------------------------
// HTTP smoke test – simple compile check, no network needed
// -----------------------------------------------------------------------------
test "HTTPSmoke: import and init" {
    const http = @import("../tools/http_smoke.zig");
    const allocator = getTestAllocator();

    if (@hasDecl(http, "init")) {
        const ctx = try http.init(allocator);
        defer if (@hasDecl(ctx, "deinit")) ctx.deinit();
    }
}

// -----------------------------------------------------------------------------
// Interactive CLI test – ensure main entry can be called
// -----------------------------------------------------------------------------
test "InteractiveCLI: import and run" {
    const icli = @import("../tools/interactive_cli.zig");
    const allocator = getTestAllocator();

    if (@hasDecl(icli, "run")) {
        // Pass an empty argument list
        try icli.run(allocator, &[_][]const u8{});
    }
}

// -----------------------------------------------------------------------------
// Perf guard basic test
// -----------------------------------------------------------------------------
test "PerfGuard: import and guard" {
    const pg = @import("../tools/perf_guard.zig");

    if (@hasDecl(pg, "guard")) {
        const g = pg.guard();
        defer g.deinit();
    }
}

// -----------------------------------------------------------------------------
// Performance CI test – compile only
// -----------------------------------------------------------------------------
test "PerformanceCI: import" {
    const pci = @import("../tools/performance_ci.zig");
    _ = pci; // suppress unused warning
}

// -----------------------------------------------------------------------------
// Performance profiler basic test
// -----------------------------------------------------------------------------
test "PerformanceProfiler: import and start" {
    const pp = @import("../tools/performance_profiler.zig");

    if (@hasDecl(pp, "start")) {
        const handle = pp.start();
        defer pp.stop(handle);
    }
}

// -----------------------------------------------------------------------------
// Stress test basic compile check
// -----------------------------------------------------------------------------
test "StressTest: import and execute" {
    const st = @import("../tools/stress_test.zig");
    const allocator = getTestAllocator();

    if (@hasDecl(st, "run")) {
        try st.run(allocator);
    }
}

// -----------------------------------------------------------------------------
// Windows network test – compile only (skip on non‑Windows)
// -----------------------------------------------------------------------------
test "WindowsNetworkTest: import (non‑Windows guard)" {
    const wnt = @import("../tools/windows_network_test.zig");
    _ = wnt; // ensure the module compiles on any platform
}
