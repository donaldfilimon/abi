const std = @import("std");
const build_policy = @import("build_gpu_policy");
const runtime_policy = @import("runtime_gpu_policy");

const TargetCase = struct {
    name: []const u8,
    os: std.Target.Os.Tag,
    abi: std.Target.Abi,
};

const target_cases = [_]TargetCase{
    .{ .name = "macos", .os = .macos, .abi = .none },
    .{ .name = "linux", .os = .linux, .abi = .none },
    .{ .name = "windows", .os = .windows, .abi = .none },
    .{ .name = "ios", .os = .ios, .abi = .none },
    .{ .name = "tvos", .os = .tvos, .abi = .none },
    .{ .name = "android", .os = .linux, .abi = .android },
    .{ .name = "wasi/web", .os = .wasi, .abi = .none },
    .{ .name = "emscripten/web", .os = .emscripten, .abi = .none },
    .{ .name = "freebsd", .os = .freebsd, .abi = .none },
    .{ .name = "netbsd", .os = .netbsd, .abi = .none },
    .{ .name = "openbsd", .os = .openbsd, .abi = .none },
    .{ .name = "dragonfly", .os = .dragonfly, .abi = .none },
    .{ .name = "haiku", .os = .haiku, .abi = .none },
    .{ .name = "illumos", .os = .illumos, .abi = .none },
    .{ .name = "freestanding", .os = .freestanding, .abi = .none },
};

pub fn main(_: std.process.Init) !void {
    var mismatches: usize = 0;

    for (target_cases) |target_case| {
        checkCase(target_case, &mismatches);
    }

    if (mismatches > 0) {
        std.debug.print(
            "ERROR: GPU policy consistency checks failed ({d} mismatch(es)).\n",
            .{mismatches},
        );
        std.process.exit(1);
    }

    std.debug.print("OK: GPU build/runtime policy consistency checks passed\n", .{});
}

fn checkCase(target_case: TargetCase, mismatches: *usize) void {
    const build_class = build_policy.classify(target_case.os, target_case.abi);
    const runtime_class = runtime_policy.classify(target_case.os, target_case.abi);

    if (!std.mem.eql(u8, @tagName(build_class), @tagName(runtime_class))) {
        mismatch(
            mismatches,
            "{s}: classify mismatch (build={t}, runtime={t})\n",
            .{ target_case.name, build_class, runtime_class },
        );
    }

    const build_order = build_policy.defaultOrder(build_class);
    const runtime_order = runtime_policy.defaultOrder(runtime_class);
    if (!sameStringSlice(build_order, runtime_order)) {
        mismatch(
            mismatches,
            "{s}: default order mismatch\n",
            .{target_case.name},
        );
    }

    const build_hints = build_policy.optimizationHintsForPlatform(build_class);
    const runtime_hints = runtime_policy.optimizationHintsForPlatform(runtime_class);
    if (!sameHints(build_hints, runtime_hints)) {
        mismatch(
            mismatches,
            "{s}: optimization hints mismatch\n",
            .{target_case.name},
        );
    }
}

fn sameStringSlice(a: []const []const u8, b: []const []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |lhs, rhs| {
        if (!std.mem.eql(u8, lhs, rhs)) return false;
    }
    return true;
}

fn sameHints(a: build_policy.OptimizationHints, b: runtime_policy.OptimizationHints) bool {
    return a.default_local_size == b.default_local_size and
        a.default_queue_depth == b.default_queue_depth and
        a.prefer_unified_memory == b.prefer_unified_memory and
        a.prefer_pinned_staging == b.prefer_pinned_staging and
        a.transfer_chunk_bytes == b.transfer_chunk_bytes;
}

fn mismatch(mismatches: *usize, comptime fmt: []const u8, args: anytype) void {
    mismatches.* += 1;
    std.debug.print("MISMATCH: " ++ fmt, args);
}
