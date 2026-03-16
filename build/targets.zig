const std = @import("std");

/// Descriptor for a build target (example, tool, etc.).
pub const BuildTarget = struct {
    name: []const u8,
    step_name: []const u8,
    description: []const u8,
    source_path: []const u8,
    optimize: ?std.builtin.OptimizeMode = null,
};

/// All example programs shipped with the project.
pub const example_targets = [_]BuildTarget{
    .{ .name = "example-hello", .step_name = "run-hello", .description = "Run hello example", .source_path = "examples/hello.zig" },
    .{ .name = "example-database", .step_name = "run-database", .description = "Run database example", .source_path = "examples/database.zig" },
    .{ .name = "example-llm-real", .step_name = "run-llm-real", .description = "Run real LLM via connectors (Ollama/LM Studio/vLLM)", .source_path = "examples/llm_real.zig" },
    .{ .name = "example-ha", .step_name = "run-ha", .description = "Run HA example", .source_path = "examples/ha.zig" },
    .{ .name = "example-train-demo", .step_name = "run-train-demo", .description = "Run LLM training demo", .source_path = "examples/training/train_demo.zig" },
    .{ .name = "example-orchestration", .step_name = "run-orchestration", .description = "Run multi-model orchestration example", .source_path = "examples/orchestration.zig" },
    .{ .name = "example-train-ava", .step_name = "run-train-ava", .description = "Train Ava assistant from gpt-oss", .source_path = "examples/train_ava.zig" },
    .{ .name = "example-concurrency", .step_name = "run-concurrency", .description = "Run concurrency primitives example", .source_path = "examples/concurrency.zig" },
    .{ .name = "example-observability", .step_name = "run-observability", .description = "Run observability example", .source_path = "examples/observability.zig" },
    .{ .name = "example-gpu", .step_name = "run-gpu", .description = "Run GPU example", .source_path = "examples/gpu.zig" },
    .{ .name = "example-streaming", .step_name = "run-streaming", .description = "Run streaming API example", .source_path = "examples/streaming.zig" },
    .{ .name = "example-registry", .step_name = "run-registry", .description = "Run feature registry example", .source_path = "examples/registry.zig" },
    .{ .name = "example-embeddings", .step_name = "run-embeddings", .description = "Run embeddings example", .source_path = "examples/embeddings.zig" },
    .{ .name = "example-config", .step_name = "run-config", .description = "Run configuration example", .source_path = "examples/config.zig" },
    .{ .name = "example-tensor-ops", .step_name = "run-tensor-ops", .description = "Run tensor + matrix + SIMD example", .source_path = "examples/tensor_ops.zig" },
    .{ .name = "example-concurrent-pipeline", .step_name = "run-concurrent-pipeline", .description = "Run channel + thread pool + DAG pipeline example", .source_path = "examples/concurrent_pipeline.zig" },
    .{ .name = "example-cache", .step_name = "run-cache", .description = "Run cache example", .source_path = "examples/cache.zig" },
    .{ .name = "example-search", .step_name = "run-search", .description = "Run search example", .source_path = "examples/search.zig" },
    .{ .name = "example-messaging", .step_name = "run-messaging", .description = "Run messaging example", .source_path = "examples/messaging.zig" },
    .{ .name = "example-storage", .step_name = "run-storage", .description = "Run storage example", .source_path = "examples/storage.zig" },
    .{ .name = "example-gateway", .step_name = "run-gateway", .description = "Run gateway example", .source_path = "examples/gateway.zig" },
    .{ .name = "example-pages", .step_name = "run-pages", .description = "Run pages example", .source_path = "examples/pages.zig" },
    .{ .name = "example-auth", .step_name = "run-auth", .description = "Run auth example", .source_path = "examples/auth.zig" },
    .{ .name = "example-analytics", .step_name = "run-analytics", .description = "Run analytics example", .source_path = "examples/analytics.zig" },
    .{ .name = "example-cloud", .step_name = "run-cloud", .description = "Run cloud example", .source_path = "examples/cloud.zig" },
    .{ .name = "example-web", .step_name = "run-web", .description = "Run web example", .source_path = "examples/web.zig" },
    .{ .name = "example-ai-suite", .step_name = "run-ai-suite", .description = "Run consolidated AI suite (core, inference, training, reasoning, multimodal, connectors)", .source_path = "examples/ai_suite.zig" },
    .{ .name = "example-mobile", .step_name = "run-mobile", .description = "Run mobile example", .source_path = "examples/mobile.zig" },
    .{ .name = "example-gpu-training", .step_name = "run-gpu-training", .description = "Run GPU + training integration example", .source_path = "examples/gpu_training.zig" },
    .{ .name = "example-distributed-db", .step_name = "run-distributed-db", .description = "Run distributed database integration example", .source_path = "examples/distributed_db.zig" },
    .{ .name = "example-web-observability", .step_name = "run-web-observability", .description = "Run web + observability integration example", .source_path = "examples/web_observability.zig" },
    .{ .name = "example-bare-metal-riscv32", .step_name = "run-bare-metal-riscv32", .description = "Run RISC-V 32 bare metal example", .source_path = "examples/bare_metal_riscv32.zig" },
    .{ .name = "example-bare-metal-thumb", .step_name = "run-bare-metal-thumb", .description = "Run ARM Thumb bare metal example", .source_path = "examples/bare_metal_thumb.zig" },
    .{ .name = "benchmarks", .step_name = "benchmarks", .description = "Run comprehensive benchmark suite", .source_path = "benchmarks/main.zig" },
};

// ── Cross-compilation target matrix ─────────────────────────────────────

/// Descriptor for a cross-compilation verification target.
pub const CrossTarget = struct {
    name: []const u8,
    arch: std.Target.Cpu.Arch,
    os: std.Target.Os.Tag,
    abi: std.Target.Abi = .none,
};

/// All platform targets the ABI module must compile for.
/// Used by the `cross-check` build step to verify portability.
pub const cross_check_targets = [_]CrossTarget{
    // ── Linux variants ──────────────────────────────────────────────────
    .{ .name = "linux-x86_64", .arch = .x86_64, .os = .linux, .abi = .gnu },
    .{ .name = "linux-aarch64", .arch = .aarch64, .os = .linux, .abi = .gnu },
    .{ .name = "linux-riscv64", .arch = .riscv64, .os = .linux, .abi = .gnu },
    .{ .name = "linux-arm", .arch = .arm, .os = .linux, .abi = .gnueabihf },

    // ── macOS ───────────────────────────────────────────────────────────
    .{ .name = "macos-aarch64", .arch = .aarch64, .os = .macos },
    .{ .name = "macos-x86_64", .arch = .x86_64, .os = .macos },

    // ── Windows ─────────────────────────────────────────────────────────
    .{ .name = "windows-x86_64", .arch = .x86_64, .os = .windows },
    .{ .name = "windows-aarch64", .arch = .aarch64, .os = .windows },

    // ── BSD family ──────────────────────────────────────────────────────
    .{ .name = "freebsd-x86_64", .arch = .x86_64, .os = .freebsd },
    .{ .name = "freebsd-aarch64", .arch = .aarch64, .os = .freebsd },
    .{ .name = "netbsd-x86_64", .arch = .x86_64, .os = .netbsd },
    .{ .name = "openbsd-x86_64", .arch = .x86_64, .os = .openbsd },
    .{ .name = "dragonfly-x86_64", .arch = .x86_64, .os = .dragonfly },

    // ── Mobile ──────────────────────────────────────────────────────────
    .{ .name = "ios-aarch64", .arch = .aarch64, .os = .ios },
    .{ .name = "android-aarch64", .arch = .aarch64, .os = .linux, .abi = .android },
    .{ .name = "android-arm", .arch = .arm, .os = .linux, .abi = .android },

    // ── Embedded / Bare Metal ───────────────────────────────────────────
    .{ .name = "riscv32-freestanding", .arch = .riscv32, .os = .freestanding },
    .{ .name = "thumb-freestanding", .arch = .thumb, .os = .freestanding },
    .{ .name = "aarch64-freestanding", .arch = .aarch64, .os = .freestanding },

    // ── WASM / Freestanding ─────────────────────────────────────────────
    .{ .name = "wasm32-freestanding", .arch = .wasm32, .os = .freestanding },
    .{ .name = "wasm32-wasi", .arch = .wasm32, .os = .wasi },
};

/// Check whether a path exists within the build root.
pub fn pathExists(b: *std.Build, path: []const u8) bool {
    b.build_root.handle.access(b.graph.io, path, .{}) catch return false;
    return true;
}

/// Register build steps for a table of `BuildTarget` entries.  Skips
/// targets whose source files are missing.
pub fn buildTargets(
    b: *std.Build,
    table: []const BuildTarget,
    abi_module: *std.Build.Module,
    build_opts: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    aggregate: ?*std.Build.Step,
    aggregate_runs: bool,
) void {
    for (table) |t| {
        if (!pathExists(b, t.source_path)) continue;
        const exe_optimize = t.optimize orelse optimize;
        const is_blocked_darwin = @import("builtin").os.tag == .macos and @import("builtin").os.version_range.semver.min.major >= 26;

        const exe = if (is_blocked_darwin) b.addObject(.{
            .name = t.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(t.source_path),
                .target = target,
                .optimize = exe_optimize,
                .link_libc = true,
            }),
        }) else b.addExecutable(.{
            .name = t.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(t.source_path),
                .target = target,
                .optimize = exe_optimize,
                .link_libc = true,
            }),
        });

        if (is_blocked_darwin) {
            exe.use_llvm = true;
        }
        exe.root_module.addImport("abi", abi_module);
        exe.root_module.addImport("build_options", build_opts);
        applyPerformanceTweaks(exe, exe_optimize);
        const step = b.step(t.step_name, t.description);

        const run = if (is_blocked_darwin) blk: {
            const link_mod = @import("link.zig");
            const rt_path = link_mod.findCompilerRt(b);
            const relink = b.addSystemCommand(&.{ "/usr/bin/ld", "-dynamic" });
            relink.addArg("-platform_version");
            relink.addArg("macos");
            relink.addArg("15.0");
            relink.addArg("15.0");

            const sdk_path = link_mod.detectSdkPath(b.graph.io) orelse "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk";
            relink.addArg("-syslibroot");
            relink.addArg(sdk_path);

            relink.addArg("-e");
            relink.addArg("_main");
            relink.addArg("-o");
            const bin = relink.addOutputFileArg(b.fmt("{s}_linked", .{t.name}));
            relink.addArtifactArg(exe);
            relink.addArg("-lSystem");
            if (rt_path) |path| relink.addArg(path);

            const r = std.Build.Step.Run.create(b, b.fmt("run {s} linked", .{t.name}));
            r.addFileArg(bin);
            r.step.dependOn(&relink.step);
            break :blk r;
        } else b.addRunArtifact(exe);

        if (b.args) |args| run.addArgs(args);
        step.dependOn(&run.step);

        if (!is_blocked_darwin) {
            b.installArtifact(exe);
        }

        if (aggregate) |agg| {
            if (aggregate_runs) {
                agg.dependOn(&run.step);
            } else {
                agg.dependOn(&exe.step);
            }
        }
    }
}

/// Strip release builds by default.
pub fn applyPerformanceTweaks(exe: *std.Build.Step.Compile, optimize: std.builtin.OptimizeMode) void {
    if (optimize == .ReleaseFast or optimize == .ReleaseSmall) {
        if (exe.root_module.strip == null)
            exe.root_module.strip = true;
    }
}
