//! ABI build root — Zig 0.16, self-contained.
const std = @import("std");
const build_flags = @import("build/flags.zig");
const build_cross = @import("build/cross.zig");
const build_linking = @import("build/linking.zig");
const build_validation = @import("build/validation.zig");
const FeatureFlags = build_flags.FeatureFlags;
const hasBackend = build_flags.hasBackend;
const addAllBuildOptions = build_flags.addAllBuildOptions;
const linkDarwinArtifact = build_linking.linkDarwinArtifact;
const linkIfDarwin = build_linking.linkIfDarwin;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Feature flags ───────────────────────────────────────────────────
    const feat_gpu = b.option(bool, "feat-gpu", "GPU compute backends") orelse true;
    const feat_ai = b.option(bool, "feat-ai", "AI services") orelse true;
    const feat_database = b.option(bool, "feat-database", "Vector database") orelse true;
    const feat_network = b.option(bool, "feat-network", "Networking / Raft") orelse true;
    const feat_observability_opt = b.option(bool, "feat-observability", "Observability (metrics, tracing, profiling)");
    const feat_profiling_opt = b.option(bool, "feat-profiling", "Deprecated alias for feat-observability");
    const feat_web = b.option(bool, "feat-web", "Web framework") orelse true;
    const feat_pages = b.option(bool, "feat-pages", "Dashboard pages") orelse true;
    const feat_analytics = b.option(bool, "feat-analytics", "Analytics") orelse true;
    const feat_cloud = b.option(bool, "feat-cloud", "Cloud integration") orelse true;
    const feat_auth = b.option(bool, "feat-auth", "Authentication") orelse true;
    const feat_messaging = b.option(bool, "feat-messaging", "Messaging / pub-sub") orelse true;
    const feat_cache = b.option(bool, "feat-cache", "Caching") orelse true;
    const feat_storage = b.option(bool, "feat-storage", "Storage backends") orelse true;
    const feat_search = b.option(bool, "feat-search", "Full-text search") orelse true;
    const feat_mobile = b.option(bool, "feat-mobile", "Mobile (iOS/Android)") orelse false;
    const feat_gateway = b.option(bool, "feat-gateway", "API gateway") orelse true;
    const feat_benchmarks = b.option(bool, "feat-benchmarks", "Benchmark suites") orelse true;
    const feat_compute = b.option(bool, "feat-compute", "Distributed compute") orelse true;
    const feat_documents = b.option(bool, "feat-documents", "Document processing") orelse true;
    const feat_desktop = b.option(bool, "feat-desktop", "Desktop integration") orelse true;
    const feat_tui = b.option(bool, "feat-tui", "Terminal user interface") orelse false;
    if (feat_observability_opt != null and feat_profiling_opt != null and feat_observability_opt.? != feat_profiling_opt.?) {
        std.log.err("Conflicting feature flags: -Dfeat-observability={} and -Dfeat-profiling={}", .{
            feat_observability_opt.?,
            feat_profiling_opt.?,
        });
        std.process.exit(1);
    }
    const feat_observability = feat_observability_opt orelse feat_profiling_opt orelse true;
    if (feat_profiling_opt != null and feat_observability_opt == null) {
        std.log.warn("feat-profiling is deprecated; use feat-observability", .{});
    }

    // AI sub-feature flags
    const feat_llm = b.option(bool, "feat-llm", "LLM inference") orelse feat_ai;
    const feat_training = b.option(bool, "feat-training", "Model training") orelse feat_ai;
    const feat_vision = b.option(bool, "feat-vision", "Vision models") orelse feat_ai;
    const feat_explore = b.option(bool, "feat-explore", "AI exploration") orelse feat_ai;
    const feat_reasoning = b.option(bool, "feat-reasoning", "Reasoning engine") orelse feat_ai;

    // Protocol flags
    const feat_lsp = b.option(bool, "feat-lsp", "Language Server Protocol") orelse true;
    const feat_mcp = b.option(bool, "feat-mcp", "Model Context Protocol") orelse true;
    const feat_acp = b.option(bool, "feat-acp", "Agent Communication Protocol") orelse true;
    const feat_ha = b.option(bool, "feat-ha", "High Availability / replication") orelse true;
    const feat_connectors = b.option(bool, "feat-connectors", "External service connectors") orelse true;
    const feat_tasks = b.option(bool, "feat-tasks", "Task management") orelse true;
    const feat_inference = b.option(bool, "feat-inference", "ML inference engine") orelse true;

    // GPU backend flags
    const gpu_backend_str = b.option([]const u8, "gpu-backend", "GPU backends: metal,cuda,vulkan,webgpu,opengl,opengles,webgl2,stdgpu,fpga,tpu (comma-separated)");
    const gpu_metal = feat_gpu and hasBackend(gpu_backend_str, "metal");
    const gpu_cuda = feat_gpu and hasBackend(gpu_backend_str, "cuda");
    const gpu_vulkan = feat_gpu and hasBackend(gpu_backend_str, "vulkan");
    const gpu_webgpu = feat_gpu and hasBackend(gpu_backend_str, "webgpu");
    const gpu_opengl = feat_gpu and hasBackend(gpu_backend_str, "opengl");
    const gpu_opengles = feat_gpu and hasBackend(gpu_backend_str, "opengles");
    const gpu_webgl2 = feat_gpu and hasBackend(gpu_backend_str, "webgl2");
    const gpu_stdgpu = feat_gpu and (gpu_backend_str == null or hasBackend(gpu_backend_str, "stdgpu"));
    const gpu_fpga = feat_gpu and hasBackend(gpu_backend_str, "fpga");
    const gpu_tpu = feat_gpu and hasBackend(gpu_backend_str, "tpu");

    // ── GPU backend validation ──────────────────────────────────────────
    if (gpu_backend_str) |str| {
        const valid_backends: []const []const u8 = &.{
            "metal",    "cuda",   "vulkan", "webgpu", "opengl",
            "opengles", "webgl2", "stdgpu", "fpga",   "tpu",
        };
        var backend_it = std.mem.splitScalar(u8, str, ',');
        while (backend_it.next()) |part| {
            const trimmed = std.mem.trim(u8, part, " ");
            if (trimmed.len == 0) continue;
            var found = false;
            for (valid_backends) |vb| {
                if (std.mem.eql(u8, trimmed, vb)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std.log.err("Unknown GPU backend '{s}'. Valid: metal, cuda, vulkan, webgpu, opengl, opengles, webgl2, stdgpu, fpga, tpu", .{trimmed});
                std.process.exit(1);
            }
        }
    }
    if (gpu_metal and gpu_cuda)
        std.log.warn("Both Metal and CUDA enabled — unusual; intended for cross-compilation only", .{});

    // ── Feature dependency warnings ─────────────────────────────────────
    if (!feat_ai) {
        if (feat_llm) std.log.warn("feat_llm requires feat_ai — llm will be stubbed", .{});
        if (feat_training) std.log.warn("feat_training requires feat_ai — training will be stubbed", .{});
        if (feat_vision) std.log.warn("feat_vision requires feat_ai — vision will be stubbed", .{});
        if (feat_explore) std.log.warn("feat_explore requires feat_ai — explore will be stubbed", .{});
        if (feat_reasoning) std.log.warn("feat_reasoning requires feat_ai — reasoning will be stubbed", .{});
    }
    if (feat_mcp and !feat_database)
        std.log.info("feat_mcp benefits from feat_database for DB tools", .{});

    // ── Build options module ────────────────────────────────────────────
    const flags = FeatureFlags{
        .feat_gpu = feat_gpu,
        .feat_ai = feat_ai,
        .feat_database = feat_database,
        .feat_network = feat_network,
        .feat_observability = feat_observability,
        .feat_web = feat_web,
        .feat_pages = feat_pages,
        .feat_analytics = feat_analytics,
        .feat_cloud = feat_cloud,
        .feat_auth = feat_auth,
        .feat_messaging = feat_messaging,
        .feat_cache = feat_cache,
        .feat_storage = feat_storage,
        .feat_search = feat_search,
        .feat_mobile = feat_mobile,
        .feat_gateway = feat_gateway,
        .feat_benchmarks = feat_benchmarks,
        .feat_compute = feat_compute,
        .feat_documents = feat_documents,
        .feat_desktop = feat_desktop,
        .feat_tui = feat_tui,
        .feat_llm = feat_llm,
        .feat_training = feat_training,
        .feat_vision = feat_vision,
        .feat_explore = feat_explore,
        .feat_reasoning = feat_reasoning,
        .feat_lsp = feat_lsp,
        .feat_mcp = feat_mcp,
        .feat_acp = feat_acp,
        .feat_ha = feat_ha,
        .feat_connectors = feat_connectors,
        .feat_tasks = feat_tasks,
        .feat_inference = feat_inference,
        .gpu_metal = gpu_metal,
        .gpu_cuda = gpu_cuda,
        .gpu_vulkan = gpu_vulkan,
        .gpu_webgpu = gpu_webgpu,
        .gpu_opengl = gpu_opengl,
        .gpu_opengles = gpu_opengles,
        .gpu_webgl2 = gpu_webgl2,
        .gpu_stdgpu = gpu_stdgpu,
        .gpu_fpga = gpu_fpga,
        .gpu_tpu = gpu_tpu,
    };

    const build_opts = b.addOptions();
    const pkg_version = comptime blk: {
        const zon = @import("build.zig.zon");
        break :blk zon.version;
    };
    addAllBuildOptions(build_opts, flags, pkg_version);

    const build_options_module = build_opts.createModule();
    _ = build_cross.addSteps(.{
        .b = b,
        .target = target,
        .optimize = optimize,
        .build_options_module = build_options_module,
        .package_version = pkg_version,
    });

    // ── ABI library module ──────────────────────────────────────────────
    const abi_module = b.addModule("abi", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    abi_module.addImport("build_options", build_options_module);

    // ── Static library ──────────────────────────────────────────────────
    const static_lib = b.addLibrary(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    static_lib.root_module.addImport("build_options", build_options_module);
    linkIfDarwin(static_lib, .static_lib, feat_gpu, gpu_metal);
    b.installArtifact(static_lib);
    b.step("lib", "Build static library").dependOn(&b.addInstallArtifact(static_lib, .{}).step);

    // ── MCP server binary ────────────────────────────────────────────────
    const mcp_exe = b.addExecutable(.{
        .name = "abi-mcp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mcp_main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    mcp_exe.root_module.addImport("build_options", build_options_module);
    linkIfDarwin(mcp_exe, .executable, feat_gpu, gpu_metal);
    b.step("mcp", "Build MCP stdio server").dependOn(&b.addInstallArtifact(mcp_exe, .{}).step);

    // ── CLI binary ──────────────────────────────────────────────────────
    const cli_exe = b.addExecutable(.{
        .name = "abi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    cli_exe.root_module.addImport("build_options", build_options_module);
    linkIfDarwin(cli_exe, .executable, feat_gpu, gpu_metal);
    b.step("cli", "Build ABI command-line interface").dependOn(&b.addInstallArtifact(cli_exe, .{}).step);

    _ = build_validation.addSteps(.{
        .b = b,
        .target = target,
        .optimize = optimize,
        .flags = flags,
        .build_options_module = build_options_module,
        .abi_module = abi_module,
        .package_version = pkg_version,
    });

    // ── Lint / format ───────────────────────────────────────────────────
    const fmt_paths = &.{ "build.zig", "build", "src", "test" };
    b.step("lint", "Check formatting").dependOn(&b.addFmt(.{ .paths = fmt_paths, .check = true }).step);
    b.step("fix", "Fix formatting").dependOn(&b.addFmt(.{ .paths = fmt_paths, .check = false }).step);

    // ── Doctor step ────────────────────────────────────────────────────
    const doctor_step = b.step("doctor", "Report build configuration and diagnostics");
    const doc1 = b.addSystemCommand(&.{
        "echo",
        b.fmt(
            \\ABI Build Configuration Report
            \\==============================
            \\Features:
            \\  feat_ai={} feat_gpu={} feat_database={} feat_network={}
            \\  feat_observability={} feat_web={} feat_pages={} feat_analytics={}
            \\  feat_cloud={} feat_auth={} feat_messaging={} feat_cache={}
            \\  feat_storage={} feat_search={} feat_mobile={} feat_gateway={}
            \\  feat_benchmarks={} feat_compute={} feat_documents={} feat_desktop={}
        , .{
            feat_ai,            feat_gpu,     feat_database,  feat_network,
            feat_observability, feat_web,     feat_pages,     feat_analytics,
            feat_cloud,         feat_auth,    feat_messaging, feat_cache,
            feat_storage,       feat_search,  feat_mobile,    feat_gateway,
            feat_benchmarks,    feat_compute, feat_documents, feat_desktop,
        }),
    });
    const doc2 = b.addSystemCommand(&.{
        "echo",
        b.fmt(
            \\AI Sub-features:
            \\  feat_llm={} feat_training={} feat_vision={} feat_explore={} feat_reasoning={}
            \\Protocols:
            \\  feat_lsp={} feat_mcp={} feat_acp={} feat_ha={}
            \\GPU Backends:
            \\  metal={} cuda={} vulkan={} webgpu={} opengl={}
            \\  opengles={} webgl2={} stdgpu={} fpga={} tpu={}
        , .{
            feat_llm,   feat_training, feat_vision, feat_explore, feat_reasoning,
            feat_lsp,   feat_mcp,      feat_acp,    feat_ha,      gpu_metal,
            gpu_cuda,   gpu_vulkan,    gpu_webgpu,  gpu_opengl,   gpu_opengles,
            gpu_webgl2, gpu_stdgpu,    gpu_fpga,    gpu_tpu,
        }),
    });
    doc2.step.dependOn(&doc1.step);
    doctor_step.dependOn(&doc2.step);
}
